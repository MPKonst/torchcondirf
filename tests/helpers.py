"""Helpers for tests"""

import torch
import numpy as np
from itertools import groupby
import pandas as pd
from torchcondirf.base_crf_head import BaseCrfHead


def groupby_slices(labels, key=None):
    group_lengths = []
    values = []
    for value, group in groupby(labels, key):
        group_lengths.append(sum(1 for _ in group))
        values.append(value)
    ends = np.cumsum(group_lengths)
    starts = np.r_[0, np.roll(ends, 1)[1:]]

    return [*zip(starts, ends, values)]


def instantiate_crf_head_for_test(
    crf_head_class,
    num_tags,
    log_emissions,
    lengths,
    log_transitions,
    start_end_transitions=None,
    log_emissions_scaling=0.0,
):
    include_start_end_transitions = start_end_transitions is not None
    crf_head = crf_head_class(
        num_tags=num_tags, include_start_end_transitions=include_start_end_transitions
    )
    crf_head.log_transitions.data = log_transitions
    crf_head.log_emissions_scaling.data = torch.tensor(log_emissions_scaling)
    log_emissions = log_emissions.clone()
    if include_start_end_transitions:
        start_transitions, end_transitions = start_end_transitions
        crf_head.start_transitions.data = start_transitions
        crf_head.end_transitions.data = end_transitions
        log_emissions[:, 0] -= start_transitions[None, :]
        for i, length in enumerate(lengths):
            log_emissions[i, length - 1] -= end_transitions
    log_emissions = log_emissions * torch.exp(-crf_head.log_emissions_scaling)
    return crf_head, log_emissions


def verify_viterbi_predictions(
    expected_scores,
    top_k,
    viterbi_nbest_predictions,
    mask_for_tags,
    lengths,
    padding_tag_id=0,
):
    # mask out the padding tag, to be able to correctly
    # calculate the max number of meaningful sequences
    mask_for_tags = mask_for_tags.clone()
    mask_for_tags[..., padding_tag_id] = False
    max_variations_per_example = mask_for_tags.sum(2)
    max_variations_per_example[max_variations_per_example == 0] = 1
    max_variations_per_example = max_variations_per_example.prod(1)
    num_examples = expected_scores["example_index"].max() + 1
    for i in range(num_examples):
        hand_predictions_for_example = expected_scores[
            expected_scores["example_index"] == i
        ]
        score_to_tag_seqences = {
            score: group["tag_sequence"].tolist()
            for score, group in hand_predictions_for_example.groupby("score")
        }
        for j in range(top_k):
            if (
                j < max_variations_per_example[i]
            ):  # the maximum number of tag-sequences for the given example
                model_prediction = viterbi_nbest_predictions[0][i, j].tolist()[
                    : lengths[i]
                ]
                model_score = viterbi_nbest_predictions[1][i, j].item()
                # test that the score is close to one of the existing ones
                try:
                    score_key = next(
                        score
                        for score in score_to_tag_seqences
                        if np.isclose(model_score, score)
                    )
                except StopIteration:
                    raise AssertionError(
                        f"Viterbi sequence score {model_score} is not valid."
                    )
                assert model_prediction in score_to_tag_seqences[score_key]


def convert_constraints(batch_tag_observations):
    """
    Converts partial observations into the format for tag constraints,
    namely (start, end, allowed_tag)
    """
    constraints = []
    for i, tag_observations in enumerate(batch_tag_observations):
        grouped_tags = groupby_slices(tag_observations)
        constraints.append(
            [(start, end, [tag]) for start, end, tag in grouped_tags if tag != -1]
        )
    return constraints


def get_mask_for_scores_by_hand(
    scores_by_hand: pd.DataFrame, constraints: list[list[tuple]]
):
    if constraints is None:
        return pd.Series(True, index=scores_by_hand.index)

    def row_satisfies_constraint(row):
        example_index = row["example_index"]
        constraints_for_element = constraints[example_index]
        tag_sequence = row["tag_sequence"]
        return all(
            tag_sequence[i] in tags
            for start, stop, tags in constraints_for_element
            for i in range(start, stop)
        )

    return scores_by_hand.apply(row_satisfies_constraint, axis=1)


def compute_log_partition_by_hand(scores_by_hand, constraints=None):
    mask_for_constraints = get_mask_for_scores_by_hand(scores_by_hand, constraints)
    scores_for_constrained_sequences = scores_by_hand.loc[mask_for_constraints]

    def log_partition_for_group(group):
        return torch.logsumexp(
            torch.tensor(group["score"].values, dtype=torch.float), dim=0
        )

    return scores_for_constrained_sequences.groupby("example_index").apply(
        log_partition_for_group, include_groups=False
    )


def compute_marginal_by_hand(
    scores_by_hand,
    position,
    tag,
    example_index,
    constraints=None,
):
    log_partition_by_hand = compute_log_partition_by_hand(scores_by_hand, constraints)
    mask_for_constraints = get_mask_for_scores_by_hand(scores_by_hand, constraints)
    scores_for_constrained_sequences = scores_by_hand.loc[mask_for_constraints]
    scores_for_example = scores_for_constrained_sequences.query(
        "example_index == @example_index"
    )
    mask_for_sequences_with_correct_tag = np.array(
        [seq[position] == tag for seq in scores_for_example["tag_sequence"]]
    )
    if not mask_for_sequences_with_correct_tag.any():
        return BaseCrfHead.VERY_NEGATIVE_VALUE
    scores_for_sequences_with_correct_tag = torch.tensor(
        scores_for_example.loc[mask_for_sequences_with_correct_tag]["score"].values,
        dtype=torch.float,
    )
    return (
        torch.logsumexp(scores_for_sequences_with_correct_tag, dim=0)
        - log_partition_by_hand[example_index]
    )
