"""Unit tests for the CrfHead"""

import torch
from torchcondirf import CrfHead
import pytest


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
                assert model_prediction in score_to_tag_seqences[model_score]


NUM_TAGS = 3  # 2 actual tags (1, 2) with 0 for pad
BATCH_SIZE = 3
LENGTHS = torch.LongTensor([4, 2, 1])

# emissions of shape
LOG_EMISSIONS = torch.Tensor(
    [
        [[1, 2, 3], [5, 6, 7], [12, 11, 10], [-1, -2, 0]],
        [
            [1, 4, 9],
            [5, 1, 5],
            [12, 121, 140],  # irrelevant, length is 2
            [-1, -200, 0],  # irrelevant, lenght is 2
        ],
        [
            [1, 15, 23],
            [5000, 3211, 2123],  # irrelevant, lenght is 1
            [12, 121, 140],  # irrelevant, length is 1
            [-1, -200, 0],  # irrelevant, lenght is 1
        ],
    ]
)
LOG_TRANSITIONS = torch.tensor([[-1e4, -1e4, -1e4], [-1e4, 5, 2], [-1e4, 4, 3]])


torch.set_grad_enabled(False)
torch.manual_seed(1)


@pytest.mark.parametrize(
    "crf_head_class, start_end_transitions, log_emissions_scaling",
    [
        (CrfHead, None, 0.0),
        (CrfHead, None, 11.3),
        (CrfHead, (torch.tensor([12.0, 1.0, 4.0]), torch.tensor([2.0, 3.0, 6.0])), 0.0),
    ],
)
def test_scores_computed_correctly(
    crf_head_class,
    start_end_transitions,
    log_emissions_scaling,
    EXPECTED_SCORES_DF,
):
    crf_head, log_emissions = instantiate_crf_head_for_test(
        crf_head_class,
        NUM_TAGS,
        LOG_EMISSIONS,
        LENGTHS,
        LOG_TRANSITIONS,
        start_end_transitions,
        log_emissions_scaling,
    )
    for _, row in EXPECTED_SCORES_DF.iterrows():
        predicted_score = crf_head(
            log_emissions[row.example_index : row.example_index + 1, : row.length],
            lengths=LENGTHS[row.example_index : row.example_index + 1],
            tags=torch.tensor([row.tag_sequence]),
        )["logits"].item()
        assert row.score == predicted_score


@pytest.mark.parametrize(
    "crf_head_class, start_end_transitions, log_emissions_scaling",
    [
        (CrfHead, None, 0.0),
        (CrfHead, None, 11.3),
        (CrfHead, (torch.tensor([12.0, 1.0, 4.0]), torch.tensor([2.0, 3.0, 6.0])), 0.0),
        (CrfHead, (torch.tensor([12.0, 1.0, 4.0]), torch.tensor([2.0, 3.0, 6.0])), 2.7),
    ],
)
def test_scores_computed_correctly_in_batch(
    crf_head_class,
    start_end_transitions,
    log_emissions_scaling,
    EXPECTED_SCORES_DF,
):
    crf_head, log_emissions = instantiate_crf_head_for_test(
        crf_head_class,
        NUM_TAGS,
        LOG_EMISSIONS,
        LENGTHS,
        LOG_TRANSITIONS,
        start_end_transitions,
        log_emissions_scaling,
    )
    all_emissions = torch.stack(
        [log_emissions[0]] * 16 + [log_emissions[1]] * 4 + [log_emissions[2]] * 2
    )
    all_tags = torch.vstack(
        [torch.tensor(t) for t in EXPECTED_SCORES_DF["tag_sequence"][:16].values]
        + [
            torch.tensor(t + [0, 0])
            for t in EXPECTED_SCORES_DF["tag_sequence"][16:20].values
        ]
        + [
            torch.tensor(t + [0, 0, 0])
            for t in EXPECTED_SCORES_DF["tag_sequence"][20:].values
        ]
    )
    predicted_score = crf_head(
        all_emissions,
        lengths=torch.LongTensor([4] * 16 + [2] * 4 + [1] * 2),
        tags=all_tags,
    )["logits"]
    assert torch.allclose(
        predicted_score, torch.tensor(EXPECTED_SCORES_DF["score"], dtype=torch.float)
    )


# def test_viterbi(
#     crf_head_class,
#     start_end_transitions,
#     log_emissions_scaling,
#     mask_constraints,
#     top_k,
#     EXPECTED_SCORES_DF,
# ):
#     pass