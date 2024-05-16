"""Unit tests for the CrfHead"""

import torch
from torchcondirf import CrfHead, util
import pytest
import numpy as np
from itertools import groupby
from tests.helpers import (
    instantiate_crf_head_for_test,
    verify_viterbi_predictions,
    compute_log_partition_by_hand,
    compute_marginal_by_hand,
)


torch.set_grad_enabled(False)
torch.manual_seed(1)


@pytest.mark.parametrize(
    "start_end_transitions",
    [None, (torch.tensor([12.0, 1.0, 4.0]), torch.tensor([2.0, 3.0, 6.0]))],
)
@pytest.mark.parametrize("log_emissions_scaling", [0.0, 11.3, 2.71])
def test_scores_computed_correctly(
    start_end_transitions,
    log_emissions_scaling,
    EXPECTED_SCORES_DF,
    NUM_TAGS,
    LOG_EMISSIONS,
    LENGTHS,
    LOG_TRANSITIONS,
):
    crf_head, log_emissions = instantiate_crf_head_for_test(
        CrfHead,
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
        np.testing.assert_approx_equal(row.score, predicted_score)


@pytest.mark.parametrize(
    "start_end_transitions",
    [None, (torch.tensor([12.0, 1.0, 4.0]), torch.tensor([2.0, 3.0, 6.0]))],
)
@pytest.mark.parametrize("log_emissions_scaling", [0.0, 11.3, 2.7])
def test_scores_computed_correctly_in_batch(
    start_end_transitions,
    log_emissions_scaling,
    EXPECTED_SCORES_DF,
    NUM_TAGS,
    LOG_EMISSIONS,
    LENGTHS,
    LOG_TRANSITIONS,
):
    crf_head, log_emissions = instantiate_crf_head_for_test(
        CrfHead,
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


@pytest.mark.parametrize(
    "start_end_transitions",
    [None, (torch.tensor([12.0, 1.0, 4.0]), torch.tensor([2.0, 3.0, 6.0]))],
)
@pytest.mark.parametrize("log_emissions_scaling", [0.0, 11.3, 2.7])
@pytest.mark.parametrize(
    "tag_constraints",
    [
        None,
        [[(2, 3, [1])], [(1, 2, [1])], []],
        [
            [(1, 3, 1), (3, 4, 2)],
            [(2, 3, 2)],
            [],
        ],
    ],
)
@pytest.mark.parametrize("top_k", [1, 2, 10])
def test_viterbi(
    start_end_transitions,
    log_emissions_scaling,
    tag_constraints,
    top_k,
    EXPECTED_SCORES_DF,
    NUM_TAGS,
    LOG_EMISSIONS,
    LENGTHS,
    LOG_TRANSITIONS,
):
    crf_head, log_emissions = instantiate_crf_head_for_test(
        CrfHead,
        NUM_TAGS,
        LOG_EMISSIONS,
        LENGTHS,
        LOG_TRANSITIONS,
        start_end_transitions,
        log_emissions_scaling,
    )
    length_mask = util.get_mask_from_sequence_lengths(LENGTHS, LENGTHS.max())
    mask = util.get_mask_for_tags(length_mask, NUM_TAGS, tag_constraints)
    viterbi_topk_predictions = crf_head.viterbi_algorithm(
        log_emissions=log_emissions, lengths=LENGTHS, mask=mask, top_k=top_k
    )
    verify_viterbi_predictions(
        EXPECTED_SCORES_DF,
        top_k,
        viterbi_topk_predictions,
        mask,
        LENGTHS,
        padding_tag_id=0,
    )


@pytest.mark.parametrize(
    "start_end_transitions",
    [None, (torch.tensor([12.0, 1.0, 4.0]), torch.tensor([2.0, 3.0, 6.0]))],
)
@pytest.mark.parametrize("log_emissions_scaling", [0.0, 11.3, 2.7])
@pytest.mark.parametrize(
    "tag_constraints",
    [
        None,
        [[(2, 3, [1])], [(1, 2, [1])], []],
        [
            [(1, 3, [1]), (3, 4, [2])],
            [(1, 2, [2])],
            [],
        ],
    ],
)
@pytest.mark.parametrize("backward", [False, True])
def test_partition_function(
    start_end_transitions,
    log_emissions_scaling,
    tag_constraints,
    backward,
    EXPECTED_SCORES_DF,
    NUM_TAGS,
    LOG_EMISSIONS,
    LENGTHS,
    LOG_TRANSITIONS,
):
    crf_head, log_emissions = instantiate_crf_head_for_test(
        CrfHead,
        NUM_TAGS,
        LOG_EMISSIONS,
        LENGTHS,
        LOG_TRANSITIONS,
        start_end_transitions,
        log_emissions_scaling,
    )
    length_mask = util.get_mask_from_sequence_lengths(LENGTHS, LENGTHS.max())
    mask = util.get_mask_for_tags(length_mask, NUM_TAGS, tag_constraints)

    out = crf_head(
        log_emissions=log_emissions,
        lengths=LENGTHS,
        mask=mask,
        compute_log_beta=backward,
    )
    constrained_log_partition_by_hand = torch.stack(
        compute_log_partition_by_hand(
            EXPECTED_SCORES_DF, constraints=tag_constraints
        ).tolist()
    )
    torch.testing.assert_close(
        constrained_log_partition_by_hand,
        out["log_partition"] if not backward else out["backwards_partition"],
    )


@pytest.mark.parametrize(
    "start_end_transitions",
    [None, (torch.tensor([12.0, 1.0, 4.0]), torch.tensor([2.0, 3.0, 6.0]))],
)
@pytest.mark.parametrize("log_emissions_scaling", [0.0, 11.3, 2.7])
def test_weighted_loglik_agrees_with_loglik_when_weights_are_one(
    start_end_transitions,
    log_emissions_scaling,
    EXPECTED_SCORES_DF,
    NUM_TAGS,
    LOG_EMISSIONS,
    LENGTHS,
    LOG_TRANSITIONS,
):
    crf_head, log_emissions = instantiate_crf_head_for_test(
        CrfHead,
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
    all_lengths = torch.LongTensor([4] * 16 + [2] * 4 + [1] * 2)
    out = crf_head(
        all_emissions,
        lengths=all_lengths,
        tags=all_tags,
    )
    log_likelihoods = crf_head.log_likelihood(
        logits=out["logits"],
        log_partition=out["log_partition"],
    )
    weighted_log_likelihoods = crf_head.weighted_log_likelihood(
        log_emissions=all_emissions,
        lengths=all_lengths,
        tags=all_tags,
        log_alpha=out["log_alpha"],
        weights=torch.tensor([1, 1, 1]),
    )
    assert torch.allclose(log_likelihoods, weighted_log_likelihoods)


@pytest.mark.parametrize(
    "start_end_transitions",
    [None, (torch.tensor([12.0, 1.0, 4.0]), torch.tensor([2.0, 3.0, 6.0]))],
)
@pytest.mark.parametrize("log_emissions_scaling", [0.0, 11.3, 2.7])
@pytest.mark.parametrize(
    "tag_constraints",
    [
        None,
        [[(2, 3, [1])], [(1, 2, [1])], []],
        [
            [(1, 3, [1]), (3, 4, [2])],
            [(1, 2, [2])],
            [],
        ],
    ],
)
def test_marginals(
    start_end_transitions,
    log_emissions_scaling,
    tag_constraints,
    EXPECTED_SCORES_DF,
    NUM_TAGS,
    LOG_EMISSIONS,
    LENGTHS,
    LOG_TRANSITIONS,
):
    crf_head, log_emissions = instantiate_crf_head_for_test(
        CrfHead,
        NUM_TAGS,
        LOG_EMISSIONS,
        LENGTHS,
        LOG_TRANSITIONS,
        start_end_transitions,
        log_emissions_scaling,
    )
    length_mask = util.get_mask_from_sequence_lengths(LENGTHS, LENGTHS.max())
    mask = util.get_mask_for_tags(length_mask, NUM_TAGS, tag_constraints)

    out = crf_head(
        log_emissions=log_emissions,
        lengths=LENGTHS,
        mask=mask,
        compute_log_beta=True,
    )
    log_marginals = crf_head.get_point_log_marginals(
        log_emissions=log_emissions,
        log_alpha=out["log_alpha"],
        log_beta=out["log_beta"],
        log_partition=out["log_partition"],
    )
    # compute all marginals by hand
    marginals_by_hand = []
    for example_index in sorted(EXPECTED_SCORES_DF["example_index"].unique()):
        seq_length = EXPECTED_SCORES_DF.query("example_index == @example_index")[
            "length"
        ].tolist()[0]
        marginals_by_hand_for_example = torch.zeros(seq_length, 2)
        for position in range(seq_length):
            for j, tag in enumerate([1, 2]):
                marginals_by_hand_for_example[position, j] = compute_marginal_by_hand(
                    EXPECTED_SCORES_DF,
                    position=position,
                    tag=tag,
                    example_index=example_index,
                    constraints=tag_constraints,
                )
        marginals_by_hand.append(marginals_by_hand_for_example)
    for i in range(log_emissions.shape[0]):
        # when we have constraints, some log marginals are "very negative"
        # to get equality, exponentiate to make the marginals 0
        torch.testing.assert_close(
            torch.exp(marginals_by_hand[i]), 
            torch.exp(log_marginals[i, : LENGTHS[i], 1:])
        )
