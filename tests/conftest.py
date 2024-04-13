"""Artifacts to use throughout tests."""
import pandas as pd
import pytest
import torch
from torchcondirf import CrfHead

NUM_TAGS = 3  # 2 actual tags (1, 2) with 0 for pad
BATCH_SIZE = 3
LENGTHS = torch.LongTensor([4, 2, 1])

# emissions of shape (3, 4, 3)
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
# transitions of shape (3, 3), nothing transitions to/from padding tag 0
LOG_TRANSITIONS = torch.tensor([[-1e4, -1e4, -1e4], [-1e4, 5, 4], [-1e4, 2, 3]])

@pytest.fixture
def EXPECTED_SCORES_DF():
    num_tags = 3
    batch_size = 3
    lengths = [4, 2, 1]
    expected_scores = pd.Series(
        {
            # first example of length 4
            "1111": 32,
            "1112": 33,
            "1121": 27,
            "1122": 30,
            "1211": 29,
            "1212": 30,
            "1221": 26,
            "1222": 29,
            "2111": 30,
            "2112": 31,
            "2121": 25,
            "2122": 28,
            "2211": 29,
            "2212": 30,
            "2221": 26,
            "2222": 29,
            # second example of length 2
            "11": 10,
            "12": 13,
            "21": 12,
            "22": 17,
            # third example of length 1
            "1": 15,
            "2": 23,
        }
    ).sort_values(ascending=False)

    expected_scores.name = "score"
    expected_scores = (
        expected_scores.reset_index()
        .rename(columns={"index": "tag_sequence"})
        .apply(
            lambda row: pd.Series(
                {"tag_sequence": [int(tag) for tag in row.tag_sequence], "score": row.score}
            ),
            axis=1,
        )
    )
    expected_scores["length"] = expected_scores.apply(
        lambda row: len(row["tag_sequence"]), axis=1
    )
    expected_scores = expected_scores.sort_values(
        ["length", "score"], ascending=[False, False], ignore_index=True
    )

    expected_scores["example_index"] = sum(
        [[i] * ((num_tags - 1) ** lengths[i]) for i in range(batch_size)], []
    )
    return expected_scores

@pytest.fixture
def crf_head():
    crf_head = CrfHead(num_tags=NUM_TAGS, include_start_end_transitions=False)
    # disable grad to set desired transitions
    crf_head.log_transitions.requires_grad_(False)
    crf_head.log_transitions.data = LOG_TRANSITIONS
    crf_head.log_transitions.requires_grad_()
    return crf_head
