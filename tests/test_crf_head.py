"""Unit tests for the CrfHead"""
import torch
from torchcondirf import CrfHead

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
LOG_TRANSITIONS = torch.tensor([[-1e4, -1e4, -1e4], [-1e4, 5, 4], [-1e4, 2, 3]])


torch.set_grad_enabled(False)
torch.manual_seed(1)

def test_scores_computed_correctly(EXPECTED_SCORES_DF, crf_head):
    for _, row in EXPECTED_SCORES_DF.iterrows():
        predicted_score = crf_head(
            LOG_EMISSIONS[row.example_index : row.example_index + 1, : row.length],
            lengths=LENGTHS[row.example_index : row.example_index + 1],
            tags=torch.tensor([row.tag_sequence]),
        )["logits"].item()
        import pdb; pdb.set_trace()
        assert row.score == predicted_score

def test_scores_computed_correctly_in_batch(EXPECTED_SCORES_DF, crf_head):
    all_emissions = torch.stack(
        [LOG_EMISSIONS[0]] * 16 + [LOG_EMISSIONS[1]] * 4 + [LOG_EMISSIONS[2]] * 2
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
    )
    
    predicted_score = predicted_score["logits"]
    assert torch.allclose(
        predicted_score, torch.tensor(EXPECTED_SCORES_DF["score"], dtype=torch.float)
    )
