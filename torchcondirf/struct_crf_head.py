"""
Linear chain conditional random fields using TorchStruct library that performs
message passing on context free grammars via inside-out algorithms where the inside
pass is performed via back-prop.

This is an alternative to the CrfHead with the same API but based on 
Sasha Rush's torch_struct library.
"""
from typing import List, Tuple, Union, Optional

import torch
from torch_struct import LinearChainCRF

import util
from base_crf_head import BaseCrfHead

class StructCrfHead(BaseCrfHead):
    def __init__(
        self,
        *,
        num_tags: int,
        allowed_transitions: List[Tuple[int, int]] = None,
        include_start_end_transitions: bool = True,
        padding_tag_id: int = 0,
    ) -> None:

        super().__init__(
            num_tags=num_tags,
            include_start_end_transitions=include_start_end_transitions,
            padding_tag_id=padding_tag_id,
            allowed_transitions=allowed_transitions,
        )

    def forward(
        self,
        log_emissions: torch.Tensor,
        lengths: torch.LongTensor,
        *,
        tags: torch.Tensor = None,
        mask: torch.BoolTensor = None,
        **_,
    ) -> torch.Tensor:
        # TODO: Add option to deal with different loss functions, e.g. GTI/Dice/Tversky.
        distribution = self._build_distribution(log_emissions, lengths, mask)
        result = {"log_partition": distribution.partition}
        logits = None
        if tags is not None:

            # * Get the labels in `parts` using torch_struct functionality
            # * result is a (batch, seq_length, num_tags, num_tags) tensor, in which
            # * each (num_tags, num_tags) matrix is one-hot
            labels_parts = distribution.struct.to_parts(
                tags,
                self.num_tags,
                lengths=lengths,  # sequence_lengths,
            ).type_as(distribution.log_potentials)

            logits = distribution._struct().score(
                distribution.log_potentials, labels_parts, batch_dims=[0]
            )

        result["logits"] = logits
        return result

    def compute_log_potentials(
        self,
        log_emissions: torch.Tensor,
        lengths: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Given input features (log probabilities of emission for each tag in the
        tag space) it will create a tensor holding the log of the emission probabilities
        over `parts` (edges in the chain) and add the transitions to such `parts`.
        Returns: tensor in appropriate form for torch_struct inference.
        """
        # ? TODO: here we consider from end of first edge as we view the chain as a right
        # ? branching tree. Should we be using first edge and end at second last edge?
        batch_size, max_len, *_ = log_emissions.size()
        log_emissions = self._scale_and_mask_log_emissions(log_emissions, lengths, mask)
        # * lp[i, t, j, k] = le[i, t + 1, j] + lt[j, k]
        log_potentials = log_emissions.view(batch_size, max_len, self.num_tags, 1)[
            :, 1:
        ] + self.log_transitions.view(1, 1, self.num_tags, self.num_tags)

        # * lp[i, 0, j, k] += le[i, 0, k]
        log_potentials[:, 0] += log_emissions[:, 0].view(batch_size, 1, self.num_tags)

        batch_indexer = torch.arange(
            batch_size, dtype=torch.long, device=log_emissions.device
        )
        # * lp[i, 0, j, k] += st[k]
        log_potentials[:, 0, :, :] += self.start_transitions.view(1, 1, -1)
        # * lp[i, l_i - 2, j, k] += et[j]
        log_potentials[batch_indexer, lengths - 2, :, :] += self.end_transitions.view(
            1, -1, 1
        )
        return log_potentials

    def _build_distribution(self, log_emissions, lengths, mask=None):

        log_potentials = self.compute_log_potentials(log_emissions, lengths, mask=mask)
        # the lengths required by the LinearChainCrf distribution are
        # the sequence lengths, not the parts lengths.
        # See: https://github.com/harvardnlp/pytorch-struct/blob/7146de5659ff17ad7be53023c025ffd099866412/torch_struct/linearchain.py#L33
        distribution = LinearChainCRF(log_potentials, lengths=lengths)
        return distribution

    def sample_paths():
        ...

    def viterbi_algorithm(
        self,
        log_emissions: torch.Tensor,
        lengths: torch.LongTensor,
        *,
        mask: Optional[torch.BoolTensor] = None,
        top_k: int = 1,
        **_,
    ) -> Union[List[Tuple], List[List[Tuple]]]:

        distribution = self._build_distribution(log_emissions, lengths, mask)

        # * TODO: align outputs. Maybe don't do the outputs processing in the functions.
        # * This will give advantage of not having to pass the flag
        if top_k > 1:
            return n_best(
                distribution,
                top_k=top_k,
                lengths=lengths,
            )
        return viterbi(
            distribution,
            lengths=lengths,
        )

    def get_point_marginals(self, *, log_emissions, lengths, mask=None):
        distribution = self._build_distribution(log_emissions, lengths, mask)
        parts_marginals = distribution.marginals
        # * at position 0, sum the marginals of all parts coming out of a given tag
        marginals_at_start = parts_marginals[:, :1, :, :].sum(-2)
        # * at other positions sum the marginals of all parts going into a given tag
        remaining_marginals = parts_marginals.sum(-1)
        return torch.cat([marginals_at_start, remaining_marginals], dim=1)


def n_best(distribution, *, top_k, lengths, padding_tag_id=0):
    # TODO: fix doc-strings and add more comments
    """
    Runs max-sum algorithm (max-product in log space) to find the `top+k` most
    likely sequences of tags given the inputs.
    Returns:
        (List[List[[Tuple[torch.LongTensor, float]]]):
    """
    # * Compute the k best configuration of `parts`.
    # * This will be a k x batch x (max_len - 1) x num_tags x num_tags
    k_argmax = distribution.topk(top_k)

    # * compute each sequences' best configuration's score
    # * This will be a k x batch
    top_k_scores = distribution.kmax(top_k)

    # * Scores are sufficient if we just want to rank.
    # TODO: Not sure the normalisation works as intended)
    decoded_tags_probabilities = top_k_scores

    k_best_paths = []
    for i in range(top_k):
        ith_argmax = k_argmax[i]
        # * map from `parts configuration space` to `tag configuration space`
        decoded_tags, _ = distribution.struct.from_parts(ith_argmax)

        decoded_tags = _postprocess_start_end_tags(
            decoded_tags, lengths, padding_tag_id=padding_tag_id
        )

        k_best_paths.append(decoded_tags)

    # * decoded_tags_probabilities is k x batch. Make it batch x k for
    # * torch distributed to work without any arg changes
    decoded_tags_probabilities = decoded_tags_probabilities.T
    # * k_best_paths stacked return a batch x k x max_len.
    return torch.stack(k_best_paths, 1), decoded_tags_probabilities


def viterbi(distribution, *, lengths, padding_tag_id=0):
    # TODO: fix doc-strings and add more comments
    """
    Runs max-sum algorithm (max-product in log space) to find the most likely
    sequence of tags given the inputs.
    Returns:
        (List[Tuple[torch.LongTensor, float]]): Viterbi sequence and respective
            probability, one for each element in the batch. Thus list of
            length = batch_size.
    """
    # * compute the best configuration of `parts`
    argmax = distribution.argmax
    # * map from `parts configuration space` to `tag configuration space`
    decoded_tags, _ = distribution.struct.from_parts(argmax)

    # * compute each sequences' best configuration's score
    max_scores = distribution.max
    # partition = distribution.partition

    # * Scores are sufficient if we just want to rank.
    # * normalise to obtain probability (NB we're in log-space)
    decoded_tags_probabilities = max_scores  # - partition

    # * slice off the start tag if we have start and end tags included
    decoded_tags = _postprocess_start_end_tags(
        decoded_tags, lengths, padding_tag_id=padding_tag_id
    )
    # Unsqueeze the top_k dimension; The tensors are now of shapes:
    # (batch_size, top_k=1, max_length) and (batch_size, top_k=1)
    return decoded_tags.unsqueeze(1), decoded_tags_probabilities.unsqueeze(1)


def _postprocess_start_end_tags(decoded_tags, lengths, padding_tag_id=0):
    """
    Sets all tags beyond the orignal sequence length (i.e. from end_tag onwards) to
    `padding_tag_id`.
    """
    mask = util.get_mask_from_sequence_lengths(lengths, decoded_tags.size(1))
    decoded_tags.masked_fill_(~mask.to(decoded_tags.device), padding_tag_id)

    return decoded_tags
