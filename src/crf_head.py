"""
A CRF implementation which implements fast constrained and batched versions
of the forward, backward and Viterbi algorithms.
"""
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn

from eignet.torchcrf import util


class CrfHead(BaseCrfHead):
    """
    A Conditional random field nn.Module which can act as a predictive head for any
    encoder.
    """

    def forward(
        self,
        log_emissions: torch.Tensor,
        lengths: torch.LongTensor,
        *,
        tags: Optional[torch.LongTensor] = None,
        mask: Optional[torch.BoolTensor] = None,
        compute_log_beta: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the partition functions for a batch of examples.

        If a corresponding set of tag-sequences is provided, computes the logit for
        these tags as well.

        Args:
            log_emissions (torch.Tensor): a tensor of size
                (batch_size, max_sequence_length, num_tags) containing the logs of the
                emission factors, evaluated at each position of each sequence and for
                each tag
            lengths (torch.Tensor): a tensor of size (batch_size,) holding the
                integer lengths of the sequences in the batch
            tags (Optional[torch.LongTensor], optional): A tensor of shape
                (batch_size, max_sequence_length) holding tag ids for each sequence.
                If provided, the logit for the tag-sequences will be computed.
                Defaults to None.
            mask (Optional[torch.BoolTensor], optional): a tensor of size either
                (batch_size, max_sequence_length), if no tag-pinning is applied or
                (batch_size, max_sequence_length, num_tags) if tag-pinning is applied.
                That is, if there is no tag-pinning, then
                mask[i, j] = True iff j < lengths[i].
                If tag-pinning is_applied, then mask[i, j, l] = True
                if and only if j < lengths[i] and
                position j in sequence i is allowed to assume tag l. Defaults to None.
            compute_log_beta (bool, optional): Whether to also run the
                "backward algorithm" and compute the "log beta" tensor. Useful for
                computing position-level marginals but will roughly double the time
                for running this method. Defaults to False.

        Returns:
            (Dict[str, torch.Tensor]): a dictionary with keys
                ["log_partition", "log_alpha", "logits", "log_beta"] holding the
                respective tensors. The value against "logits" will be None, if
                `tags` are not provided and the value against "log_beta" will be None,
                if `compute_log_beta` is False.
        """
        logits = None
        if tags is not None:
            padding_mask = tags != self.padding_tag_id
            logits = self._calculate_logits(log_emissions, tags, padding_mask, lengths)
        else:
            padding_mask = util.get_mask_from_sequence_lengths(
                # do not pass lengths.max() to max_length, otherwise
                # this may not work with data-parallel training
                # (when the batch gets sharded, log_emissions.size(1) can be larger
                # than any of the lengths in the same shard of the batch)
                lengths,
                log_emissions.size(1),
            )

        mask = mask if mask is not None else padding_mask
        log_alpha, log_partition = self._forward_algorithm(log_emissions, lengths, mask)
        log_beta = None
        if compute_log_beta:
            log_beta, backwards_partition = self._backward_algorithm(
                log_emissions, lengths, mask
            )
            assert torch.allclose(
                log_partition, backwards_partition, rtol=1e-4, atol=1e-5, equal_nan=True
            )

        return {
            "log_partition": log_partition,
            "log_alpha": log_alpha,
            "logits": logits,
            "log_beta": log_beta,
            "backwards_partition": backwards_partition if compute_log_beta else None,
        }

    def weighted_log_likelihood(
        self,
        *,
        log_emissions: torch.Tensor,
        tags: torch.Tensor,
        lengths: torch.Tensor,
        log_alpha: torch.Tensor,
        weights: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Implements a calculation of a "log_likelihood" which allows for weighting
        the contribution of each tag-prediction. The idea is taken directly from
        Lennoy et al, "Weighted Conditional Random Fields for Supervised
        Interpatient Heartbeat Classification".
        Args:
            log_emissions (torch.Tensor): a tensor of shape
                (batch_size, max_sequence_length, num_tags)
                representing the log emission scores for each tag
            tags (torch.Tensor): a tensor of shape (batch_size, max_sequence_length)
                representing the true tags for each element in the sequence
            lengths (torch.Tensor): a tensor of shape (batch_size,) representing the
                lengths of each sequence in the batch
            log_alpha (torch.Tensor): a tensor of shape
                (batch_size, sequence_length, num_tags)
                representing the log alpha values computed by the forward
                algorithm for each tag
            weights (torch.Tensor): a tensor of shape (num_tags,) representing the
                weight for each tag
        Returns:
            (torch.Tensor): a tensor of shape (batch_size,) representing the
                weighted log likelihood of the supplied tags
        """
        # weights_per_position[i, t] = weights[tags[i, t]]
        weights_per_position = weights[tags]
        padding_mask = util.get_mask_from_sequence_lengths(
            # do not pass lengths.max() to max_length, otherwise
            # this may not work with multi-gpu training
            # (when the batch gets sharded, log_emissions.size(1) can be larger
            # than any of the lengths in the same shard of the batch)
            lengths,
            log_emissions.size(1),
        )

        normalized_log_alpha = log_alpha.clone()
        normalized_log_alpha[:, 1:, :] = normalized_log_alpha[
            :, 1:, :
        ] - util.logsumexp(normalized_log_alpha[:, :-1, :], dim=-1, keepdim=True)

        batch_size, max_sequence_length, num_tags = log_emissions.size()
        log_emissions = self._scale_and_mask_log_emissions(
            log_emissions, lengths, padding_mask
        )
        device = log_emissions.device
        starts = log_emissions[
            torch.arange(batch_size, dtype=torch.long, device=device), 0, tags[:, 0]
        ].type(torch.float)
        if self.include_start_end_transitions:
            starts += self.start_transitions[tags[:, 0]]
        assert starts.size() == (batch_size,)

        # the formula for the middle transitions is
        # MT_it = log_transitions[tags[i, t + 1], tags[i, t]]
        middle_transitions = self.log_transitions[tags[:, 1:], tags[:, :-1]]
        assert middle_transitions.size() == (batch_size, max_sequence_length - 1)

        # the formula for the middle emissions is
        # ME_it = log_emissions[i, t + 1, tags[i, t + 1]]
        middle_emissions = log_emissions[
            torch.arange(batch_size, dtype=torch.long, device=device).view(
                batch_size, 1
            ),
            torch.arange(1, max_sequence_length, dtype=torch.long, device=device).view(
                1, max_sequence_length - 1
            ),
            tags[:, 1:],
        ]
        assert middle_emissions.size() == (batch_size, max_sequence_length - 1)

        weighted_loglik = (
            starts - util.logsumexp(normalized_log_alpha[:, 0], dim=-1)
        ) * weights_per_position[:, 0]
        weighted_loglik += (
            (
                (middle_transitions + middle_emissions)
                - util.logsumexp(normalized_log_alpha[:, 1:], dim=-1)
            )
            * weights_per_position[:, 1:]
            * padding_mask[:, 1:]  # mask before sum
        ).sum(dim=1)

        if self.include_start_end_transitions:
            final_weights = util.get_final_values(weights_per_position, lengths=lengths)
            final_tags = util.get_final_values(tags, lengths=lengths)
            normalized_last_log_alphas = util.get_final_values(
                normalized_log_alpha, lengths
            )
            weighted_loglik += final_weights * (
                self.end_transitions[final_tags]
                + util.logsumexp(normalized_last_log_alphas)
                - util.logsumexp(
                    normalized_last_log_alphas
                    + self.end_transitions.view(1, self.num_tags)
                )
            )

        assert weighted_loglik.size() == (batch_size,)
        return weighted_loglik

    def get_point_log_marginals(
        self,
        *,
        log_emissions: torch.Tensor,
        log_alpha: torch.Tensor,
        log_beta: torch.Tensor,
        log_partition: torch.Tensor,
        **_,
    ):
        """
        Computes the log-marginals for each tag at each position of each sequence in
        the batch.
        Args:
            log_emissions (torch.Tensor): a tensor of shape
                (batch_size, max_sequence_length, num_tags) computed by
                _calculate_log_emissions
            log_alpha (torch.Tensor): a tensor of shape
                (batch_size, max_sequence_length, num_tags) computed by
                _forward_algorithm
            log_beta: (torch.Tensor): a tensor of shape
                (batch_size, max_sequence_length, num_tags) computed by
                _backward_algorithm
            log_partition (torch.Tensor): a tensor of shape (batch_size,) holding the
                log partition function computed for each sequence in the batch

        Returns:
            (torch.Tensor): a tensor of shape
                (max_sequence_length, batch_size, num_tags) holding the log-marginals
        """
        # make log_partition have shape (batch_size, 1, 1)
        log_partition = log_partition.view(-1, 1, 1)

        return (
            log_alpha + log_beta - log_emissions * torch.exp(self.log_emissions_scaling)
        ) - log_partition

    def viterbi_algorithm(
        self,
        log_emissions: torch.Tensor,
        lengths: torch.LongTensor,
        *,
        mask: Optional[torch.BoolTensor] = None,
        top_k=1,
        min_prediction_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the max-sum algorithm for computing the log-score of the top_k best
        sequences and then backtracks to recover the sequences themselves.
        Computation is done in batch.

        The implementation also supports constrained max-sum which
        computes the highest score over configurations with arbitrary
        constraints for positions. Thus the supplied mask can have size
        (batch_size, max_sequence_length, num_tags), allowing to pin any position of
        any example to any subset of tags.

        Args:
            log_emissions (torch.Tensor): a tensor of size
                (batch_size, max_sequence_length, num_tags) containing the logs of the
                emission factors, evaluated at each position of each sequence and for
                each tag
            lengths (torch.Tensor): a tensor of size (batch_size,) holding the
                integer lengths of the sequences in the batch
            mask (torch.BoolTensor): a tensor of size either
                (batch_size, max_sequence_length), if no tag-pinning is applied or
                (batch_size, max_sequence_length, num_tags) if tag-pinning is applied.
                That is, if there is no tag-pinning, then
                mask[i, j] = True iff j < lengths[i].
                If tag-pinning is_applied, then mask[i, j, l] = True
                iff j < lengths[i] and
                position j in sequence i is allowed to assume tag l.
            top_k (int): the number of highest-scoring sequences to retrieve
            min_prediction_length (int, optional): if an integer, each prediction in
                the batch that's shorter than the specified length will be
                padded to max(min_prediction_length, length of longest prediction)
                This is useful when doing multi GPU training,
                where batches get split and recombined.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: a 2-tuple of tensors, holding
                respectively the predictions and the logits. More precisly, 
                the tuple consists of 
                1. a (batch_size, top_k, max_len) tensor holding the 
                   top_k sequences of tags for sequence each sequence in the
                   batch (padded with the pad_token_id),
                2. a (batch_size, top_k) tensor holding the logits of each  
                   sequence
        """
        batch_size, max_sequence_length, num_tags = log_emissions.size()
        device = log_emissions.device
        topk_function = util.get_topk_function(k=top_k, dim=-1)

        log_emissions = (
            self._scale_and_mask_log_emissions(log_emissions, lengths, mask)
            .transpose(0, 1)
            .contiguous()
        )
        # Instantiate the topk_log_scores tensor which will hold the (logs of) the top k
        # highest scores, i.e.
        # topk_log_scores(t, i, l, j) = log(
        #       j-th highest score of a sequence of
        #       first t tags starting
        #       with START and ending with tag l
        #       for sequence i in the batch
        #   )
        topk_log_scores = torch.empty(
            max_sequence_length, batch_size, num_tags, top_k, device=device
        )

        # Define a "ranktag" to be a number in {0, 1, ..., (top_k * num_tags) - 1}.
        # A ranktag m encodes a:
        #   rank = m % top_k in {0, 1, ..., top_k} and a
        #   tag = m // top_k in  {0, 1, ..., num_tags}.
        # We now define a tensor topk_backpointers of
        # size (max_sequence_length - 1, batch_size, num_tags, top_k), which will be
        # populated by ranktags and will satisfy:
        # topk_backpointers[t, i, l,k] = m means that
        #   for sequence i in the batch,
        #   the k-th best-scoring
        #   tag-sequence of length t + 2
        #   with tag l at position t + 1, (the last position, since 0-based indexing)
        #   is prefixed by
        #   the (m % top_k)-th best-scoring tag-sequence of length t + 1
        #   with tag (m // top_k) at positions t.

        # Visually:
        # (m % top_k)-th highest scoring sequence of length t + 1 ending in (m // top_k)
        #        _______^________
        #       /                \
        #       . . . (m // top_k) l .
        #       \___________________/
        #         k-th highest-scoring sequence of length t + 2 ending in tag l
        topk_backpointers = torch.empty(
            max_sequence_length - 1,
            batch_size,
            num_tags,
            top_k,
            device=device,
            dtype=torch.long,
        )
        # One thing to note: the above description makes sense only when the sequence
        # is at least of length 2. Indeed, since we treat the start and end positions,
        # in a special way, sequences of length 1 are automatically handled separately.
        # The topk_backpointers array will be filled with jibberish for nonsensical
        # positions (i.e. where t > lengths[i] - 2) but we don't care as
        # our backtracking algorithm takes care of that.

        # Initial topk_log_scores combines the transitions from the initial state and
        # the log_emissions for the first time step.
        starting_log_scores = log_emissions[0]
        if self.include_start_end_transitions:
            starting_log_scores += self.start_transitions.view(1, num_tags)
        topk_log_scores[0] = util.extend_dim_to_k(
            # we add a top_k dimension; however, for now it's only top_1,
            # so we fill that dimension up to top_k with very negative values
            starting_log_scores.view(batch_size, num_tags, 1),
            k=top_k,
            fill_value=self.VERY_NEGATIVE_VALUE,
            dim=-1,
        )

        for t in range(1, max_sequence_length):
            # The top k messages from the previous time step for example i and tag l are
            # calculated by the formula:
            # top_k_messages[t]{ilk} = kth_best_{l', k'}(
            #   topk_log_scores[t - 1][i, l', k']
            #   + log_transitions_{l'l}
            # ),
            # where l' ranges over all possible tags ("kth_best_{i, j..}" here means the
            # operator which picks the k-th largest value from a set which is indexed by
            # i, j, etc.
            # E.g. 1th_best_{i, j}(S_{i, j}) is the same as max_{i, j}(S_{i, j}).
            #
            # In order to take kth_best along l' and k' simultaneously, we need to
            # collapse the two dimensions (corresponding to the from-tag and the rank,
            # respectively) into one: a "ranktag" dimension. To achieve this,
            # these two dimensions need to come last in the tensor (i.e., they should
            # increment the fastest in C-order): that way .view / .reshape
            # will combine them correctly.
            previous_topk_plus_transitions = (
                topk_log_scores[t - 1].view(batch_size, 1, num_tags, top_k)
                # the transition matrix is aligned along the from-tag dimension
                + self.log_transitions.view(1, num_tags, num_tags, 1)
            )
            # We now have:
            # previous_topk_plus_transitions.shape = (
            #       batch_size x num_tags x num_tags x top_k
            #   ), where
            # previous_topk_plus_transitions[i, j, l, k] is:
            #   for sequence i in the batch,
            #   the k-th best score
            #   for a tag-sequence of length t + 1
            #   whose last two tags are [l, j]
            #   not including the emission for the tag j at position t.

            (
                topk_messages_from_previous_timestep,
                topk_ranktags_at_previous_timestep,
            ) = topk_function(
                # unify the rank and from-tag dimensions
                previous_topk_plus_transitions.reshape(batch_size, num_tags, -1)
            )

            assert (
                topk_messages_from_previous_timestep.size()
                == topk_ranktags_at_previous_timestep.size()
                == (batch_size, num_tags, top_k)
            )

            # Now we have:
            # topk_tags_at_previous_timestep[i, j, k] = m means that
            #   for sequence i in the batch,
            #   the k-th best-scoring
            #   tag-sequence of length t + 1
            #   with tag j at position t (0-based indexing!)
            #   is prefixed by the (m % top_k)-th bese scoring
            #   tag sequence of length t with
            #   has m // top_k as its tag at position t - 1
            topk_backpointers[t - 1] = topk_ranktags_at_previous_timestep

            # We now compute the topk_log_scores for position t by the formula
            # topk_log_scores[t][i, l, k] = (
            #      log_emissions[t][i, l]
            #      + topk_messages_from_previous_timestep[i, l, k]
            # Note that adding the log_emissions automatically "masks out"
            # the forbidden tags at various positions.
            topk_log_scores[t] = (
                log_emissions[t].view(batch_size, num_tags, 1)
                + topk_messages_from_previous_timestep
            )

        # We need to treat the final tag differently due to the transitions to "STOP".
        topk_final_scores = util.get_final_values(
            tensor=topk_log_scores, lengths=lengths, batch_dim=1, sequence_dim=0
        )
        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            topk_final_scores += self.end_transitions.view(1, num_tags, 1)

        assert topk_backpointers.size() == (
            max_sequence_length - 1,
            batch_size,
            num_tags,
            top_k,
        )
        assert topk_final_scores.size() == (batch_size, num_tags, top_k)
        # topk_final_scores[i, l, k] is:
        #   for sequence i in the batch
        #   the k-th highest score
        #   for tag-sequence of length T_i
        #   ending in tag l

        # we repeat top_k-extracting procedure one last time
        topk_total_scores, topk_final_ranktags = topk_function(
            topk_final_scores.view(batch_size, -1)  # unify the rank and tag dimensions
        )
        # We now have:
        # topk_final_ranktags[i, k] = m means
        #   for sequence i in the batch
        #   the k-th best-scoring compete tag-sequence
        #   is the (m % top_k)-th best-scoring sequence
        #   ending in tag (m // top_k).
        assert topk_total_scores.size() == (batch_size, top_k)
        assert topk_final_ranktags.size() == (batch_size, top_k)

        # We now backtrack and actually get the top_k sequences:
        predictions = self._backtrack(topk_final_ranktags, topk_backpointers, lengths)

        if (
            min_prediction_length is not None
            and min_prediction_length > max_sequence_length
        ):
            predictions = torch.cat(
                [
                    predictions,
                    torch.full(
                        (
                            batch_size,
                            top_k,
                            min_prediction_length - max_sequence_length,
                        ),
                        self.padding_tag_id,
                        device=log_emissions.device,
                        dtype=predictions.dtype,
                    ),
                ],
                dim=2,
            )

        # we return the (batch-size, topk, max_seq_length)-tensor of predictions
        # and the (batch_size, topk)-tensor of scores
        # Note that for some very short sequences
        # (ones for which num_tags ** length < top_k), we may have added some gibberish
        # tag sequences. However, their scores will be extremely low, since they
        # would have passed through one of the paddings in the top_k direction.
        return predictions, topk_total_scores

    def _calculate_logits(
        self,
        log_emissions: torch.Tensor,
        tags: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        lengths: torch.LongTensor,
    ) -> torch.Tensor:
        """
        For each sequence in the batch, computes the non-normalized log probability of
        the corresponding tags.
        Args:
            log_emissions (torch.FloatTensor): a tensor of shape
                (batch_size, max_sequence_length, num_tags) holding the emission
                features
            tags (torch.LongTensor): a tensor of shape
                (batch_size, max_sequence_length) holding the tags for all sequences in
                the batch
            padding_mask (torch.BoolTensor): boolean tensor of shape
                (batch_size, max_sequence_length) masking the padding
            lengths (torch.LongTensor): a tensor of shape (batch_size,) holding the
                lengths of each element in the sequence. It should be equal to
                padding_mask.sum(dim=1) but we pass it, so that we don't have to
                rerun this sum.
        Returns:
            (torch.FloatTensor): a tensor of shape (batch_size,) holding the
                non-normalized log probability
        """
        batch_size, max_sequence_length, num_tags = log_emissions.size()
        log_emissions = log_emissions * torch.exp(self.log_emissions_scaling)
        device = log_emissions.device
        batch_sized_indexer = torch.arange(batch_size, dtype=torch.long, device=device)
        # formula for the starts: starts[i] = log_emissions[i, 0, tags[i, 0]]
        starts = log_emissions[batch_sized_indexer, 0, tags[:, 0]].type(torch.float)
        if self.include_start_end_transitions:
            starts += self.start_transitions[tags[:, 0]]
        assert starts.size() == (batch_size,)

        # the formula for the middle transitions is
        # MT_it = log_transitions[tags[i, t + 1], tags[i, t]]
        middle_transitions = self.log_transitions[tags[:, 1:], tags[:, :-1]]
        assert middle_transitions.size() == (batch_size, max_sequence_length - 1)

        # the formula for the middle emissions is
        # ME_it = log_emissions[i, t + 1, tags[i, t + 1]]
        middle_emissions = log_emissions[
            batch_sized_indexer.view(batch_size, 1),
            torch.arange(1, max_sequence_length, dtype=torch.long, device=device).view(
                1, max_sequence_length - 1
            ),
            tags[:, 1:],
        ]
        assert middle_emissions.size() == (batch_size, max_sequence_length - 1)

        logits = starts + (
            (middle_transitions + middle_emissions) * padding_mask[:, 1:]
        ).sum(dim=1)
        if self.include_start_end_transitions:
            final_tags = util.get_final_values(tags, lengths=lengths)
            logits += self.end_transitions[final_tags]

        assert logits.size() == (batch_size,)
        return logits

    def _forward_algorithm(
        self, log_emissions: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the forward algorithm for computing the log-partition function.
        Computation is done in batch and intermediate values are stored in a tensor
        so that they can be used for fast calculations of contiguous subsequence
        marginals (when combined with the output of the backward algorithm).

        This implementation also supports a constrained forward algorithm which
        computes the log-partition function over configurations with arbitrary
        constraints for positions. Thus, the supplied mask can have size
        (batch_size, max_sequence_length, num_tags), allowing to pin any position of
        any example to any subset of tags.

        Args:
            log_emissions (torch.Tensor): a tensor of size
                (batch_size, max_sequence_length, num_tags) containing the logs of the
                emission factors, evaluated at each position of each sequence and for
                each tag
            lengths (torch.Tensor): a tensor of size (batch_size,) holding the
                integer lengths of the sequences in the batch
            mask (torch.BoolTensor): a tensor of size either
                (batch_size, max_sequence_length), if no tag-pinning is applied or
                (batch_size, max_sequence_length, num_tags) if tag-pinning is applied,
                indicating the meaningful positions of each sequence in the batch
                (and the allowed tags in each position, if tag-pinning is
                applied).
        Returns:
            log_alpha (torch.Tensor): a tensor of size
                (batch_size, max_sequence_length, num_tags) holding the logs of
                messages sent from the start of each sequence to each other position
                for each tag
            log_partition (torch.Tensor): a tensor of size (batch_size,)
        """
        log_emissions = self._scale_and_mask_log_emissions(log_emissions, lengths, mask)

        return forward_algorithm(
            log_emissions,
            self.log_transitions,
            lengths=lengths,
            start_transitions=self.start_transitions,
            end_transitions=self.end_transitions,
        )

    def _backward_algorithm(
        self,
        log_emissions: torch.Tensor,
        lengths: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the backward algorithm for computing the log partition function. Computation is done in batch and the message values are stored in a tensor
        so that they can be used for fast calculations of contiguous subsequence
        marginals (when combined with the output of the forward algorithm).
        """
        log_emissions = self._scale_and_mask_log_emissions(log_emissions, lengths, mask)

        log_beta, log_partition = forward_algorithm(
            util.padded_flip(log_emissions, lengths),
            self.log_transitions.T,
            lengths=lengths,
            start_transitions=self.end_transitions,
            end_transitions=self.start_transitions,
        )

        return util.padded_flip(log_beta, lengths), log_partition

    def _backtrack(
        self,
        topk_final_ranktags: torch.Tensor,
        topk_backpointers: torch.Tensor,
        lengths: torch.LongTensor,
    ) -> List[torch.Tensor]:
        """Runs the backtracking part of the Viterbi algorithm.

        Args:
            topk_final_ranktags (torch.Tensor): a tensor of shape (batch_size, top_k)
                holding the topk best ranktags to end each sequence in. That is,
                topk_final_ranktags[i, k] = m means that the overall k-th best
                tag-sequence for example i is the (m % top_k) - th best tag-sequence
                ending in tag (m // top_k).
            topk_backpointers (torch.Tensor): a tensor of thape
                (max_sequence_length - 1, batch_size, num_tags, top_k) holding the
                top_k preceeding ranktags for each sequence, each position and each tag
            lengths (torch.Tensor): a tensor of size (batch_size,) holding the
                integer lengths of the sequences in the batch
        Returns:
            torch.Tensor: a tensor of shape (batch_size, top_k, max_sequence_length)
                holding the top_k sequences of predicted tags for each sequence in the
                batch (padded with the padding tag up to max_sequence_length)
        """
        max_seq_length_minus_one, batch_size, num_tags, top_k = topk_backpointers.size()
        assert topk_final_ranktags.size() == (batch_size, top_k)
        assert lengths.size() == (batch_size,)
        # the following is nedded for indexing
        range_batch_size = (
            torch.arange(
                batch_size, device=topk_final_ranktags.device, dtype=torch.long
            )
            .view(batch_size, 1)
            .contiguous()
        )

        viterbi_sequences = []
        topk_current_ranktags = topk_final_ranktags
        # Recall that each element in the backpointers holds the ranktag of what's
        # coming before it. In order to do backtracking, we want to iteratively
        # use the current ranktags to index in the backpointers, so we get the previous
        # ranktags. For that, we need the tag and rank dimensions of the backpointers
        # to be united in a ranktag dimension.
        backpointers_viewed_for_backtracking = topk_backpointers.view(
            max_seq_length_minus_one, batch_size, num_tags * top_k
        )
        for t in range(1, max_seq_length_minus_one + 1):
            viterbi_sequences.append(topk_current_ranktags)
            # We now want to update topk_current_ranktags by the rule:
            # topk_current_ranktags[i, k] = topk_backpointers[
            #     T_i - t - 1,
            #     i,
            #     topk_current_ranktags[i, k] // top_k,
            #     topk_current_ranktags[i, k] % top_k
            # ]
            # Since we've combined the tag and rank dimensions of the backpointers
            # we don't need to do any // or %.
            topk_current_ranktags = backpointers_viewed_for_backtracking[
                # index in the correct length position for each sequence.
                # Yes, once t exceeds the length of a shorter sequence,
                # this will start indexing in the wrong positions
                # but we don't care because we will
                # slice each sequence to the appropriate length in the end.
                (lengths - t - 1).view(batch_size, 1),
                range_batch_size,
                topk_current_ranktags,  # this has shape batch_size x top_k,
            ]
            # Now we have that:
            #   topk_current_ranktags[i, k] = m means
            #   for sequence i in the batch
            #   the k-th best-scoring tag-sequence
            #   is prefixed by the (m % top_k) - th besk-scoring
            #   tag-sequence of length (T_i - t)
            #   with m // top_k as its tag at position (T_i - t) - 1

        viterbi_sequences.append(topk_current_ranktags)
        viterbi_sequences = torch.stack(viterbi_sequences, dim=0).div(
            top_k, rounding_mode="floor"  # drop the rank information from the ranktags
        )

        max_sequence_length = max_seq_length_minus_one + 1
        assert viterbi_sequences.size() == (max_sequence_length, batch_size, top_k)

        # We need to flip the sequence length dimension and replace the
        # jibberish tags beyond each sequence's length by the padding tag.
        # It is useful to return padded predictions tensor for the case when
        # predictions are run on multiple GPUs and the pytorch machinery shards
        # each batch into sub-batches that are then recombined.
        mask = util.get_mask_from_sequence_lengths(lengths, max_sequence_length)
        viterbi_sequences = viterbi_sequences.transpose(0, 1).masked_fill(
            ~mask.unsqueeze(-1), self.padding_tag_id
        )
        viterbi_sequences = util.padded_flip(viterbi_sequences, lengths).transpose(1, 2)
        assert viterbi_sequences.size() == (batch_size, top_k, max_sequence_length)
        return viterbi_sequences


def forward_algorithm(
    log_emissions,
    log_transitions,
    lengths,
    start_transitions=None,
    end_transitions=None,
    # mask=None,
):
    """
    Runs the forward algorithm for computing the log-partition function.
    Computation is done in batch and intermediate values are stored in a tensor
    so that they can be used for fast calculations of contiguous subsequence
    marginals (when combined with the output of the backward algorithm).

    This implementation also supports a constrained forward algorithm which
    computes the log-partition function over configurations with arbitrary
    constraints for positions. Thus, the supplied mask can have size
    (batch_size, max_sequence_length, num_tags), allowing to pin any position of
    any example to any subset of tags.

    Args:
        log_emissions (torch.Tensor): a tensor of size
            (batch_size, max_sequence_length, num_tags) containing the logs of the
            emission factors, evaluated at each position of each sequence and for
            each tag
        log_transitions (torch.Tensor): a tensor of size (num_tags, num_tags)
            holding the transition factors for each pair of tags
        lengths (torch.Tensor): a tensor of size (batch_size,) holding the
            integer lengths of the sequences in the batch
        mask (torch.BoolTensor): a tensor of size either
            (batch_size, max_sequence_length), if no tag-pinning is applied or
            (batch_size, max_sequence_length, num_tags) if tag-pinning is applied,
            indicating the meaningful positions of each sequence in the batch (
            and the allowed tags in each position, if tag-pinning is applied
            )
    Returns:
        log_alpha (torch.Tensor): a tensor of size
            (batch_size, max_sequence_length, num_tags) holding the logs of
            messages sent from the start of each sequence to each other position
            for each tag
        log_partition (torch.Tensor): a tensor of size (batch_size,)
    """
    batch_size, max_sequence_length, num_tags = log_emissions.size()

    log_emissions = log_emissions.transpose(0, 1).contiguous()
    # Instantiate the log-alpha tensor which will hold the (logs of) messages
    # passed from the start of a sequence to every other position in it.
    # (use new_zeros to keep the same dtype and device of log_emissions)
    log_alpha = log_emissions.new_zeros(max_sequence_length, batch_size, num_tags)
    # mask = mask.transpose(0, 1).reshape(max_sequence_length, batch_size, 1).expand(max_sequence_length, batch_size, num_tags)
    # Initial log_alpha combines the transitions from the initial state and
    # the log_emissions for the first time step.
    if start_transitions is not None:
        log_alpha[0] = start_transitions.view(1, num_tags) + log_emissions[0]
    else:
        log_alpha[0] = log_emissions[0]

    for t in range(1, max_sequence_length):
        # The messages from the previous time step for example i and tag l are
        # calculated by the formula M_{il} = lse_{l'} (A[t - 1]_{il'} + \mu_{l'l}),
        # where l' ranges over all possible tags. This is achieved by creating a
        # new tags-dimension for log_alpha (to be indexed by l) and a new batch
        # dimension for self.log_transitions (to be indexed by i) and the rest is
        # broadcasting.
        # NB!: do not use torch.logsumexp, causes gradient instability!
        messages_from_previous_timestep = util.logsumexp(
            log_alpha[t - 1].view(batch_size, 1, num_tags)
            + log_transitions.view(1, num_tags, num_tags),
            # # we only want to mask the tag which is the source of the message,
            # # not the destination, so we broadcast the destination dimension
            # mask=mask[t - 1].view(batch_size, num_tags, 1),
            dim=2,
        )
        assert messages_from_previous_timestep.size() == (batch_size, num_tags)

        log_alpha[t] = log_emissions[t] + messages_from_previous_timestep

    final_alphas = util.get_final_values(
        tensor=log_alpha, lengths=lengths, batch_dim=1, sequence_dim=0
    )
    # Every sequence needs to end with a transition to the stop_tag.
    if end_transitions is not None:
        stops = final_alphas + end_transitions.view(1, num_tags)
    else:
        stops = final_alphas

    assert stops.size() == (batch_size, num_tags)

    log_partition = util.logsumexp(stops, dim=-1)

    return log_alpha.transpose(0, 1).contiguous(), log_partition
