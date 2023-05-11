"""
A base class for the Crf heads.
"""
from typing import List, Tuple

import torch
from torch import nn
import util


class BaseCrfHead(nn.Module):
    # set a negative value, representing the log of probability 0
    # Don't use values that are "too negative" to avoid inf and nan in tensors and grads
    VERY_NEGATIVE_VALUE = -10000.0

    def __init__(
        self,
        *,
        num_tags: int,
        allowed_transitions: List[Tuple[int, int]] = None,
        include_start_end_transitions: bool = True,
        padding_tag_id: int = 0,
    ):
        """
        Args:
            num_tags (int): the number of tags that can be assigned to each sequence
                element
            allowed_transitions (List[Tuple[int, int]], optional): A list of pairs of
                tag ids, where the pair (i, j) means that the transition from tag i
                to tag j is allowed. Transitions which are not in this list are
                considered forbidden. If None, then all transitions are allowed.
                Defaults to None.
            include_start_end_transitions (bool, optional): Whether to include weights
                for starting or ending the tag sequence with each tag. Defaults to True.
            padding_tag_id (int, optional): the tag id corresponding to the padding tag.
                Transitions to and from the padding tag are always forbidden.
                Defaults to 0.
        """
        super().__init__()
        self.include_start_end_transitions = include_start_end_transitions
        self.num_tags = num_tags
        self.padding``_tag_id = padding_tag_id

        # transitions_constraint_mask indicates valid transitions
        # (based on supplied constraints), including constraints for
        # start of sequence (tag = num_tags) and end of sequence (tag = num_tags + 1)
        if allowed_transitions is None:
            # All transitions are valid.
            transitions_constraint_mask = torch.ones(num_tags + 2, num_tags + 2)
        else:
            transitions_constraint_mask = torch.zeros(num_tags + 2, num_tags + 2)
            for i, j in allowed_transitions:
                transitions_constraint_mask[j, i] = 1

        # the attribute for constrained transitions does not include start and end
        self.transitions_constraint_mask = transitions_constraint_mask[
            :num_tags, :num_tags
        ].type(torch.bool)

        # nothing can transitions from and to the padding tag
        self.transitions_constraint_mask[:, self.padding_tag_id] = False
        self.transitions_constraint_mask[self.padding_tag_id, :] = False

        # start and end constraints are saved in their separate masks
        self.start_transitions_mask = transitions_constraint_mask[
            :num_tags, num_tags
        ].type(torch.bool)
        self.end_transitions_mask = transitions_constraint_mask[
            num_tags + 1, :num_tags
        ].type(torch.bool)

        self.reset_parameters()  # initialise all weights

    def reset_parameters(self):
        """
        (Re)initializes the transition weights and the emission-to-transition ratio
        of the CrfHead instance.
        """
        # setting as Parameter turns the gradient on
        self.log_emissions_scaling = torch.nn.Parameter(torch.randn(1).squeeze())
        # self.register_buffer("log_emissions_scaling", torch.tensor(0.0))
        self.log_transitions = torch.nn.Parameter(
            torch.randn(self.num_tags, self.num_tags).masked_fill_(
                ~self.transitions_constraint_mask, self.VERY_NEGATIVE_VALUE
            )
        )
        # self.register_buffer(
        #   "log_transitions", torch.zeros(self.num_tags, self.num_tags)
        # )
        start_end_trans_init = (
            torch.randn if self.include_start_end_transitions else torch.zeros
        )
        start_end_trans_reg = (
            self._register_tensor_as_parameter
            if self.include_start_end_transitions
            else self.register_buffer
        )
        for name, mask in zip(
            ["start_transitions", "end_transitions"],
            [self.start_transitions_mask, self.end_transitions_mask],
        ):
            start_end_trans_reg(
                name,
                start_end_trans_init(self.num_tags).masked_fill_(
                    ~mask, self.VERY_NEGATIVE_VALUE
                ),
            )

    def _register_tensor_as_parameter(self, name, tensor):
        self.register_parameter(name, nn.Parameter(tensor))

    def _scale_and_mask_log_emissions(self, log_emissions, lengths, mask=None):
        """
        Scales the log emissions by `torch.exp(self.log_emissions_scaling)`
        and replaces the masked log-emission values by a very negative value.
        """
        # make the emissions for forbidden tags very negative
        batch_size, max_sequence_length, num_tags = log_emissions.size()
        mask = (
            mask
            if mask is not None
            else util.get_mask_from_sequence_lengths(
                # do not pass lengths.max() to max length, otherwise
                # this may not work with multi-gpu training
                # (when the batch gets sharded, log_emissions.size(1) can be larger
                # than any of the lengths in the same shard of the batch)
                lengths,
                max_length=max_sequence_length,
            )
        )
        mask = mask.clone()
        # if there are pinned tags, then the mask
        # should also have a tags dimension, otherwise we broadcast it
        if mask.ndim == 3:
            assert mask.size() == (batch_size, max_sequence_length, num_tags)
        else:
            assert mask.size() == (batch_size, max_sequence_length)
            mask = mask.unsqueeze(2).expand(batch_size, max_sequence_length, num_tags)

        log_emissions = log_emissions * torch.exp(self.log_emissions_scaling)
        # make emissions for forbidden tags essentially -infinity
        # this saves from having to do any further masking.
        log_emissions.masked_fill_(~mask, self.VERY_NEGATIVE_VALUE)
        log_emissions[:, :, self.padding_tag_id] = self.VERY_NEGATIVE_VALUE
        return log_emissions

    @staticmethod
    def log_likelihood(
        *, logits: torch.Tensor, log_partition: torch.Tensor
    ) -> torch.Tensor:
        """Computes the log-likelihood of each example.

        Can be used e.g. as follows:
            log_likelihood = self.log_likelihood(**self.forward(**batch))

        Args:
            logits (torch.Tensor): a tensor of shape (batch_size,) holding the
                logits for each example.
            log_partition (torch.Tensor): a tensor of shape (batch_size,) holding the
                values of the log-partition function for each example.

        Returns:
            torch.Tensor: a tensor of shape (batch_size,) holding the log-likelihood
                of each example.
        """
        return logits - log_partition