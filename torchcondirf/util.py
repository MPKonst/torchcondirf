"""
Some utility functions for tensor manipulation.
Some functions inspired by/copied from 
https://github.com/allenai/allennlp/blob/main/allennlp/nn/util.py
"""

import torch
from typing import List, Optional, Iterable
from functools import partial


def get_dropout_mask(
    features: torch.Tensor,
    *,
    dropout_probability: float = 0.5,
    broadcast_dims: Optional[Iterable[int]] = (),
    dropout_module: Optional[torch.nn.Dropout] = None,
) -> torch.FloatTensor:
    """
    Creates a dropout mask to multiply the features tensor with.

    If `broadcast_dims` are specified, then the dropout mask will have size 1 along
    these dimensions and can therefore be broadcast over them.
    For example, if the features array has shape
    batch_size x max_sequence_length x hidden_dim, then one can pass
    `broadcast_dims = [1]` which has the effect of dropping the same
    features for each position in the sequence. This makes sense because the point
    is to remove entire features (equivalently -- exclude certain weights from
    the given training iteration) and weights are shared across positions.
    On the other hand, we are dropping different features for each training
    example in the batch -- this seems to be the standard way of applying
    dropout; it makes sense because it makes the dropout application
    independent of batch size.

    Args:
        features (torch.Tensor): the tensor of features on which the mask will be based
        dropout_probability (float): the probability with which the features will be
            dropped. Defaults to 0.5.
        broadcast_dims (Iterable[int]): The dimensions along which to broadcast
            the mask. If specified, the mask will have size 1 along these
            dimensions. Defaults to ().
        dropout_module (torch.nn.Dropout, optional): a Dropout module to be used.
            If provided, features will be dropped using this module instead of
            dropout_probability.
            This is used so that a dropout module can be registered with the
            overarching module and the training and testing behaviour can be controlled
            automatically through `.train` and `.eval`. Defaults to None.

    Returns:
        torch.FloatTensor: the dropout mask
    """
    dropout_module = (
        dropout_module
        if dropout_module is not None
        else torch.nn.Dropout(dropout_probability, inplace=True)
    )
    mask_shape = [*features.size()]
    for dim in broadcast_dims:
        mask_shape[dim] = 1

    dropout_mask = torch.ones(*mask_shape, device=features.device)
    inplace = dropout_module.inplace
    dropout_module.inplace = True  # * set to inplace to save memory
    dropout_mask = dropout_module(dropout_mask)
    dropout_module.inplace = inplace
    return dropout_mask


def get_embeddings_with_dropout(
    embeddings: torch.nn.Embedding,
    input_ids: torch.LongTensor,
    dropout_probability: float = 0.5,
    *,
    batch_dim: int = 0,
) -> torch.Tensor:
    """
    Retrieve embeddings from an embeddings tensor and apply embedding dropout.

    If dropout_probability > 0, then for each element in the batch this function chooses
    words to mask out and sets their embeddings to 0, while all other embeddings are
    scaled by 1/(1 - dropout_probability) to preserve statistics. This is equivalent
    to applying dropout to one-hot encoded input ids.

    Based on Gal ang Ghahramani, "A Theoretically Grounded Application of Dropout
    in Recurrent Neural Networks", Section 4.2.

    Args:
        embeddings (torch.nn.Embedding): the embeddings Module
        input_ids (torch.LongTensor): tensor of input ids for which to
            retrieve embeddings
        dropout_probability (float, optional):probability of an input id's
            embedding being zeroed out. Defaults to 0.5.
        batch_dim (int, optional): The dimension of the input_ids tensor which indexes
            elements in the batch. Defaults to 0.

    Returns:
        torch.Tensor: the embeddings of the input ids with dropout applied. Has the
            shape of input_ids with one extra dimension at the end, holding the actual
            embeddings.
    """
    embedded_values = embeddings(input_ids)
    if dropout_probability == 0:
        return embedded_values

    batch_size = input_ids.size(batch_dim)
    vocab_mask = input_ids.new(size=(batch_size, embeddings.num_embeddings)).bernoulli_(
        1 - dropout_probability
    )
    shape = [1 for _ in range(input_ids.ndim)]
    shape[batch_dim] = batch_size
    # the formula for the embedded values mask is:
    # embedded_values_mask[i_1, ..., i_k] = vocab_mask[
    #   i_{batch_dim}, input_ids[i_1, ..., i_k]
    # ]
    embedded_values_mask = vocab_mask[
        torch.arange(batch_size, device=input_ids.device).view(shape), input_ids
    ]
    # * rescale the masks and unsqueeze the embedding dimension, so it can broadcasted
    embedded_values_mask = embedded_values_mask.div(1 - dropout_probability).unsqueeze(
        -1
    )

    return embedded_values * embedded_values_mask


def get_mask_from_sequence_lengths(
    sequence_lengths: torch.Tensor, max_length: int
) -> torch.BoolTensor:
    """
    Given a variable of shape `(batch_size,)` that represents the sequence lengths of
    each batch element, this function returns a `(batch_size, max_length)` mask
    variable on the same device as sequence_lengths. For example, if
    our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
    `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`.

    We require `max_length` here instead of just computing it from the input
    `sequence_lengths` to allow for padding that exceeds the lengths of any of the
    sequences. This is useful when we want to pad to a specific length, e.g.
    if other padded tensors from the batch are being sharded to multiple gpus.
    """
    # using new_ones here just so that we obtain a tensor with the same torch.device as
    # sequence_lengths. (batch_size, max_length)
    ones = sequence_lengths.new_ones((sequence_lengths.size(0), max_length))
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor


def get_mask_for_tags(
    length_mask: torch.BoolTensor,
    num_tags: int,
    constraints: List[List[tuple]] = None,
) -> torch.BoolTensor:
    """
    Given the mask for sequence lengths and some constraints on which tags are
    allowed for which positions in which sequence, creates a boolean mask of shape
    (batch_size, max_sequence_length, num_tags) which can then be used in the forward
    algorithm to compute a constrained partition function.
    Args:
        length_mask (torch.BoolTensor) -- a tensor of shape
            (batch_size, max_sequence_length) masking the padding for each sequence
            in the batch
        num_tags (int) -- the number of different possible tags
        constraints (List[List[tuple]]) -- a list of length batch_size, indicating
            which tags are allowed for which position of each sequence in the batch.
            Each element of the list is itself a list of 3-tuples, where each 3-tuple
            is of the form (start, end, tags) and indicates that for that sequence and
            for positions slice(start, end) the allowed tags are `tags`.
            The `tags` value is used to index into the tags-dimension of the boolean
            mask, so it must be an integer, or list of integers, slice, range, etc.
            For example:
                constraints = [
                                [(0, 3, 1), (3, 5, [4, 7, 8])],
                                [(0, 1, 2), (3, 4, range(1, 8))],
                            ]
            means that we have a batch of two sequences and:
              - in sequence 0, for positions 0 to 3 only tag 1 is allowed and for
              positions 3 to 5 the allowed tags are 4, 7, 8
              - in sequence 1, for position 0 only tag 2 is allows and for position 3
              the allowed tags are 1, 2, 3, 4, 5, 6, 7

            For each sequence the constrained ranges must be non-overlapping. All
            constraints which fall outside the length of a sequence will be ignored.
            Once the mask for tags is created, further constraints cannot be added to
            it, rather the whole mask must be recreated.

    Returns:
        (torch.BoolTensor) -- a tensor of shape
            (batch_size, max_sequence_length, num_tags) indicating which tags are
            allowed for which positions in which sequence

    """
    constraints = constraints or []
    # allow all tags in all positions
    mask_for_tags = length_mask.new_ones(length_mask.size() + (num_tags,))
    for i, sequence_constraints in enumerate(constraints):
        for start, stop, tags in sequence_constraints:
            # disallow all tags in the constrained positions
            mask_for_tags[i, start:stop, :] = False
            # allow only the allowed tags
            mask_for_tags[i, start:stop, tags] = True
    # multiply by the length mask to mask all padding
    return mask_for_tags * length_mask.unsqueeze(2)


def get_final_values(
    tensor: torch.Tensor,
    lengths: torch.Tensor,
    batch_dim=0,
    sequence_dim=1,
    keepdim=False,
) -> torch.Tensor:
    """
    Given a (possibly padded) Tensor containing values for each positions in each
    sequence in a batch, extract a tensor of only the values corresponding to the
    last position in each sequence in the batch.
    Args:
        tensor (torch.Tensor) -- the tensor of values
        lengths (torch.Tensor) -- a tensor of shape (batch_size,) containing
            the integer lengths of the sequences in the batch
        batch_dim (int) -- the dimension of `tensor` indicating the batch element
            (Defaults to 0)
        sequence_dim (int) -- the dimension of `tensor` indicating the sequence
            position (Defaults to 1)
        keepdim (bool) -- whether to keep the sequence_dim of the output tensor
    Returns:
        (torch.Tensor) -- tensor of the same type as `tensor`, containing the
            values that correspond to the last element in each sequence
    """
    batch_size = len(lengths)
    assert batch_size == tensor.size(batch_dim), (
        f"The number of sequence lengths must coincide with the dimension of "
        f"`tensor` along batch_dim ({batch_dim}). Instead you had: \n"
        f"len(lengths) = {batch_size},\n"
        f"tensor.size({batch_dim}) = {tensor.size(batch_dim)}."
    )

    # IntTensor needed for Size
    new_view = torch.ones(tensor.ndim, dtype=torch.int, device=lengths.device)
    new_view[batch_dim] = batch_size

    shape_for_expand = [*tensor.size()]
    shape_for_expand[sequence_dim] = 1
    broadcasted_indices = (
        (lengths - 1)
        .view(torch.Size(new_view))
        .expand(shape_for_expand)
        .type(torch.long)
    )  # LongTensor needed to use as index for torch.gather.
    result = tensor.gather(dim=sequence_dim, index=broadcasted_indices)
    return result if keepdim else result.squeeze(sequence_dim)


def get_topk_function(k, dim):
    """
    Convenience util to return either torch.max or torch.topk
    depending on whether k = 1 or not.
    """
    if k == 1:
        # .max is about 3 times faster than .topk with k=1
        return partial(
            torch.max,
            dim=dim,
            # need keepdim=True to get the same behaviour as .topk
            keepdim=True,
        )
    return partial(torch.topk, k=k, dim=dim)


def masked_topk(
    tensor: torch.Tensor, mask: torch.BoolTensor, k: int, dim: int = -1
) -> torch.Tensor:
    """
    Compute the topk of the given tensor along the given dimension,
    ignoring all masked values.

    Args:
        tensor (torch.Tensor): tensor to take topk of
        mask (torch.BoolTensor): boolea mask broadcastable to the shape of `tensor`.
            determining which tensor values to ignore
        k (int): the number of highest-scoring values to return
        dim (int, optional): Dimension along which to take topk. Defaults to -1.

    Returns:
        torch.Tensor: a tensor of the same shape as `tensor` except for dimension `dim`
            which is equal to k, holding the top k values along that dimension.
    """
    topk_function = get_topk_function(k, dim)
    masked_input = tensor.masked_fill(~mask, min_value_of_dtype(tensor.dtype))
    return topk_function(masked_input)


def extend_dim_to_k(tensor, k, fill_value, dim):
    filler_size = [*tensor.size()]
    filler_size[dim] = k - filler_size[dim]
    filler = torch.empty(filler_size, device=tensor.device).fill_(fill_value)
    return torch.cat([tensor, filler], dim=dim)


def logsumexp(
    tensor: torch.Tensor,
    mask: torch.Tensor = None,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp. This is mathematically equivalent to
    `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for
    summing log probabilities. The optional mask can be used to set some of the
    exponentiated values to 0 before suming and taking log. Note that if all values
    are masked along dim for a given remaining multiindex, the returned value at that
    multiindex will be nan.

    Args:
        tensor (torch.FloatTensor) -- a tensor of arbitrary size
        mask (torch.BoolTensor) -- a boolean tensor of shape which is broadcastable to
            that of `tensor`
        dim (int) -- the dimension along which to take the sum (defaults to -1)
        keepdim (bool) -- Whether to retain a dimension of size one at the dimension we
            reduce over.optional (Defaults to False)

    Returns:
        (torch.Tensor)
    """
    if mask is None:
        max_score, _ = tensor.max(dim, keepdim=True)
        stable_vec = tensor - max_score
        result = max_score + stable_vec.exp().sum(dim, keepdim=True).log()
        return result if keepdim else result.squeeze(dim)

    max_score, _ = masked_topk(tensor, mask, k=1, dim=dim)
    # multiplying the difference by mask only to avoid infinities in the exponential
    # whenever an entire row (along dim) is masked
    stable_vec = (tensor - max_score) * mask
    result = (
        max_score
        + (
            (
                stable_vec.exp()
                * mask  # this is the correct place to multiply by mask.
            ).sum(dim, keepdim=True)
            # to avoid -inf when log is called, we add a tiny value to where mask is all 0
            + ~mask.any(dim=dim, keepdim=True) * tiny_value_of_dtype(stable_vec.dtype)
        ).log()
    )
    return result if keepdim else result.squeeze(dim)


def padded_flip(sequences: torch.Tensor, lengths: torch.LongTensor) -> torch.Tensor:
    """Flip only the unpadded entries in a tensor of sequences,
    leving the padded ones in-place.

    Args:
        sequences (torch.Tensor): a tensor of shape
            (batch_size, max_sequence_length, ...)
        lengths (torch.LongTensor): a tensor of shape (batch_size,),
            with lengths[i] equal to the length of the i-th sequence in
            the batch. The entries in the `sequences` tensor with second
            coordinate beyond the corresponding length are considered padding.

    Returns:
        (torch.Tensor): a tensor of the same shape as `sequences`, with the
            entries which don't correspond to padding reversed.
    """
    flipped_sequences = sequences.clone()
    for i, length in enumerate(lengths):
        flipped_sequences[i, :length] = sequences[i, :length].flip(0)
    return flipped_sequences


def info_value_of_dtype(dtype: torch.dtype):
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype):
    """
    Returns the minimum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).min


def max_value_of_dtype(dtype: torch.dtype):
    """
    Returns the maximum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).max


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))
