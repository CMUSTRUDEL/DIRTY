import math
from typing import Tuple

import numpy as np
import torch


SMALL_NUMBER = 1e-8


def glorot_init(params):
    for p in params:
        if len(p.data.size()) > 1:
            torch.nn.init.xavier_normal_(p.data)


def to(data, device: torch.device):
    if 'adj_lists' in data:
        [x.to(device) for x in data['adj_lists']]

    if isinstance(data, dict):
        for key, val in data.items():
            if torch.is_tensor(val):
                data[key] = val.to(device)
            # recursively move tensors to GPU
            elif isinstance(val, dict):
                data[key] = to(val, device)
    elif isinstance(data, list):
        for i in range(len(data)):
            val = data[i]
            if torch.is_tensor(val):
                data[i] = val.to(device)
    else:
        raise ValueError()

    return data


def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        yield examples


def get_tensor_dict_size(tensor_dict):
    total_num_elements = 0
    for key, val in tensor_dict.items():
        if isinstance(val, dict):
            num_elements = get_tensor_dict_size(val)
        elif torch.is_tensor(val):
            num_elements = val.nelement()
        else:
            num_elements = 0

        total_num_elements += num_elements

    return total_num_elements


def dot_prod_attention(h_t: torch.Tensor,
                       src_encoding: torch.Tensor,
                       src_encoding_att_linear: torch.Tensor,
                       mask: torch.Tensor = None):
    # type: (...) -> Tuple[torch.Tensor, torch.Tensor]
    att_weight = \
        torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
    if mask is not None:
        att_weight.data.masked_fill_((1. - mask).bool(), -float('inf'))

    softmaxed_att_weight = torch.softmax(att_weight, dim=-1)

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    ctx_vec = torch.bmm(
        softmaxed_att_weight.view(*att_view),
        src_encoding
    ).squeeze(1)

    return ctx_vec, softmaxed_att_weight


def get_lengths_from_binary_sequence_mask(mask: torch.Tensor):
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.
    Parameters
    ----------
    mask : torch.Tensor, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.
    Returns
    -------
    A torch.LongTensor of shape (batch_size,) representing the lengths
    of the sequences in the batch.
    """
    return mask.long().sum(-1)


def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    """Sort a batch first tensor by some specified lengths.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A batch first Pytorch tensor.
    sequence_lengths : torch.LongTensor, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.
    Returns
    -------
    sorted_tensor : torch.FloatTensor
        The original tensor sorted along the batch dimension with respect to
        sequence_lengths.
    sorted_sequence_lengths : torch.LongTensor
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : torch.LongTensor
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) ==
        original_tensor``
    permuation_index : torch.LongTensor
        The indices used to sort the tensor. This is useful if you want to sort
        many tensors using the same ordering.

    """

    if not isinstance(tensor, torch.Tensor) \
       or not isinstance(sequence_lengths, torch.Tensor):
        raise ValueError(
            "Both the tensor and sequence lengths must be torch.Tensors."
        )

    sorted_sequence_lengths, permutation_index = \
        sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)

    index_range = torch.arange(
        0, len(sequence_lengths), device=sequence_lengths.device
    )
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return (sorted_tensor,
            sorted_sequence_lengths,
            restoration_indices,
            permutation_index)
