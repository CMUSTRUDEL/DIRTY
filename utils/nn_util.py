import math
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