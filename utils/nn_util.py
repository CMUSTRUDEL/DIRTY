import math
import numpy as np
import torch


def glorot_init(params):
    for p in params:
        if len(p.data.size()) > 1:
            torch.nn.init.xavier_normal(p.data)


def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        yield examples