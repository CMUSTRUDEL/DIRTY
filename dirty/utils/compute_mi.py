import argparse

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    normalized_mutual_info_score,
    mutual_info_score,
    adjusted_mutual_info_score,
)

from utils.evaluate import load_data


def add_options(parser):
    parser.add_argument("--config-file", type=str, required=True)


def compute_mi(dataset):
    name_dict = {}
    type_dict = {}
    names = []
    types = []
    for idx, example in tqdm(enumerate(dataset)):
        for tgt_name, tgt_type in zip(example.tgt_var_names, example.tgt_var_types_str):
            tgt_name = tgt_name[2:-2]
            if tgt_name not in name_dict:
                name_dict[tgt_name] = len(name_dict)
            if tgt_type not in type_dict:
                type_dict[tgt_type] = len(type_dict)
            names.append(name_dict[tgt_name])
            types.append(type_dict[tgt_type])
        if idx == 10000:
            break
    np.random.shuffle(names)
    return adjusted_mutual_info_score(names, types)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    dataset = load_data(args.config_file)
    dataset = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=None)
    print(compute_mi(dataset))
