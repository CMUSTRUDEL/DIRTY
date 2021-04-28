import json

import _jsonnet
import torch
from tqdm import tqdm
from utils.dataset import Dataset

if __name__ == "__main__":
    config = json.loads(_jsonnet.evaluate_file("retype.xfmr.jsonnet"))
    dataset = Dataset(config["data"]["test_file"], config["data"])
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=None)
    types_model = dataset.vocab.types

    results = {}
    for example in tqdm(dataset):
        for src_name, src_type in zip(example.src_var_names, example.src_var_types_str):
            results.setdefault(example.binary, {}).setdefault(example.name, {})[
                src_name[2:-2]
            ] = (src_type, "")

    json.dump(results, open("copy_decompiler.json", "w"))
