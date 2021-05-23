import json
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict

import _jsonnet
from torch.utils.data import DataLoader
from tqdm import tqdm

from dirty.utils.dataset import Dataset  # type: ignore


def evaluate(config, most_common_for_src: Dict[int, int]):
    dataset = Dataset(config["data"]["test_file"], config["data"])
    dataloader: DataLoader = DataLoader(dataset, num_workers=8, batch_size=None)

    results: Dict[str, Dict[str, Any]] = {}
    for example in tqdm(dataloader):
        src_name: str
        src_type: int
        for src_name, src_type in zip(example.src_var_names, example.src_var_types_str):
            results.setdefault(example.binary, {}).setdefault(example.name, {})[
                src_name[2:-2]
            ] = (
                most_common_for_src.get(src_type, src_type),
                "",
            )

    json.dump(results, open("most_common_decompiler.json", "w"))


def compute(config) -> Dict[int, int]:
    dataset = Dataset(config["data"]["train_file"], config["data"])
    dataloader: DataLoader = DataLoader(dataset, num_workers=8, batch_size=None)
    most_common_for_src: DefaultDict[str, Counter] = defaultdict(Counter)
    for example in tqdm(dataloader):
        for src_type, tgt_type in zip(
            example.src_var_types_str, example.tgt_var_types_str
        ):
            most_common_for_src[src_type][tgt_type] += 1
    for key in most_common_for_src:
        most_common_for_src[key] = most_common_for_src[key].most_common(1)[0][0]
    return most_common_for_src  # type: ignore


def main():
    config = json.loads(_jsonnet.evaluate_file("retype.xfmr.jsonnet"))
    most_common_for_src = compute(config)
    evaluate(config, most_common_for_src)


if __name__ == "__main__":
    main()
