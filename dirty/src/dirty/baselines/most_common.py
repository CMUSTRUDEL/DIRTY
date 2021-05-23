import json
from typing import Any, Dict

import _jsonnet
from csvnpm.binary.dire_types import TypeLibCodec
from torch.utils.data import DataLoader
from tqdm import tqdm

from dirty.utils.dataset import Dataset  # type: ignore

if __name__ == "__main__":
    config = json.loads(_jsonnet.evaluate_file("retype.xfmr.jsonnet"))
    dataset = Dataset(config["data"]["test_file"], config["data"])
    dataloader: DataLoader = DataLoader(dataset, num_workers=8, batch_size=None)
    with open(config["data"]["typelib_file"]) as type_f:
        typelib = TypeLibCodec.decode(type_f.read())
    most_common_for_size = {}
    types_model = dataset.vocab.types
    for size in typelib:
        freq, tp = typelib[size][0]
        most_common_for_size[size] = str(tp)

    results: Dict[str, Dict[str, Any]] = {}
    for example in tqdm(dataloader):
        for src_name, src_type, tgt_var_mem in zip(
            example.src_var_names,
            example.src_var_types_str,
            example.tgt_var_src_mems,
        ):
            results.setdefault(example.binary, {}).setdefault(example.name, {})[
                src_name[2:-2]
            ] = (
                most_common_for_size.get(tgt_var_mem[1] - 3, ""),
                "",
            )
    json.dump(results, open("most_common.json", "w"))
