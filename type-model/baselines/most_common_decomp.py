import json

import _jsonnet
import torch
import wandb
from pytorch_lightning.metrics.functional.classification import accuracy
from tqdm import tqdm
from utils.dataset import Dataset
from utils.dire_types import TypeLibCodec
from typing import Dict
from collections import defaultdict, Counter

from .most_common import make_struct_mask

def evaluate(config, most_common_for_src: Dict[int, int]):
    dataset = Dataset(config["data"]["test_file"], config["data"])
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=8, batch_size=64, collate_fn=Dataset.collate_fn
    )
    types_model = dataset.vocab.types

    preds_list = []
    targets_list = []
    for batch in tqdm(dataloader):
        input_dict, target_dict = batch
        targets = target_dict["target_type_id"][target_dict["target_mask"]]
        preds = input_dict["src_type_id"][target_dict["target_mask"]]
        preds = torch.tensor([most_common_for_src.get(pred, pred) for pred in preds.tolist()])
        preds_list.append(preds)
        targets_list.append(targets)
    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)

    wandb.init(name="most_common_decomp", project="dire")
    wandb.log({"test_retype_acc": accuracy(preds, targets)})
    wandb.log(
        {
            "test_retype_acc_macro": accuracy(
                preds, targets, num_classes=len(types_model), class_reduction="macro"
            )
        }
    )
    struc_mask = make_struct_mask(types_model, targets)
    wandb.log({"test_retype_struc_acc": accuracy(preds[struc_mask], targets[struc_mask])})

def compute(config) -> Dict[int, int]:
    dataset = Dataset(config["data"]["train_file"], config["data"])
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=8, batch_size=64, collate_fn=Dataset.collate_fn
    )
    most_common_for_src = defaultdict(Counter)
    for batch in tqdm(dataloader):
        input_dict, target_dict = batch
        targets = target_dict["target_type_id"][target_dict["target_mask"]]
        preds = input_dict["src_type_id"][target_dict["target_mask"]]
        for pred, target in zip(preds.tolist(), targets.tolist()):
            most_common_for_src[pred][target] += 1
    for key in most_common_for_src:
        most_common_for_src[key] = most_common_for_src[key].most_common(1)[0][0]
    return most_common_for_src

if __name__ == "__main__":
    config = json.loads(_jsonnet.evaluate_file("config.xfmr.jsonnet"))
    most_common_for_src = compute(config)
    evaluate(config, most_common_for_src)

