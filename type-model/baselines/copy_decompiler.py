import json

import _jsonnet
import torch
import wandb
from pytorch_lightning.metrics.functional.classification import accuracy
from tqdm import tqdm
from utils.dataset import Dataset
from utils.dire_types import TypeLibCodec

from .most_common import make_struct_mask

if __name__ == "__main__":
    config = json.loads(_jsonnet.evaluate_file("config.xfmr.jsonnet"))
    dataset = Dataset("data1/dev-*.tar", config["data"])
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
        preds_list.append(preds)
        targets_list.append(targets)
    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)

    wandb.init(name="copy_decompiler", project="dire")
    wandb.log({"test_acc": accuracy(preds, targets)})
    wandb.log(
        {
            "test_acc_macro": accuracy(
                preds, targets, num_classes=len(types_model), class_reduction="macro"
            )
        }
    )
    struc_mask = make_struct_mask(types_model, targets)
    wandb.log({"test_struc_acc": accuracy(preds[struc_mask], targets[struc_mask])})
