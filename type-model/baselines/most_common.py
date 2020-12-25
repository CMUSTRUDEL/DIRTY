import json

import _jsonnet
import torch
import wandb
from pytorch_lightning.metrics.functional.classification import accuracy
from tqdm import tqdm
from utils.dataset import Dataset
from utils.dire_types import TypeLibCodec


def make_struct_mask(types_model, targets):
    struct_set = set()
    for idx, type_str in types_model.id2word.items():
        if type_str.startswith("struct"):
            struct_set.add(idx)
    struc_mask = torch.zeros(len(targets), dtype=torch.bool)
    for idx, target in enumerate(targets):
        if target.item() in struct_set:
            struc_mask[idx] = 1
    return struc_mask


if __name__ == "__main__":
    config = json.loads(_jsonnet.evaluate_file("config.xfmr.jsonnet"))
    dataset = Dataset(config["data"]["dev_file"], config["data"])
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=8, batch_size=64, collate_fn=Dataset.collate_fn
    )
    with open(config["data"]["typelib_file"]) as type_f:
        typelib = TypeLibCodec.decode(type_f.read())
    most_common_for_size = {}
    types_model = dataset.vocab.types
    for size in typelib:
        freq, tp = typelib[size][0]
        most_common_for_size[size] = types_model[str(tp)]

    preds_list = []
    targets_list = []
    for batch in tqdm(dataloader):
        input_dict, target_dict = batch
        targets = target_dict["target_type_id"][target_dict["target_mask"]]
        preds = []
        for mems, target in zip(input_dict["target_type_src_mems"], targets.tolist()):
            size = mems[mems != 0].tolist()[0] - 3
            if size not in most_common_for_size:
                preds.append(types_model.unk_id)
                continue
            preds.append(most_common_for_size[size])
        preds_list.append(torch.tensor(preds))
        targets_list.append(targets)
    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)
    print(preds.shape, targets.shape)

    wandb.init(name="most_common", project="dire")
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
