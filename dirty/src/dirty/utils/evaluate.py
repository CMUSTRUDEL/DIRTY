import argparse
import json

import _jsonnet
import numpy as np
import wandb
from tqdm import tqdm

from dirty.utils.dataset import Dataset


def add_options(parser):
    parser.add_argument(
        "--pred-file",
        type=str,
        required=True,
        help="Saved predictions on a dataset",
    )
    parser.add_argument("--config-file", type=str, required=True)


def load_data(config_file):
    config = json.loads(_jsonnet.evaluate_file(config_file))["data"]
    config["max_num_var"] = 1 << 30
    dataset = Dataset(config["test_file"], config)
    return dataset


def acc(preds, results, test_metas=None):
    return (preds == results).mean()


def mask_acc(preds, results, mask):
    return (preds[mask] == results[mask]).mean()


def body_in_train_acc(preds, results, test_metas):
    body_in_train_mask = np.array(
        [test_meta["function_body_in_train"] for test_meta in test_metas]
    )
    return mask_acc(preds, results, body_in_train_mask)


def body_not_in_train_acc(preds, results, test_metas):
    body_in_train_mask = np.array(
        [test_meta["function_body_in_train"] for test_meta in test_metas]
    )
    return mask_acc(preds, results, ~body_in_train_mask)


def struct_acc(preds, results, test_metas):
    struct_mask = np.array([test_meta["is_struct"] for test_meta in test_metas])
    return mask_acc(preds, results, struct_mask)


def struct_body_in_train_acc(preds, results, test_metas):
    mask = np.array(
        [
            test_meta["is_struct"] and test_meta["function_body_in_train"]
            for test_meta in test_metas
        ]
    )
    return mask_acc(preds, results, mask)


def struct_body_not_in_train_acc(preds, results, test_metas):
    mask = np.array(
        [
            test_meta["is_struct"] and not test_meta["function_body_in_train"]
            for test_meta in test_metas
        ]
    )
    return mask_acc(preds, results, mask)


TYPE_METRICS = {
    "acc": acc,
    "body_in_train_acc": body_in_train_acc,
    "body_not_in_train_acc": body_not_in_train_acc,
    "struct_acc": struct_acc,
    "struct_body_in_train_acc": struct_body_in_train_acc,
    "struct_body_not_in_train_acc": struct_body_not_in_train_acc,
}

NAME_METRICS = {
    "accuracy": acc,
    "body_in_train_acc": body_in_train_acc,
    "body_not_in_train_acc": body_not_in_train_acc,
}


def evaluate(dataset, results, type_metrics, name_metrics):
    pred_names, ref_names, pred_types, ref_types = [], [], [], []
    test_meta_types, test_meta_names = [], []
    for example in tqdm(dataset):
        for src_name, tgt_name, tgt_type in zip(
            example.src_var_names,
            example.tgt_var_names,
            example.tgt_var_types_str,
        ):
            pred_type, _ = (
                results.get(example.binary, {})
                .get(example.name, {})
                .get(src_name[2:-2], ("", ""))
            )
            pred_types.append(pred_type)
            ref_types.append(tgt_type)
            test_meta = example.test_meta.copy()
            test_meta["is_struct"] = dataset.dataset.vocab.types.id2word[
                dataset.dataset.vocab.types[tgt_type]
            ].startswith("struct ")
            test_meta_types.append(test_meta)
            if src_name != tgt_name and tgt_name != "@@@@":
                # only report need_rename
                _, pred_name = (
                    results.get(example.binary, {})
                    .get(example.name, {})
                    .get(src_name[2:-2], ("", ""))
                )
                pred_names.append(pred_name)
                ref_names.append(tgt_name[2:-2])
                test_meta_names.append(test_meta)
    pred_types = np.array(pred_types, dtype=object)
    ref_types = np.array(ref_types, dtype=object)
    for metric_name, metric in type_metrics.items():
        wandb.log(
            {
                f"test_retype_{metric_name}": metric(
                    pred_types, ref_types, test_meta_types
                )
            }
        )

    pred_names = np.array(pred_names, dtype=object)
    ref_names = np.array(ref_names, dtype=object)

    for metric_name, metric in name_metrics.items():
        wandb.log(
            {
                f"test_rename_{metric_name}": metric(
                    pred_names, ref_names, test_meta_names
                )
            }
        )


#     mt_evaluate(dataset, results)


# def mt_evaluate(dataset, results):
#     mt_results = json.load(open("pred_mt.json"))
#     pred_names, ref_names, pred_types, ref_types = [], [], [], []
#     for example in tqdm(dataset):
#         for src_name, tgt_name, tgt_type in zip(
#             example.src_var_names, example.tgt_var_names, example.tgt_var_types_str
#         ):
#             pred_type, _ = (
#                 results.get(example.binary, {})
#                 .get(example.name, {})
#                 .get(src_name[2:-2], ("", ""))
#             )
#             _, mt_pred_name = (
#                 mt_results.get(example.binary, {})
#                 .get(example.name, {})
#                 .get(src_name[2:-2], ("", ""))
#             )
#             if mt_pred_name == tgt_name[2:-2] and tgt_name != "@@@@":
#                 pred_types.append(pred_type)
#                 ref_types.append(tgt_type)
#             if src_name != tgt_name and tgt_name != "@@@@":
#                 _, pred_name = (
#                     results.get(example.binary, {})
#                     .get(example.name, {})
#                     .get(src_name[2:-2], ("", ""))
#                 )
#                 mt_pred_type, _ = (
#                     mt_results.get(example.binary, {})
#                     .get(example.name, {})
#                     .get(src_name[2:-2], ("", ""))
#                 )
#                 if mt_pred_type == tgt_type:
#                     pred_names.append(pred_name)
#                     ref_names.append(tgt_name[2:-2])
#     pred_types = np.array(pred_types, dtype=object)
#     ref_types = np.array(ref_types, dtype=object)

#     pred_names = np.array(pred_names, dtype=object)
#     ref_names = np.array(ref_names, dtype=object)

#     wandb.log({"test_rnonrt_accuracy": acc(pred_names, ref_names)})
#     wandb.log({"test_rtonrn_accuracy": acc(pred_types, ref_types)})


def main():
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()

    results = json.load(open(args.pred_file))
    dataset = load_data(args.config_file)
    import torch

    dataset = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=None)

    wandb.init(name=f"test_{args.pred_file}", project="dire")
    evaluate(dataset, results, TYPE_METRICS, NAME_METRICS)


if __name__ == "__main__":
    main()
