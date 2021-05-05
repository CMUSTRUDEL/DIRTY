import argparse
import json
from collections import Counter, defaultdict

import torch
from tqdm import tqdm

from dirty.utils.evaluate import add_options, load_data


def view(c, cc, dcc):
    for typ, cnt in c.most_common(10):
        print(typ)
        for pred_typ, n in cc[typ].most_common(6):
            print(f"{pred_typ}, {n / cnt:.3f};", end=" ")
        print()
        print("-" * 30)
        # for src_typ, n in dcc[typ].most_common(6):
        #     print(f"{src_typ}, {n / cnt:.3f};", end=" ")
        # print()
        # print("-" * 30)


def find_most_common(results, dataset):
    c = Counter()
    cc = defaultdict(Counter)
    dcc = defaultdict(Counter)
    sc = Counter()
    scc = defaultdict(Counter)
    sdcc = defaultdict(Counter)
    for idx, example in tqdm(enumerate(dataset)):
        for src_name, src_type, tgt_type in zip(
            example.src_var_names,
            example.src_var_types_str,
            example.tgt_var_types_str,
        ):
            pred_type, _ = (
                results.get(example.binary, {})
                .get(example.name, {})
                .get(src_name[2:-2], ("", ""))
            )
            if (
                not example.test_meta["function_body_in_train"]
                and pred_type
                and pred_type != "<unk>"
            ):
                c[tgt_type] += 1
                cc[tgt_type][pred_type] += 1
                dcc[tgt_type][src_type] += 1
                if tgt_type.startswith("struct"):
                    sc[tgt_type] += 1
                    scc[tgt_type][pred_type] += 1
                    sdcc[tgt_type][src_type] += 1
    view(c, cc, dcc)
    view(sc, scc, sdcc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    results = json.load(open(args.pred_file))
    dataset = load_data(args.config_file)
    dataset = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=None)
    find_most_common(results, dataset)
