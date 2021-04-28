from contextlib import ContextDecorator
import glob
import json
import sys
from collections import Counter

import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm


def _file_iter_to_line_iter(jsonl_iter):
    for jsonl in jsonl_iter:
        lines = jsonl["jsonl"].split(b"\n")
        for line in lines:
            if not line:
                continue
            json_line = json.loads(line)
            json_line["binary"] = jsonl["__key__"]
            yield json_line


if __name__ == "__main__":
    print(sys.argv[1])
    urls = sorted(glob.glob(sys.argv[1]))
    dataset = wds.Dataset(urls).pipe(_file_iter_to_line_iter)
    dataset = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=None)
    uniq_code = set()
    uniq_binary = set()
    token_len = []
    num_vars = []

    def tokenlen(example):
        return len(example["code_tokens"])

    def num_var(example):
        return len(example["source"])

    def name(example):
        return example["name"]

    # def num_var(example):
    #     return len(set([token for token in example["code_tokens"] if token.startswith("@@") and token.endswith("@@") and len(token) > 4]))
    # def name(example):
    #     return example["function"]
    body_in_train = []
    for example in tqdm(dataset):
        token_len.append(tokenlen(example))
        num_vars.append(num_var(example))
        uniq_code.add(name(example) + "".join(example["code_tokens"]))
        uniq_binary.add(example["binary"][:64])
        # body_in_train.append(example["test_meta"]["function_body_in_train"])

    print(np.mean(token_len), np.median(token_len))
    print(np.mean(num_vars), np.median(num_vars))
    print(len(uniq_code))
    print(len(uniq_binary))
    # print(np.mean(body_in_train))
