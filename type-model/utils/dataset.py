import glob
import multiprocessing as mp
import tarfile
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Mapping, Set

import numpy as np
import torch
import ujson as json
import webdataset as wds
from tqdm import tqdm

from utils.code_processing import tokenize_raw_code
from utils.function import CollectedFunction
from utils.variable import Location, Variable, location_from_json_key


class Example:
    def __init__(
        self,
        name: str,
        code_tokens: str,
        source: Dict[str, Mapping[Location, Set[Variable]]],
        target: Dict[str, Mapping[Location, Set[Variable]]],
        binary_file: str = "",
        valid: bool = True,
        raw_code: str = "",
    ):
        self.name = name
        self.code_tokens = code_tokens
        self.source = source
        self.target = target
        self.binary_file = binary_file
        self._is_valid = valid
        self.raw_code = raw_code

    @classmethod
    def from_json(cls, d: Dict):
        source = defaultdict(dict)
        for loc in ["a", "l"]:
            if not loc in d:
                continue
            for key, args in d[loc].items():
                source[loc][location_from_json_key(key)] = {
                    Variable.from_json(arg) for arg in args
                }
        target = defaultdict(dict)
        for loc in ["a", "l"]:
            if not loc in d:
                continue
            for key, args in d[loc].items():
                target[loc][location_from_json_key(key)] = {
                    Variable.from_json(arg) for arg in args
                }
        return cls(d["name"], d["code_tokens"], source, target)

    def to_json(self):
        assert self._is_valid
        source = defaultdict(dict)
        for loc in ["a", "l"]:
            for key, args in self.source[loc].items():
                source[loc][key.json_key()] = [arg.to_json() for arg in args]
        target = defaultdict(dict)
        for loc in ["a", "l"]:
            for key, args in self.target[loc].items():
                target[loc][key.json_key()] = [arg.to_json() for arg in args]
        return {
            "name": self.name,
            "code_tokens": self.code_tokens,
            "source": source,
            "target": target,
        }

    @classmethod
    def from_cf(cls, cf: CollectedFunction, **kwargs):
        """Convert from a decoded CollectedFunction"""
        raw_code = cf.decompiler.raw_code
        code_tokens = tokenize_raw_code(raw_code)
        name = cf.decompiler.name

        source_locals = Example.filter(cf.decompiler.local_vars)
        source_args = Example.filter(cf.decompiler.arguments)
        target_locals = Example.filter(cf.debug.local_vars)
        target_args = Example.filter(cf.debug.arguments)

        valid = (
            name == cf.debug.name
            and set(source_args.keys()) == set(target_args.keys())
            and len(source_args) + len(source_locals) > 0
            and len(code_tokens) < 500
        )

        return cls(
            name,
            code_tokens,
            {"a": source_args, "l": source_locals},
            {"a": target_args, "l": target_locals},
            kwargs["binary_file"],
            valid,
            raw_code,
        )

    @staticmethod
    def filter(mapping: Mapping[Location, Set[Variable]]):
        """Discard and leave these for future work:

        Register locations
        Locations are reused for multiple variables
        """
        ret: Mapping[Location, Set[Variable]] = {}
        for location, variable_set in mapping.items():
            if location.json_key().startswith("r"):
                continue
            if len(variable_set) > 1:
                continue
            ret[location] = variable_set
        return ret

    @property
    def is_valid_example(self):
        return self._is_valid


class Dataset(wds.Dataset):

    def __init__(self, url, vocab_fname=None):
        # support wildcards
        urls = glob.glob(url)
        super().__init__(urls)
        if vocab_fname:
            # annotate example for training
            from utils.vocab import Vocab
            self.vocab = Vocab.load(vocab_fname)
            annotate_and_filter = self._annotate_and_filter
        else:
            # for creating the vocab
            annotate_and_filter = lambda x: x
        self = (
            self.pipe(Dataset._file_iter_to_line_iter)
            .decode()
            .map(lambda x: x["json"])
            .map(Example.from_json)
            .pipe(annotate_and_filter)
            .shuffle(5000)
        )

    @staticmethod
    def _file_iter_to_line_iter(jsonl_iter):
        for jsonl in jsonl_iter:
            lines = jsonl["jsonl"].split(b"\n")
            for line in lines:
                if not line:
                    continue
                yield {"json": line}

    def _annotate_and_filter(self, example_iter):
        for example in example_iter:
            src_bpe_model = self.vocab.source_tokens.subtoken_model
            snippet = example.code_tokens
            snippet = ' '.join(snippet)
            sub_tokens = ['<s>'] + src_bpe_model.encode_as_pieces(snippet) + ['</s>']
            sub_token_ids = [src_bpe_model.bos_id()] + src_bpe_model.encode_as_ids(snippet) + [src_bpe_model.eos_id()]
            setattr(example, 'sub_tokens', sub_tokens)
            setattr(example, 'sub_token_ids', sub_token_ids)
            setattr(example, 'source_seq_length', len(sub_tokens))
            if len(sub_tokens) < 510:
                yield example

    @staticmethod
    def collate_fn(batch_examples: List[Example]):
        print([len(e.code_tokens) for e in batch_examples])


if __name__ == "__main__":
    dataset = Dataset("data/train-shard-*.tar", "data/vocab.bpe10000")
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=4, batch_size=16, collate_fn=Dataset.collate_fn
    )
    for x in tqdm(dataloader):
        pass
