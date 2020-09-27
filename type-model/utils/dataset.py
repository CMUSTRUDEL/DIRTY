import multiprocessing as mp
import tarfile
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Mapping, Set
import glob

import numpy as np
import ujson as json
from tqdm import tqdm

from utils.code_processing import canonicalize_code, tokenize_raw_code
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
            if not loc in d: continue
            for key, args in d[loc].items():
                source[loc][location_from_json_key(key)] = {
                    Variable.from_json(arg) for arg in args
                }
        target = defaultdict(dict)
        for loc in ["a", "l"]:
            if not loc in d: continue
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

class Dataset(object):
    def __init__(self, file_paths):
        assert isinstance(file_paths, str)
        self.file_paths = glob.glob(file_paths)

    def get_iterator(self, shuffle=False, num_workers=1):
        json_iter = get_json_iterator_from_tar_file(self.file_paths, shuffle=shuffle)
        with mp.Pool(processes=num_workers) as pool:
            examples_iter = pool.imap(example_generator, json_iter)
            for example in examples_iter:
                yield example

def get_json_iterator_from_tar_file(file_paths, shuffle=False) -> Iterable:
    if shuffle:
        np.random.shuffle(file_paths)

    for file_path in file_paths:
        with tarfile.open(file_path, 'r') as f:
            files = [x.name for x in f.getmembers() if x.name.endswith('.jsonl')]
            if shuffle:
                np.random.shuffle(files)

            for filename in files:
                jsonl_file = f.extractfile(filename)
                if jsonl_file is not None:
                    for line_no, json_str in enumerate(jsonl_file):
                        payload = json_str, dict(file_name=filename, line_num=line_no)
                        yield payload

def example_generator(json_str_list):
    json_str, meta = json_str_list
    json_dict = json.loads(json_str)
    example = Example.from_json(json_dict)
    # if example.is_valid_example:
    #     canonical_code = canonicalize_code(example.raw_code)
    #     example.canonical_code = canonical_code
    return example