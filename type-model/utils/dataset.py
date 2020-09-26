from collections import defaultdict
from typing import DefaultDict, Dict, Mapping, Set

import numpy as np
import ujson as json
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
            for key, args in d[loc].items():
                source[loc][location_from_json_key(key)] = {
                    Variable.from_json(arg) for arg in args
                }
        target = defaultdict(dict)
        for loc in ["a", "l"]:
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
