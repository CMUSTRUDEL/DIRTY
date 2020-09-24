import glob
import pickle
import sys
import os
import gc
import time
import ujson as json
import tarfile
from typing import Iterable, List, Dict, Union, Tuple
import multiprocessing
import threading
import queue

from tqdm import tqdm
import numpy as np

class Example(object):
    def __init__(self, ea: str, num_vars: int, code_tokens: str, **kwargs):
        self.ea = ea
        self.num_vars = num_vars
        self.code_tokens = code_tokens

        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def from_json_dict(cls, json_dict, **kwargs):

        target_args = json_dict["b"]["a"]
        target_locals = json_dict["b"]["l"]

        source_args = json_dict["c"]["a"]
        source_locals = json_dict["c"]["l"]

        num_vars = 0
        for loc in target_args:
            if loc in source_args:
                num_vars += 1
        for loc in target_locals:
            if loc in source_locals:
                num_vars += 1

        ea = json_dict["e"]
        code_tokens = json_dict["code_tokens"]

        if 'test_meta' in json_dict:
            kwargs['test_meta'] = json_dict['test_meta']

        return cls(ea, num_vars, code_tokens, **kwargs)
