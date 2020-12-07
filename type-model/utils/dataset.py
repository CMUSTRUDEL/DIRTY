import glob
import json
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union

import _jsonnet
import numpy as np
import torch
import webdataset as wds
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils.code_processing import tokenize_raw_code
from utils.function import CollectedFunction, Function
from utils.variable import Location, Variable, location_from_json_key
from utils.dire_types import Struct, TypeLibCodec, TypeLib, UDT, TypeInfo


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
        test_meta: Dict[str, Dict[str, bool]] = None,
        binary: str = None,
    ):
        self.name = name
        self.code_tokens = code_tokens
        self.source = source
        self.target = target
        self.binary_file = binary_file
        self._is_valid = valid
        self.raw_code = raw_code
        self.test_meta = test_meta
        self.binary = binary

    @classmethod
    def from_json(cls, d: Dict):
        source = defaultdict(dict)
        for loc in ["a", "l"]:
            if not loc in d["source"]:
                continue
            for key, args in d["source"][loc].items():
                source[loc][location_from_json_key(key)] = {
                    Variable.from_json(arg) for arg in args
                }
        target = defaultdict(dict)
        for loc in ["a", "l"]:
            if not loc in d["target"]:
                continue
            for key, args in d["target"][loc].items():
                target[loc][location_from_json_key(key)] = {
                    Variable.from_json(arg) for arg in args
                }
        return cls(d["name"], d["code_tokens"], source, target, test_meta=d.get("test_meta", None), binary=d.get("binary", None))

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

        # Remove variables that overlaps on a memory location or don't appear in the code tokens
        code_tokens_set = set(code_tokens)
        source_locals = Example.filter(cf.decompiler.local_vars, code_tokens_set)
        source_args = Example.filter(cf.decompiler.arguments, code_tokens_set)
        target_locals = Example.filter(
            cf.debug.local_vars, None, set(source_locals.keys())
        )
        target_args = Example.filter(cf.debug.arguments, None, set(source_args.keys()))

        # Add special tokens to variables in tokens to prevent being sub-tokenized in BPE
        varnames = set()
        for _, vars in {**source_locals, **source_args}.items():
            for var in list(vars):
                varname = var.name
                varnames.add(varname)
        for idx in range(len(code_tokens)):
            if code_tokens[idx] in varnames:
                code_tokens[idx] = f"@@{code_tokens[idx]}@@"

        valid = (
            name == cf.debug.name
            # and set(source_args.keys()) == set(target_args.keys())
            and set(source_locals.keys()) == set(target_locals.keys())
            # and len(source_args) + len(source_locals) > 0
            and len(source_locals) > 0
            and len(code_tokens) < 500
        )
        # Remove functions that are too large
        tgt_var_type_objs = [list(target_locals[key])[0].typ for key in target_locals]
        valid &= all([m.size <= kwargs["max_type_size"] for m in tgt_var_type_objs])
        if valid:
            src_a, src_s, _ = Function.stack_layout(source_locals)
            tgt_a, tgt_s, t_overlap = Function.stack_layout(target_locals)
            valid &= bool(len(source_locals) > 0 and tgt_a and tgt_a[-1] <= kwargs["max_stack_length"] and src_a and src_a[-1] <= kwargs["max_stack_length"] and tgt_s and tgt_s[-1] <= kwargs["max_stack_length"] and tgt_s[0] >= 0 and src_s and src_s[-1] <= kwargs["max_stack_length"] and src_s[0] >= 0)
            valid &= not t_overlap

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
    def filter(
        mapping: Mapping[Location, Set[Variable]],
        code_tokens: Optional[Set[str]] = None,
        locations: Optional[Set[Location]] = None,
    ):
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
            if code_tokens and not list(variable_set)[0].name in code_tokens:
                continue
            # HACK: discard padding for now
            if isinstance(list(variable_set)[0].typ, UDT.Padding):
                continue
            if locations and not location in locations:
                continue
            ret[location] = variable_set
        return ret

    @property
    def is_valid_example(self):
        return self._is_valid

# HACK: Stupid global lambda functions required for distributed data loading
def identity(x):
    return x
def get_src_len(e):
    return e.source_seq_length

class Dataset(wds.Dataset):

    SHUFFLE_BUFFER = 5000
    SORT_BUFFER = 512

    def __init__(self, url: str, config: Optional[Dict] = None):
        # support wildcards
        urls = glob.glob(url)
        super().__init__(urls)
        if config:
            # annotate example for training
            from utils.vocab import Vocab

            self.vocab = Vocab.load(config["vocab_file"])
            with open(config["typelib_file"]) as type_f:
                self.typelib = TypeLibCodec.decode(type_f.read())
            self.max_src_tokens_len = config["max_src_tokens_len"]
            assert not config["args"] # deal with local variables only for now
            self.locations = ["a", "l"] if config["args"] else ["l"]
            annotate = self._annotate
            # sort = Dataset._sort
            sort = identity
        else:
            # for creating the vocab
            annotate = identity
            sort = identity
            self.locations = ["a", "l"]
        self = (
            self.pipe(Dataset._file_iter_to_line_iter)
            .map(Example.from_json)
            .map(annotate)
            .shuffle(Dataset.SHUFFLE_BUFFER)
            .pipe(sort)
        )

    @staticmethod
    def _sort(example_iter):
        sort_pool = []
        sort_pool_new = []
        for example in example_iter:
            if sort_pool:
                yield sort_pool[len(sort_pool_new)]
            sort_pool_new.append(example)
            if len(sort_pool_new) == Dataset.SORT_BUFFER:
                sort_pool_new.sort(key=get_src_len)
                sort_pool = sort_pool_new
                sort_pool_new = []
        if sort_pool:
            yield from sort_pool[len(sort_pool_new):]
        if sort_pool_new:
            sort_pool_new.sort(key=get_src_len)
            yield from sort_pool

    @staticmethod
    def _file_iter_to_line_iter(jsonl_iter):
        for jsonl in jsonl_iter:
            lines = jsonl["jsonl"].split(b"\n")
            for line in lines:
                if not line:
                    continue
                json_line = json.loads(line)
                json_line["binary"] = jsonl["__key__"][:jsonl["__key__"].index("_")]
                yield json_line

    def _annotate(self, example: Example):
        src_bpe_model = self.vocab.source_tokens.subtoken_model
        snippet = example.code_tokens
        snippet = " ".join(snippet)
        sub_tokens = (
            ["<s>"]
            + src_bpe_model.encode_as_pieces(snippet)[: self.max_src_tokens_len]
            + ["</s>"]
        )
        sub_token_ids = (
            [src_bpe_model.bos_id()]
            + src_bpe_model.encode_as_ids(snippet)[: self.max_src_tokens_len]
            + [src_bpe_model.eos_id()]
        )
        setattr(example, "sub_tokens", sub_tokens)
        setattr(example, "sub_token_ids", sub_token_ids)
        setattr(example, "source_seq_length", len(sub_tokens))

        types_model = self.vocab.types
        subtypes_model = self.vocab.subtypes
        src_var_names = []
        tgt_var_names = []
        tgt_var_types = []
        tgt_var_subtypes = []
        tgt_var_type_sizes = []
        tgt_var_type_objs = []
        tgt_var_src_mems = []
        tgt_names = []
        for loc in self.locations:
            for key in sorted(example.source[loc], key=lambda x: x.offset):
                src_var = list(example.source[loc][key])[0]
                tgt_var = list(example.target[loc][key])[0]
                src_var_names.append(f"@@{src_var.name}@@")
                tgt_var_names.append(f"@@{tgt_var.name}@@")
                tgt_var_types.append(types_model[str(tgt_var.typ)])
                if types_model[str(tgt_var.typ)] == types_model.unk_id:
                    subtypes = [subtypes_model.unk_id, subtypes_model["<eot>"]]
                else:
                    subtypes = [subtypes_model[subtyp] for subtyp in tgt_var.typ.tokenize()]
                tgt_var_type_sizes.append(len(subtypes))
                tgt_var_subtypes += subtypes
                tgt_var_type_objs.append(tgt_var.typ)
                tgt_var_src_mems.append(types_model.encode_memory(src_var.typ.start_offsets() + (src_var.typ.size,)))
                tgt_names.append(tgt_var.name)
        
        src_a, src_s, _ = Function.stack_layout(example.source["l"])
        tgt_a, tgt_s, _ = Function.stack_layout(example.target["l"])
        setattr(example, "src_var_names", src_var_names)
        setattr(example, "tgt_var_names", tgt_var_names)
        setattr(example, "tgt_var_types", tgt_var_types)
        setattr(example, "tgt_var_subtypes", tgt_var_subtypes)
        setattr(example, "tgt_var_type_sizes", tgt_var_type_sizes)
        setattr(example, "mems", (src_a, src_s))
        setattr(example, "tgt_var_src_mems", tgt_var_src_mems)

        return example

    @staticmethod
    def collate_fn(examples: List[Example]) -> Tuple[Dict[str, Union[torch.Tensor, int]], Dict[str, Union[torch.Tensor, List]]]:
        token_ids = [torch.tensor(e.sub_token_ids) for e in examples]
        input = pad_sequence(token_ids, batch_first=True)
        max_time_step = input.shape[1]
        # corresponding var_id of each token in sub_tokens
        variable_mention_to_variable_id = torch.zeros(
            len(examples), max_time_step, dtype=torch.long
        )
        # if each token in sub_tokens is a variable
        variable_mention_mask = torch.zeros(len(examples), max_time_step)
        # the number of mentioned times for each var_id
        variable_mention_num = torch.zeros(
            len(examples), max(len(e.src_var_names) for e in examples)
        )

        for e_id, example in enumerate(examples):
            var_name_to_id = {name: i for i, name in enumerate(example.src_var_names)}
            for i, sub_token in enumerate(example.sub_tokens):
                if sub_token in example.src_var_names:
                    var_id = var_name_to_id[sub_token]
                    variable_mention_to_variable_id[e_id, i] = var_id
                    variable_mention_mask[e_id, i] = 1.0
                    variable_mention_num[e_id, var_name_to_id[sub_token]] += 1
        # if mentioned for each var_id
        variable_encoding_mask = (variable_mention_num > 0).float()

        type_ids = [torch.tensor(e.tgt_var_types, dtype=torch.long) for e in examples]
        target_type_id = pad_sequence(type_ids, batch_first=True)
        assert target_type_id.shape == variable_mention_num.shape

        subtype_ids = [torch.tensor(e.tgt_var_subtypes, dtype=torch.long) for e in examples]
        target_subtype_id = pad_sequence(subtype_ids, batch_first=True)
        type_sizes = [torch.tensor(e.tgt_var_type_sizes, dtype=torch.long) for e in examples]
        target_type_sizes= pad_sequence(type_sizes, batch_first=True)

        target_type_src_mems = [torch.tensor(mems, dtype=torch.long) for e in examples for mems in e.tgt_var_src_mems]
        target_type_src_mems = pad_sequence(target_type_src_mems, batch_first=True)

        return (
            dict(
                index=sum([[(e.binary, e.name, name) for name in e.src_var_names] for e in examples], []),
                src_code_tokens=input,
                variable_mention_to_variable_id=variable_mention_to_variable_id,
                variable_mention_mask=variable_mention_mask,
                variable_mention_num=variable_mention_num,
                variable_encoding_mask=variable_encoding_mask,
                target_type_src_mems=target_type_src_mems,
                batch_size=len(examples),
            ),
            dict(
                tgt_var_names=sum([e.tgt_var_names for e in examples], []),
                target_type_id=target_type_id,
                target_subtype_id=target_subtype_id,
                target_type_sizes=target_type_sizes,
                target_mask=target_type_id > 0,
                target_submask = target_subtype_id > 0,
                target_mems=[e.mems for e in examples],
                target_type_src_mems=target_type_src_mems,
                test_meta=[e.test_meta for e in examples]
            ),
        )
    
    def __len__(self):
        """HACK: fake length for testing in pl"""
        return 10 ** 4


if __name__ == "__main__":
    config = json.loads(_jsonnet.evaluate_file('config.xfmr.jsonnet'))
    dataset = Dataset("data1/dev-*.tar", config["data"])
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
    cnt = 0
    preds_list = []
    targets_list = []
    for batch in tqdm(dataloader):
        input_dict, target_dict = batch
        targets = target_dict["target_type_id"][target_dict["target_mask"]]
        preds = []
        for mems, target in zip(target_dict["target_type_src_mems"], targets.tolist()):
            size = mems[mems != 0].tolist()[-1] - 3
            if size not in most_common_for_size:
                preds.append(types_model.unk_id)
                continue
            preds.append(most_common_for_size[size])
        preds_list.append(torch.tensor(preds))
        targets_list.append(targets)
    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)
    print(preds.shape, targets.shape)
    from pytorch_lightning.metrics.functional.classification import accuracy, f1_score
    import wandb
    wandb.init(name="most_common", project="dire")
    wandb.log({"test_acc": accuracy(preds, targets)})
    wandb.log({"test_f1_macro": f1_score(preds, targets, class_reduction='macro')})
    struct_set = set()
    for idx, type_str in types_model.id2word.items():
        if type_str.startswith("struct"):
            struct_set.add(idx)
    struc_mask = torch.zeros(len(targets), dtype=torch.bool)
    for idx, target in enumerate(targets):
        if target.item() in struct_set:
            struc_mask[idx] = 1
    wandb.log({"test_struc_acc": accuracy(preds[struc_mask], targets[struc_mask])})
