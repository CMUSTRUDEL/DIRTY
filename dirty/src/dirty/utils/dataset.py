import glob
import json
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union

import _jsonnet
import torch
import webdataset as wds
from csvnpm.binary.dire_types import Disappear, TypeLibCodec
from csvnpm.binary.function import CollectedFunction
from csvnpm.binary.variable import (
    Location,
    Register,
    Stack,
    Variable,
    location_from_json_key,
)
from torch.nn.utils.rnn import pad_sequence

from dirty.utils.code_processing import tokenize_raw_code


class Example:
    def __init__(
        self,
        name: str,
        code_tokens: str,
        source: Mapping[Location, Variable],
        target: Mapping[Location, Variable],
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
        source = {
            location_from_json_key(loc): Variable.from_json(var)
            for loc, var in d["source"].items()
        }
        target = {
            location_from_json_key(loc): Variable.from_json(var)
            for loc, var in d["target"].items()
        }
        return cls(
            d["name"],
            d["code_tokens"],
            source,
            target,
            test_meta=d.get("test_meta", None),
            binary=d.get("binary", None),
        )

    def to_json(self):
        assert self._is_valid
        source = {loc.json_key(): var.to_json() for loc, var in self.source.items()}
        target = {loc.json_key(): var.to_json() for loc, var in self.target.items()}
        return {
            "name": self.name,
            "code_tokens": self.code_tokens,
            "source": source,
            "target": target,
        }

    @classmethod
    def from_cf(cls, cf: CollectedFunction, **kwargs):
        """Convert from a decoded CollectedFunction"""
        name = cf.decompiler.name
        raw_code = cf.decompiler.raw_code
        code_tokens = tokenize_raw_code(raw_code)

        source: Dict[Location, Set[Variable]] = {
            **cf.decompiler.local_vars,
            **cf.decompiler.arguments,
        }
        target: Dict[Location, Set[Variable]] = {
            **cf.debug.local_vars,
            **cf.debug.arguments,
        }

        # Remove variables that overlap on memory or don't appear in the code
        # tokens
        source_code_tokens_set = set(code_tokens)
        target_code_tokens_set = set(tokenize_raw_code(cf.debug.raw_code))

        filtered_source = Example.filter(source, source_code_tokens_set)
        filtered_target = Example.filter(
            target, target_code_tokens_set, set(source.keys())
        )

        # Assign type "Disappear" to variables not existing in the ground truth
        varnames = set()
        for loc in filtered_source.keys():
            if loc not in filtered_target.keys():
                filtered_target[loc] = Variable(Disappear(), "", False)
        # Add special tokens to variables  to prevnt being sub-tokenized in BPE
        for var in filtered_source.values():
            varname = var.name
            varnames.add(varname)
        for idx in range(len(code_tokens)):
            if code_tokens[idx] in varnames:
                code_tokens[idx] = f"@@{code_tokens[idx]}@@"

        return cls(
            name,
            code_tokens,
            filtered_source,
            filtered_target,
            kwargs["binary_file"],
            valid=(name == cf.debug.name and bool(filtered_source)),
            raw_code=raw_code,
        )

    @staticmethod
    def filter(
        mapping: Dict[Location, Set[Variable]],
        code_tokens: Optional[Set[str]] = None,
        locations: Optional[Set[Location]] = None,
    ) -> Dict[Location, Variable]:
        """Discard and leave these for future work:

        Multiple variables sharing a memory location (no way to determine ground truth);
        Variables not appearing in code (no way to get representation);
        Target variables not appearing in source (useless ground truth);
        """
        ret: Dict[Location, Variable] = {}
        for location, variable_set in mapping.items():
            if len(variable_set) > 1:
                continue
            var = list(variable_set)[0]
            if code_tokens is not None and var.name not in code_tokens:
                continue
            if locations is not None and location not in locations:
                continue
            ret[location] = var
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

    def __init__(self, url: str, config: Optional[Dict] = None, percent: float = 1.0):
        # support wildcards
        urls = sorted(glob.glob(url))
        urls = urls[: int(percent * len(urls))]
        super().__init__(urls)
        if config:
            # annotate example for training
            from dirty.utils.vocab import Vocab

            self.vocab = Vocab.load(config["vocab_file"])
            with open(config["typelib_file"]) as type_f:
                self.typelib = TypeLibCodec.decode(type_f.read())
            self.max_src_tokens_len = config["max_src_tokens_len"]
            self.max_num_var = config["max_num_var"]
            annotate = self._annotate
            self.rename = config.get("rename", False)
            # sort = Dataset._sort
            sort = identity
        else:
            # for creating the vocab
            annotate = identity
            sort = identity
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
            yield from sort_pool[len(sort_pool_new) :]
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
                json_line["binary"] = jsonl["__key__"][: jsonl["__key__"].index("_")]
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
        src_var_types_id = []
        src_var_types_str = []
        tgt_var_types_id = []
        tgt_var_types_str = []
        tgt_var_subtypes = []
        tgt_var_type_sizes = []
        tgt_var_type_objs = []
        tgt_var_src_mems = []
        tgt_names = []
        # variables on registers first, followed by those on stack
        locs = sorted(
            example.source,
            key=lambda x: sub_tokens.index(f"@@{example.source[x].name}@@")
            if f"@@{example.source[x].name}@@" in sub_tokens
            else self.max_src_tokens_len,
        )
        stack_pos = [x.offset for x in example.source if isinstance(x, Stack)]
        stack_start_pos = max(stack_pos) if stack_pos else None
        for loc in locs[: self.max_num_var]:
            src_var = example.source[loc]
            tgt_var = example.target[loc]
            src_var_names.append(f"@@{src_var.name}@@")
            tgt_var_names.append(f"@@{tgt_var.name}@@")
            src_var_types_id.append(types_model.lookup_decomp(str(src_var.typ)))
            src_var_types_str.append(str(src_var.typ))
            tgt_var_types_id.append(types_model[str(tgt_var.typ)])
            tgt_var_types_str.append(str(tgt_var.typ))
            if types_model[str(tgt_var.typ)] == types_model.unk_id:
                subtypes = [subtypes_model.unk_id, subtypes_model["<eot>"]]
            else:
                subtypes = [subtypes_model[subtyp] for subtyp in tgt_var.typ.tokenize()]
            tgt_var_type_sizes.append(len(subtypes))
            tgt_var_subtypes += subtypes
            tgt_var_type_objs.append(tgt_var.typ)
            # Memory
            # 0: absolute location of the variable in the function, e.g.,
            #   for registers: Reg 56
            #   for stack: relative position to the first variable
            # 1: size of the type
            # 2, 3, ...: start offset of fields in the type

            def var_loc_in_func(loc):
                # TODO: fix the magic number for computing vocabulary idx
                if isinstance(loc, Register):
                    return 1030 + self.vocab.regs[loc.name]
                else:
                    from dirty.utils.vocab import VocabEntry

                    return (
                        3 + stack_start_pos - loc.offset
                        if stack_start_pos - loc.offset < VocabEntry.MAX_STACK_SIZE
                        else 2
                    )

            tgt_var_src_mems.append(
                [var_loc_in_func(loc)]
                + types_model.encode_memory(
                    (src_var.typ.size,) + src_var.typ.start_offsets()
                )
            )
            tgt_names.append(tgt_var.name)

        setattr(example, "src_var_names", src_var_names)
        setattr(example, "tgt_var_names", tgt_var_names)
        if self.rename:
            setattr(
                example,
                "tgt_var_name_ids",
                [self.vocab.names[n[2:-2]] for n in tgt_var_names],
            )
        setattr(example, "src_var_types", src_var_types_id)
        setattr(example, "src_var_types_str", src_var_types_str)
        setattr(example, "tgt_var_types", tgt_var_types_id)
        setattr(example, "tgt_var_types_str", tgt_var_types_str)
        setattr(example, "tgt_var_subtypes", tgt_var_subtypes)
        setattr(example, "tgt_var_type_sizes", tgt_var_type_sizes)
        setattr(example, "tgt_var_src_mems", tgt_var_src_mems)

        return example

    @staticmethod
    def collate_fn(
        examples: List[Example],
    ) -> Tuple[
        Dict[str, Union[torch.Tensor, int]],
        Dict[str, Union[torch.Tensor, List]],
    ]:
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

        src_type_ids = [
            torch.tensor(e.src_var_types, dtype=torch.long) for e in examples
        ]
        src_type_id = pad_sequence(src_type_ids, batch_first=True)
        type_ids = [torch.tensor(e.tgt_var_types, dtype=torch.long) for e in examples]
        target_type_id = pad_sequence(type_ids, batch_first=True)
        assert target_type_id.shape == variable_mention_num.shape

        subtype_ids = [
            torch.tensor(e.tgt_var_subtypes, dtype=torch.long) for e in examples
        ]
        target_subtype_id = pad_sequence(subtype_ids, batch_first=True)
        type_sizes = [
            torch.tensor(e.tgt_var_type_sizes, dtype=torch.long) for e in examples
        ]
        target_type_sizes = pad_sequence(type_sizes, batch_first=True)

        target_mask = src_type_id > 0
        target_type_src_mems = [
            torch.tensor(mems, dtype=torch.long)
            for e in examples
            for mems in e.tgt_var_src_mems
        ]
        target_type_src_mems = pad_sequence(target_type_src_mems, batch_first=True)
        target_type_src_mems_unflattened = torch.zeros(
            *target_mask.shape, target_type_src_mems.size(-1), dtype=torch.long
        )
        target_type_src_mems_unflattened[target_mask] = target_type_src_mems
        target_type_src_mems = target_type_src_mems_unflattened

        # renaming task
        if hasattr(examples[0], "tgt_var_name_ids"):
            name_ids = [
                torch.tensor(e.tgt_var_name_ids, dtype=torch.long) for e in examples
            ]
            target_name_id = pad_sequence(name_ids, batch_first=True)
        else:
            target_name_id = None

        return (
            dict(
                index=sum(
                    [
                        [(e.binary, e.name, name) for name in e.src_var_names]
                        for e in examples
                    ],
                    [],
                ),
                src_code_tokens=input,
                variable_mention_to_variable_id=variable_mention_to_variable_id,
                variable_mention_mask=variable_mention_mask,
                variable_mention_num=variable_mention_num,
                variable_encoding_mask=variable_encoding_mask,
                target_type_src_mems=target_type_src_mems,
                src_type_id=src_type_id,
                target_mask=target_mask,
                target_submask=target_subtype_id > 0,
                target_type_sizes=target_type_sizes,
            ),
            dict(
                tgt_var_names=sum([e.tgt_var_names for e in examples], []),
                target_type_id=target_type_id,
                target_name_id=target_name_id,
                target_subtype_id=target_subtype_id,
                target_mask=target_mask,
                test_meta=[e.test_meta for e in examples],
            ),
        )


if __name__ == "__main__":
    config = json.loads(_jsonnet.evaluate_file("config.xfmr.jsonnet"))
    dataset = Dataset("data1/dev-*.tar", config["data"])
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=8, batch_size=64, collate_fn=Dataset.collate_fn
    )
    for x in dataloader:
        pass
