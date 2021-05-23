#!/usr/bin/env python
"""
Usage:
    vocab.py [options] TRAIN_FILE TYPE_FILE VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --use-bpe                  Use bpe
    --size=<int>               vocab size [default: 10000]
    --freq-cutoff=<int>        frequency cutoff [default: 5]
"""

import json
from collections import Counter
from itertools import chain

import sentencepiece as spm  # type: ignore
from csvnpm.binary.dire_types import TypeLibCodec
from csvnpm.binary.variable import Register
from docopt import docopt  # type: ignore
from tqdm import tqdm  # type: ignore

from dirty.utils.dataset import Dataset

SAME_VARIABLE_TOKEN = "<IDENTITY>"
END_OF_VARIABLE_TOKEN = "</s>"
PAD_ID = 0
assert PAD_ID == 0


class VocabEntry:
    def __init__(self, subtoken_model_path=None):
        self.word2id = dict()

        self.subtoken_model_path = subtoken_model_path
        if subtoken_model_path:
            self.subtoken_model = spm.SentencePieceProcessor()
            self.subtoken_model.load(subtoken_model_path)

            vocab_path = (
                subtoken_model_path[: subtoken_model_path.rfind(".model")] + ".vocab"
            )
            for i, line in enumerate(open(vocab_path)):
                word, prob = line.strip().split("\t")
                self.word2id[word] = len(self.word2id)
        else:
            self.subtoken_model = None
            self.word2id["<pad>"] = PAD_ID
            self.word2id["<s>"] = 1
            self.word2id["</s>"] = 2
            self.word2id["<unk>"] = 3
            self.word2id[SAME_VARIABLE_TOKEN] = 4

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    @property
    def unk_id(self):
        return self.word2id["<unk>"]

    def is_unk(self, word):
        return word not in self.word2id

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError("vocabulary is readonly")

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return "Vocabulary[size=%d]" % len(self)

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    @property
    def params(self):
        params = dict(
            unk_id=self.unk_id,
            word2id=self.word2id,
            subtoken_model_path=self.subtoken_model_path,
        )
        if hasattr(self, "word_freq"):
            params["word_freq"] = self.word_freq

        return params

    def save(self, path):
        json.dump(self.params, open(path, "w"), indent=2)

    @classmethod
    def load(cls, path=None, params=None):
        if path:
            params = json.load(open(path, "r"))
        else:
            assert params, "Params must be given when path is None!"

        if "subtoken_model_path" in params:
            subtoken_model_path = params["subtoken_model_path"]
        else:
            subtoken_model_path = None

        entry = VocabEntry(subtoken_model_path)

        setattr(entry, "word2id", params["word2id"])
        setattr(entry, "id2word", {v: k for k, v in params["word2id"].items()})
        if "word_freq" in params:
            setattr(entry, "word_freq", params["word_freq"])

        return entry

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=0, predefined_tokens=None):
        word_freq = Counter(chain(*corpus))
        return VocabEntry.from_counter(word_freq, size, freq_cutoff, predefined_tokens)

    @staticmethod
    def from_counter(word_freq, size, freq_cutoff=0, predefined_tokens=None):
        vocab_entry = VocabEntry()

        freq_words = [w for w in word_freq if word_freq[w] >= freq_cutoff]
        print(
            "number of word types: %d, number of word types w/ frequency >= %d: %d"
            % (len(word_freq), freq_cutoff, len(freq_words))
        )

        top_k_words = sorted(word_freq, key=lambda x: (-word_freq[x], x))[:size]
        print("top 30 words: %s" % ", ".join(top_k_words[:30]))

        if predefined_tokens:
            for token in predefined_tokens:
                vocab_entry.add(token)

        for word in top_k_words:
            if len(vocab_entry) < size:
                if word_freq[word] >= freq_cutoff:
                    vocab_entry.add(word)

        # store the work frequency table in the
        setattr(vocab_entry, "word_freq", word_freq)

        return vocab_entry

    MAX_MEM_LENGTH = 128
    MAX_STACK_SIZE = 1024

    @staticmethod
    def encode_memory(mems):
        """Encode memory to ids

        <pad>: 0
        <SEP>: 1
        <unk>: 2
        mem_id: mem_offset + 3
        """
        ret = []
        for mem in mems[: VocabEntry.MAX_MEM_LENGTH]:
            if mem == "<SEP>":
                ret.append(1)
            elif mem > VocabEntry.MAX_STACK_SIZE:
                ret.append(2)
            else:
                ret.append(3 + mem)
        return ret

    def lookup_decomp(self, type_str):
        if type_str in self:
            return self[type_str]
        if type_str.startswith("struct "):
            return self[type_str[7:]]
        return self[type_str]


class Vocab(object):
    def __init__(self, **kwargs):
        self.entries = []
        for key, item in kwargs.items():
            self.__setattr__(key, item)

            self.entries.append(key)

    def __repr__(self):
        return "Vocab(%s)" % (
            ", ".join(
                "%s %swords" % (entry, getattr(self, entry)) for entry in self.entries
            )
        )

    @property
    def params(self):
        params = dict()
        for key in self.entries:
            params[key] = getattr(self, key).params

        return params

    def save(self, path):
        json.dump(self.params, open(path, "w"), indent=2)

    @classmethod
    def load(cls, path):
        params = json.load(open(path, "r"))
        entries = dict()
        for key, val in params.items():
            # if key in ('grammar', ):
            #     entry = Grammar.load(val)
            # else:
            entry = VocabEntry.load(params=val)
            entries[key] = entry
        return cls(**entries)


if __name__ == "__main__":

    args = docopt(__doc__)
    vocab_size = int(args["--size"])
    vocab_file = args["VOCAB_FILE"]
    type_file = args["TYPE_FILE"]
    train_set = Dataset(args["TRAIN_FILE"])

    with open(type_file) as type_f:
        typelib = TypeLibCodec.decode(type_f.read())
        type_counter = Counter()
        subtype_counter = Counter()
        for size in typelib:
            for freq, tp in typelib[size]:
                # Treat types as discrete tokens
                type_counter[str(tp)] += freq
                # Tokenize compositonal types, mainly structs
                for subtype in tp.tokenize():
                    subtype_counter[subtype] += freq
    print(f"{len(type_counter)} types in typelib")
    print(f"{len(subtype_counter)} subtypes in typelib")

    type_vocab_entry = VocabEntry.from_counter(
        type_counter,
        size=len(type_counter),
        freq_cutoff=int(args["--freq-cutoff"]),
    )
    subtype_vocab_entry = VocabEntry.from_counter(
        subtype_counter,
        size=len(subtype_counter),
        freq_cutoff=int(args["--freq-cutoff"]),
    )

    src_code_tokens_file = vocab_file + ".src_code_tokens.txt"
    preserved_tokens = set()
    name_counter = Counter()
    reg_counter = Counter()
    with open(src_code_tokens_file, "w") as f_src_token:
        tgt_words = []
        for example in tqdm(train_set):
            code_tokens = example.code_tokens
            reg_counter.update(
                map(
                    lambda x: x.name,
                    filter(
                        lambda x: isinstance(x, Register),
                        example.target.keys(),
                    ),
                )
            )
            name_counter.update(map(lambda x: x.name, example.target.values()))
            for token in code_tokens:
                if token.startswith("@@") and token.endswith("@@"):
                    preserved_tokens.add(token)
            f_src_token.write(" ".join(code_tokens) + "\n")
    name_vocab_entry = VocabEntry.from_counter(
        name_counter,
        size=len(name_counter),
        freq_cutoff=int(args["--freq-cutoff"]),
    )
    reg_vocab_entry = VocabEntry.from_counter(
        reg_counter,
        size=len(reg_counter),
        freq_cutoff=int(args["--freq-cutoff"]),
    )

    assert args["--use-bpe"]
    print("use bpe")

    print("building source code tokens vocabulary")
    # train subtoken models
    preserved_tokens = ",".join(preserved_tokens)
    spm.SentencePieceTrainer.Train(
        f"--add_dummy_prefix=false --pad_id={PAD_ID} --bos_id=1 --eos_id=2 --unk_id=3 "
        f"--user_defined_symbols={preserved_tokens} "
        f"--vocab_size={vocab_size} "
        f"--model_prefix={vocab_file}.src_code_tokens --model_type=bpe "
        f"--input={src_code_tokens_file}"
    )
    src_code_tokens_vocab_entry = VocabEntry(vocab_file + ".src_code_tokens.model")

    vocab = Vocab(
        source_tokens=src_code_tokens_vocab_entry,
        types=type_vocab_entry,
        subtypes=subtype_vocab_entry,
        names=name_vocab_entry,
        regs=reg_vocab_entry,
    )

    vocab.save(args["VOCAB_FILE"])
