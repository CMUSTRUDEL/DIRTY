#!/usr/bin/env python
"""
Usage:
    vocab.py [options] TRAIN_FILE VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --use-bpe                  Use bpe
    --size=<int>               vocab size [default: 10000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from collections import Counter
from itertools import chain

from docopt import docopt
import json
import sentencepiece as spm

from utils.grammar import Grammar


SAME_VARIABLE_TOKEN = '<IDENTITY>'
END_OF_VARIABLE_TOKEN = '</s>'
PAD_ID = 0


class VocabEntry:
    def __init__(self, subtoken_model_path=None):
        self.word2id = dict()

        self.subtoken_model_path = subtoken_model_path
        if subtoken_model_path:
            self.subtoken_model = spm.SentencePieceProcessor()
            self.subtoken_model.load(subtoken_model_path)

            subtoken_model = \
                subtoken_model_path[:subtoken_model_path.rfind('.model')]
            vocab_path = f'{subtoken_model}.vocab'
            for i, line in enumerate(open(vocab_path)):
                word, prob = line.strip().split('\t')
                self.word2id[word] = len(self.word2id)
        else:
            self.subtoken_model = None
            self.word2id['<pad>'] = PAD_ID
            self.word2id['<s>'] = 1
            self.word2id['</s>'] = 2
            self.word2id['<unk>'] = 3
            self.word2id[SAME_VARIABLE_TOKEN] = 4

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    @property
    def unk_id(self):
        return self.word2id['<unk>']

    def is_unk(self, word):
        return word not in self.word2id

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    @property
    def params(self):
        params = dict(unk_id=self.unk_id, word2id=self.word2id,
                      subtoken_model_path=self.subtoken_model_path)
        if 'word_freq' in self.__dict__:
            params['word_freq'] = self.word_freq

        return params

    def save(self, path):
        json.dump(self.params, open(path, 'w'), indent=2)

    @classmethod
    def load(cls, path=None, params=None):
        if path:
            params = json.load(open(path, 'r'))
        else:
            assert params, 'Params must be given when path is None!'

        if 'subtoken_model_path' in params:
            subtoken_model_path = params['subtoken_model_path']
        else:
            subtoken_model_path = None

        entry = VocabEntry(subtoken_model_path)

        setattr(entry, 'word2id', params['word2id'])
        setattr(entry, 'id2word', {v: k for k, v in params['word2id'].items()})
        if 'word_freq' in params:
            setattr(entry, 'word_freq', params['word_freq'])

        return entry

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=0, predefined_tokens=None):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        freq_words = [w for w in word_freq if word_freq[w] >= freq_cutoff]
        print(f'number of word types: {len(word_freq)}, '
              f'number of word types w/ frequency >= {freq_cutoff}: '
              f'{freq_words}')
        top_k_words = sorted(
            word_freq,
            key=lambda x: (-word_freq[x], x)
        )[:size]
        print('top 30 words: %s' % ', '.join(top_k_words[:30]))

        if predefined_tokens:
            for token in predefined_tokens:
                vocab_entry.add(token)

        for word in top_k_words:
            if len(vocab_entry) < size:
                if word_freq[word] >= freq_cutoff:
                    vocab_entry.add(word)

        # store the work frequency table in the
        setattr(vocab_entry, 'word_freq', word_freq)

        return vocab_entry


class Vocab(object):
    def __init__(self, **kwargs):
        self.entries = []
        for key, item in kwargs.items():
            self.__setattr__(key, item)

            self.entries.append(key)

    def __repr__(self):
        words = ', '.join(f'{entry} {getattr(self, entry)}words'
                          for entry in self.entries)
        return f'Vocab({words})'

    @property
    def params(self):
        params = dict()
        for key in self.entries:
            params[key] = getattr(self, key).params

        return params

    def save(self, path):
        json.dump(self.params, open(path, 'w'), indent=2)

    @classmethod
    def load(cls, path):
        params = json.load(open(path, 'r'))
        entries = dict()
        for key, val in params.items():
            if key in ('grammar', ):
                entry = Grammar.load(val)
            else:
                entry = VocabEntry.load(params=val)
            entries[key] = entry
        return cls(**entries)


if __name__ == '__main__':
    from utils.dataset import Dataset

    args = docopt(__doc__)
    vocab_size = int(args['--size'])
    vocab_file = args['VOCAB_FILE']
    train_set = Dataset(args['TRAIN_FILE'])

    src_code_tokens_file = vocab_file + '.src_code_tokens.txt'
    src_preserved_tokens = set()
    f_src_token = open(src_code_tokens_file, 'w')

    # extract vocab and node types
    node_types = set()
    src_words = []
    tgt_words = []
    identifier_names = []
    type_tokens = []
    for example in train_set.get_iterator(progress=True, num_workers=5):
        for node in example.ast:
            node_types.add(node.node_type)

            if node.is_variable_node:
                old_var_name = node.old_name
                new_var_name = node.new_name

                src_words.append(old_var_name)

                if old_var_name != new_var_name:
                    tgt_words.append(new_var_name)

            if node.node_type == 'obj' \
               or node.node_type == 'block' \
               and hasattr(node, 'name'):
                identifier_names.append(node.name)

            if hasattr(node, 'type_tokens'):
                type_tokens.extend(node.type_tokens)

        code_tokens = example.code_tokens
        preserved_tokens = [token for token in code_tokens
                            if token.startswith('@@') and token.endswith('@@')]
        src_preserved_tokens.update(preserved_tokens)
        f_src_token.write(' '.join(code_tokens) + '\n')

    f_src_token.close()

    print('building source words vocabulary')
    src_var_vocab_entry = VocabEntry.from_corpus(
        [src_words], size=vocab_size, freq_cutoff=int(args['--freq-cutoff'])
    )

    if args['--use-bpe']:
        print('use bpe')

        print('building source code tokens vocabulary')
        # train subtoken models
        src_preserved_tokens = ','.join(src_preserved_tokens)
        spm.SentencePieceTrainer.Train(
            '--add_dummy_prefix=false '
            f'--pad_id={PAD_ID} '
            '--bos_id=1 '
            '--eos_id=2 '
            '--unk_id=3 '
            f'--user_defined_symbols={src_preserved_tokens} '
            f'--vocab_size={vocab_size} '
            f'--model_prefix={vocab_file}.src_code_tokens '
            '--model_type=bpe '
            f'--input={src_code_tokens_file}'
        )
        src_code_tokens_vocab_entry = \
            VocabEntry(f'{vocab_file}.src_code_tokens.model')

        print('building target words vocabulary')
        tgt_word_file = args['VOCAB_FILE'] + '.var_names.txt'
        with open(tgt_word_file, 'w') as f:
            for name in tgt_words:
                f.write(name + '\n')

        spm.SentencePieceTrainer.Train(
            '--add_dummy_prefix=false '
            f'--pad_id={PAD_ID} '
            '--bos_id=1 '
            '--eos_id=2 '
            '--unk_id=3 '
            '--control_symbols=<IDENTITY> '
            f'--vocab_size={vocab_size} '
            f'--model_prefix={vocab_file}.tgt '
            '--model_type=bpe '
            f'--input={tgt_word_file}'
        )
        tgt_var_vocab_entry = VocabEntry(f'{vocab_file}.tgt.model')
    else:
        tgt_var_vocab_entry = VocabEntry.from_corpus(
            [tgt_words],
            size=vocab_size,
            freq_cutoff=int(args['--freq-cutoff']),
            predefined_tokens=[SAME_VARIABLE_TOKEN]
        )

    id_names_file = vocab_file + '.id_names.txt'
    with open(id_names_file, 'w') as f:
        for name in identifier_names:
            f.write(name + '\n')

    print('train subtoken model for obj names')
    # train subtoken models
    spm.SentencePieceTrainer.Train(
        '--add_dummy_prefix=false '
        f'--pad_id={PAD_ID} '
        '--bos_id=1 '
        '--eos_id=2 '
        '--unk_id=3 '
        '--control_symbols=<IDENTITY> '
        f'--vocab_size={vocab_size} '
        f'--model_prefix={vocab_file}.obj_name '
        '--model_type=bpe '
        f'--input={id_names_file}'
    )
    obj_name_vocab_entry = VocabEntry(f'{vocab_file}.obj_name.model')

    type_vocab = Counter(type_tokens)
    num_types = 100
    var_types = []
    for type_token, freq in type_vocab.items():
        if freq > 100:
            print(type_token, freq)
            var_types.append(type_token)

    print('init node types and variable types')
    grammar = Grammar(node_types, var_types)

    print('Node types:', node_types)
    print('Variable types:', var_types)

    vocab = Vocab(source=src_var_vocab_entry,
                  source_tokens=src_code_tokens_vocab_entry,
                  target=tgt_var_vocab_entry,
                  obj_name=obj_name_vocab_entry,
                  grammar=grammar)

    vocab.save(args['VOCAB_FILE'])
