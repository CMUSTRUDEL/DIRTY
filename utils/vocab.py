#!/usr/bin/env python
"""
Usage:
    vocab.py [options] TRAIN_FILE VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --size=<int>               vocab size [default: 10000]
    --freq_cutoff=<int>        frequency cutoff [default: 2]
"""

from collections import Counter
from itertools import chain

import torch
from docopt import docopt
import json


class VocabEntry:
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.id2word = {v: k for k, v in self.word2id.items()}

        # insert 100 indexed unks
        for i in range(100):
            self.add('UNK_%d' % i)

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

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

    def save(self, path):
        params = dict(unk_id=self.unk_id, word2id=self.word2id, word_freq=self.word_freq)
        json.dump(params, open(path, 'w'), indent=2)

    @staticmethod
    def load(path):
        entry = VocabEntry()
        params = json.load(open(path, 'r'))

        setattr(entry, 'unk_id', params['unk_id'])
        setattr(entry, 'word2id', params['word2id'])
        setattr(entry, 'word_freq', params['word_freq'])
        setattr(entry, 'id2word', {v: k for k, v in params['word2id'].items()})

        return entry

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=0):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        freq_words = [w for w in word_freq if word_freq[w] >= freq_cutoff]
        print('number of word types: %d, number of word types w/ frequency >= %d: %d' % (len(word_freq), freq_cutoff,
                                                                                       len(freq_words)))

        top_k_words = sorted(word_freq, key=lambda x: (-word_freq[x], x))[:size]
        print('top 10 words: %s' % ', '.join(top_k_words[:10]))

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
            assert isinstance(item, VocabEntry)
            self.__setattr__(key, item)

            self.entries.append(key)

    def __repr__(self):
        return 'Vocab(%s)' % (', '.join('%s %swords' % (entry, getattr(self, entry)) for entry in self.entries))


if __name__ == '__main__':
    from utils.dataset import DataSet

    args = docopt(__doc__)
    train_set = DataSet.load_from_jsonl(args['TRAIN_FILE'])
    corpus = [change.previous_code_chunk + change.updated_code_chunk + change.context for change in train_set]

    vocab_entry = VocabEntry.from_corpus(corpus, size=int(args['--size']), freq_cutoff=int(args['--freq_cutoff']))
    print('built vocabulary %s' % vocab_entry)

    torch.save(vocab_entry, open(args['VOCAB_FILE'], 'wb'))
