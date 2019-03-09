from typing import Dict
import pickle

import torch
import torch.nn as nn

from utils import nn_util, util
from utils.vocab import Vocab, VocabEntry


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, *input):
        raise NotImplementedError


class SimpleDecoder(Decoder):
    def __init__(self, ast_node_encoding_size: int, vocab: Vocab):
        super(SimpleDecoder, self).__init__()

        self.vocab = vocab
        self.state2names = nn.Linear(ast_node_encoding_size, len(vocab.target), bias=True)
        self.config: Dict = None

    @classmethod
    def default_params(cls):
        return {
            'vocab_file': None,
            'ast_node_encoding_size': 128
        }

    @classmethod
    def build(cls, config):
        params = util.update(cls.default_params(), config)

        vocab = torch.load(params['vocab_file'])
        model = cls(params['ast_node_encoding_size'], vocab)
        model.config = params

        return model

    def forward(self, src_ast_encoding):
        """
        Given a batch of encoded ASTs, compute the log-likelihood of generating all possible renamings
        """
        # (all_var_node_num, tgt_vocab_size)
        logits = self.state2names(src_ast_encoding['variable_master_node_encoding'])
        batched_p_names = torch.log_softmax(logits, dim=-1)
        # logits = self.state2names(src_ast_encoding)
        # p = torch.log_softmax(logits, dim=-1)
        
        # idx = src_ast_encoding.unpacked_variable_node_ids
        # (batch_size, max_variable_node_num, tgt_vocab_size)
        # batched_p_names.unsqueeze(-1).expand_as(src_ast_encoding.batch_size, -1, -1).scatter(idx, dim=1)
        
        return batched_p_names
