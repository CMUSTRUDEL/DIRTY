import sys
from typing import Dict, List
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.encoder import *
from utils import nn_util, util
from utils.dataset import Example
from utils.vocab import PAD_ID, Vocab
import torch
import torch.nn as nn


class SequentialEncoder(Encoder):
    def __init__(self, config):
        super().__init__()

        self.vocab = vocab = Vocab.load(config['vocab_file'])

        self.src_word_embed = nn.Embedding(len(vocab.source_tokens), config['source_embedding_size'])

        dropout = config['dropout']
        self.lstm_encoder = nn.LSTM(input_size=self.src_word_embed.embedding_dim,
                                    hidden_size=config['source_encoding_size'] // 2, num_layers=config['num_layers'],
                                    batch_first=True, bidirectional=True, dropout=dropout)

        self.decoder_cell_init = nn.Linear(config['source_encoding_size'], config['decoder_hidden_size'])

        self.dropout = nn.Dropout(dropout)
        self.config = config

    @property
    def device(self):
        return self.src_word_embed.weight.device

    @classmethod
    def default_params(cls):
        return {
            'source_encoding_size': 256,
            'decoder_hidden_size': 128,
            'source_embedding_size': 128,
            'vocab_file': None,
            'num_layers': 1
        }

    @classmethod
    def build(cls, config):
        params = util.update(SequentialEncoder.default_params(), config)

        return cls(params)

    def forward(self, tensor_dict: Dict[str, torch.Tensor]):
        code_token_encoding, code_token_mask, (last_states, last_cells) = self.encode_sequence(tensor_dict['src_code_tokens'])

        # (batch_size, max_variable_mention_num)
        # variable_mention_positions = tensor_dict['variable_position']
        variable_mention_mask = tensor_dict['variable_mention_mask']
        variable_mention_to_variable_id = tensor_dict['variable_mention_to_variable_id']

        # (batch_size, max_variable_num)
        variable_encoding_mask = tensor_dict['variable_encoding_mask']
        variable_mention_num = tensor_dict['variable_mention_num']

        # # (batch_size, max_variable_mention_num, encoding_size)
        # variable_mention_encoding = torch.gather(code_token_encoding, 1, variable_mention_positions.unsqueeze(-1).expand(-1, -1, code_token_encoding.size(-1))) * variable_mention_positions_mask
        max_time_step = variable_mention_to_variable_id.size(1)
        variable_num = variable_mention_num.size(1)
        encoding_size = code_token_encoding.size(-1)

        variable_mention_encoding = code_token_encoding * variable_mention_mask.unsqueeze(-1)
        variable_encoding = torch.zeros(tensor_dict['batch_size'], variable_num, encoding_size, device=self.device)
        variable_encoding.scatter_add_(1,
                                       variable_mention_to_variable_id.unsqueeze(-1).expand(-1, -1, encoding_size),
                                       variable_mention_encoding) * variable_encoding_mask.unsqueeze(-1)
        variable_encoding = variable_encoding / (variable_mention_num + (1. - variable_encoding_mask) * nn_util.SMALL_NUMBER).unsqueeze(-1)

        context_encoding = dict(
            variable_encoding=variable_encoding,
            code_token_encoding=code_token_encoding,
            code_token_mask=code_token_mask,
            last_states=last_states,
            last_cells=last_cells
        )

        context_encoding.update(tensor_dict)

        return context_encoding

    def encode_sequence(self, code_sequence):
        # (batch_size, max_code_length)
        # code_sequence = tensor_dict['src_code_tokens']

        # (batch_size, max_code_length, embed_size)
        code_token_embedding = self.src_word_embed(code_sequence)

        # (batch_size, max_code_length)
        code_token_mask = torch.ne(code_sequence, PAD_ID).float()
        # (batch_size)
        code_sequence_length = code_token_mask.sum(dim=-1).long()

        sorted_seqs, sorted_seq_lens, restoration_indices, sorting_indices = nn_util.sort_batch_by_length(code_token_embedding,
                                                                                                          code_sequence_length)

        packed_question_embedding = pack_padded_sequence(sorted_seqs, sorted_seq_lens.data.tolist(), batch_first=True)

        sorted_encodings, (last_states, last_cells) = self.lstm_encoder(packed_question_embedding)
        sorted_encodings, _ = pad_packed_sequence(sorted_encodings, batch_first=True)

        # apply dropout to the last layer
        # (batch_size, seq_len, hidden_size * 2)
        sorted_encodings = self.dropout(sorted_encodings)

        # (batch_size, question_len, hidden_size * 2)
        restored_encodings = sorted_encodings.index_select(dim=0, index=restoration_indices)

        # (num_layers, direction_num, batch_size, hidden_size)
        last_states = last_states.view(self.lstm_encoder.num_layers, 2, -1, self.lstm_encoder.hidden_size)
        last_states = last_states.index_select(dim=2, index=restoration_indices)
        last_cells = last_cells.view(self.lstm_encoder.num_layers, 2, -1, self.lstm_encoder.hidden_size)
        last_cells = last_cells.index_select(dim=2, index=restoration_indices)

        return restored_encodings, code_token_mask, (last_states, last_cells)

    @classmethod
    def to_tensor_dict(cls, examples: List[Example]) -> Dict[str, torch.Tensor]:
        max_time_step = max(e.source_seq_length for e in examples)
        input = np.zeros((len(examples), max_time_step), dtype=np.int64)

        variable_mention_to_variable_id = torch.zeros(len(examples), max_time_step, dtype=torch.long)
        variable_mention_mask = torch.zeros(len(examples), max_time_step)
        variable_mention_num = torch.zeros(len(examples), max(len(e.ast.variables) for e in examples))
        variable_encoding_mask = torch.zeros(variable_mention_num.size())

        for e_id, example in enumerate(examples):
            sub_tokens = example.sub_tokens
            input[e_id, :len(sub_tokens)] = example.sub_token_ids

            variable_position_map = dict()
            var_name_to_id = {name: i for i, name in enumerate(example.ast.variables)}
            for i, sub_token in enumerate(sub_tokens):
                if sub_token.startswith('@@') and sub_token.endswith('@@'):
                    old_var_name = sub_token[2: -2]
                    if old_var_name in var_name_to_id:  # sometimes there are strings like `@@@@`
                        var_id = var_name_to_id[old_var_name]

                        variable_mention_to_variable_id[e_id, i] = var_id
                        variable_mention_mask[e_id, i] = 1.
                        variable_position_map.setdefault(old_var_name, []).append(i)

            for var_id, var_name in enumerate(example.ast.variables):
                try:
                    var_pos = variable_position_map[var_name]
                    variable_mention_num[e_id, var_id] = len(var_pos)
                except KeyError:
                    variable_mention_num[e_id, var_id] = 1
                    print(example.binary_file, f'variable [{var_name}] not found', file=sys.stderr)

            variable_encoding_mask[e_id, :len(example.ast.variables)] = 1.

        return dict(src_code_tokens=torch.from_numpy(input),
                    variable_mention_to_variable_id=variable_mention_to_variable_id,
                    variable_mention_mask=variable_mention_mask,
                    variable_mention_num=variable_mention_num,
                    variable_encoding_mask=variable_encoding_mask,
                    batch_size=len(examples))

    def get_decoder_init_state(self, context_encoder, config=None):
        fwd_last_layer_cell = context_encoder['last_cells'][-1, 0]
        bak_last_layer_cell = context_encoder['last_cells'][-1, 1]

        dec_init_cell = self.decoder_cell_init(torch.cat([fwd_last_layer_cell, bak_last_layer_cell], dim=-1))
        dec_init_state = torch.tanh(dec_init_cell)

        return dec_init_state, dec_init_cell

    def get_attention_memory(self, context_encoding, att_target='terminal_nodes'):
        assert att_target == 'terminal_nodes'

        memory = context_encoding['code_token_encoding']
        mask = context_encoding['code_token_mask']

        return memory, mask
