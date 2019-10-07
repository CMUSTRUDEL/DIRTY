from typing import Dict

import torch
from torch import nn as nn

from model.graph_encoder import GraphASTEncoder
from model.sequential_encoder import SequentialEncoder
from utils import util


class HybridEncoder(nn.Module):
    def __init__(self, config):
        super(HybridEncoder, self).__init__()

        self.graph_encoder = GraphASTEncoder.build(config['graph_encoder'])
        self.seq_encoder = SequentialEncoder.build(config['seq_encoder'])

        self.hybrid_method = config['hybrid_method']
        if self.hybrid_method == 'linear_proj':
            self.projection = nn.Linear(
                config['seq_encoder']['source_encoding_size']
                + config['graph_encoder']['gnn']['hidden_size'],
                config['source_encoding_size'], bias=False
            )
        else:
            assert self.hybrid_method == 'concat'

        self.config = config

    @property
    def device(self):
        return self.seq_encoder.device

    @classmethod
    def default_params(cls):
        return {
            "graph_encoder": GraphASTEncoder.default_params(),
            "seq_encoder": SequentialEncoder.default_params(),
            "hybrid_method": "linear_proj"
        }

    @classmethod
    def build(cls, config):
        params = util.update(cls.default_params(), config)

        return cls(params)

    def forward(self, tensor_dict: Dict[str, torch.Tensor]):
        graph_encoding = self.graph_encoder(tensor_dict['graph_encoder_input'])
        seq_encoding = self.seq_encoder(tensor_dict['seq_encoder_input'])

        graph_var_encoding = graph_encoding['variable_encoding']
        seq_var_encoding = seq_encoding['variable_encoding']

        if self.hybrid_method == 'linear_proj':
            variable_encoding = self.projection(
                torch.cat([graph_var_encoding, seq_var_encoding], dim=-1))
        else:
            variable_encoding = \
                torch.cat([graph_var_encoding, seq_var_encoding], dim=-1)

        context_encoding = dict(
            batch_size=tensor_dict['batch_size'],
            variable_encoding=variable_encoding,
            graph_encoding_result=graph_encoding,
            seq_encoding_result=seq_encoding
        )

        return context_encoding

    def get_decoder_init_state(self, context_encoding, config=None):
        gnn_dec_init_state, gnn_dec_init_cell = \
            self.graph_encoder.get_decoder_init_state(
                context_encoding['graph_encoding_result']
            )
        seq_dec_init_state, seq_dec_init_cell = \
            self.seq_encoder.get_decoder_init_state(
                context_encoding['seq_encoding_result']
            )

        dec_init_state = (gnn_dec_init_state + seq_dec_init_state) / 2
        dec_init_cell = (gnn_dec_init_cell + seq_dec_init_cell) / 2

        return dec_init_state, dec_init_cell

    def get_attention_memory(self,
                             context_encoding,
                             att_target='terminal_nodes'):
        assert att_target == 'terminal_nodes'

        seq_memory, seq_mask = \
            self.seq_encoder.get_attention_memory(
                context_encoding['seq_encoding_result'],
                att_target='terminal_nodes'
            )
        gnn_memory, gnn_mask = \
            self.graph_encoder.get_attention_memory(
                context_encoding['graph_encoding_result'],
                att_target='terminal_nodes'
            )

        if gnn_memory.size(-1) < seq_memory.size(-1):
            # pad values
            gnn_memory = torch.cat(
                [
                    gnn_memory.new_zeros(
                        gnn_memory.size(0),
                        gnn_memory.size(1),
                        seq_memory.size(-1) - gnn_memory.size(-1)
                    ),
                    gnn_memory
                ],
                dim=-1)

        memory = torch.cat([gnn_memory, seq_memory], dim=1)
        mask = torch.cat([gnn_mask, seq_mask], dim=1)

        return memory, mask
