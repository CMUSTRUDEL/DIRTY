import math
from typing import Dict

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from dirty.utils import util
from dirty.utils.vocab import PAD_ID, Vocab

from .encoder import Encoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(
            nhid, nhead, 4 * nhid, dropout, activation="gelu"
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.nhid = nhid

    def forward(self, src, src_padding):
        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_padding)
        return output


class XfmrSequentialEncoder(Encoder):
    def __init__(self, config):
        super().__init__()

        self.vocab = vocab = Vocab.load(config["vocab_file"])

        self.src_word_embed = nn.Embedding(
            len(vocab.source_tokens), config["source_embedding_size"]
        )

        dropout = config["dropout"]
        self.encoder = TransformerModel(
            self.src_word_embed.embedding_dim,
            config["num_heads"],
            config["hidden_size"],
            config["num_layers"],
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.config = config

    @property
    def device(self):
        return self.src_word_embed.weight.device

    @classmethod
    def default_params(cls):
        return {
            "vocab_file": None,
            "source_embedding_size": 256,
            "hidden_size": 256,
            "num_layers": 2,
            "num_heads": 1,
        }

    @classmethod
    def build(cls, config):
        params = util.update(XfmrSequentialEncoder.default_params(), config)

        return cls(params)

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Returns the contextualized encoding for code tokens and variables

        :param tensor_dict: [description]
        :type tensor_dict: Dict[str, Union[torch.Tensor, int]]
        :return: variable_encoding: (batch_size, variable_num, hidden)
                 code_token_encoding: (batch_size, src_len, hidden)
                 code_token_mask: (batch_size, src_len), 1 for valid tokens, 0 for pad
        :rtype: Dict[str, torch.Tensor]
        """
        (
            code_token_encoding,
            code_token_mask,
        ) = self.encode_sequence(tensor_dict["src_code_tokens"])

        # (batch_size, max_variable_mention_num)
        variable_mention_mask = tensor_dict["variable_mention_mask"]
        variable_mention_to_variable_id = tensor_dict["variable_mention_to_variable_id"]

        # (batch_size, max_variable_num)
        variable_encoding_mask = tensor_dict["variable_encoding_mask"]
        variable_mention_num = tensor_dict["variable_mention_num"]

        # (batch_size, max_variable_mention_num, encoding_size)
        max_time_step = variable_mention_to_variable_id.size(1)  # noqa: F841
        variable_num = variable_mention_num.size(1)
        encoding_size = code_token_encoding.size(-1)

        variable_mention_encoding = (
            code_token_encoding * variable_mention_mask.unsqueeze(-1)
        )
        variable_encoding = torch.zeros(
            tensor_dict["src_code_tokens"].size(0),
            variable_num,
            encoding_size,
            device=self.device,
        )
        variable_encoding.scatter_add_(
            1,
            variable_mention_to_variable_id.unsqueeze(-1).expand(-1, -1, encoding_size),
            variable_mention_encoding,
        ) * variable_encoding_mask.unsqueeze(-1)
        variable_encoding = variable_encoding / (
            variable_mention_num + (1.0 - variable_encoding_mask) * 1e-8
        ).unsqueeze(-1)

        context_encoding = dict(
            variable_encoding=variable_encoding,
            code_token_encoding=code_token_encoding,
            code_token_mask=code_token_mask,
        )

        # context_encoding.update(tensor_dict)

        return context_encoding

    def encode_sequence(self, code_sequence):

        # (batch_size, max_code_length, embed_size)
        code_token_embedding = self.src_word_embed(code_sequence)

        # (batch_size, max_code_length)
        code_token_mask = torch.ne(code_sequence, PAD_ID)

        hidden = self.encoder(code_token_embedding.transpose(0, 1), ~code_token_mask)
        hidden = hidden.transpose(0, 1)

        # apply dropout to the last layer
        # (batch_size, seq_len, hidden_size * 2)
        hidden = self.dropout(hidden)

        return hidden, code_token_mask

    def get_attention_memory(self, context_encoding, att_target="terminal_nodes"):
        assert att_target == "terminal_nodes"

        memory = context_encoding["code_token_encoding"]
        mask = context_encoding["code_token_mask"]

        return memory, mask
