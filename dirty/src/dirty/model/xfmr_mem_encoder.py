import math
from typing import Dict, Union

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
            nhid, nhead, nhid, dropout, activation="gelu"
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.nhid = nhid

    def forward(self, src, src_padding):
        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_padding)
        # set state and cell to the average of output
        return output


class XfmrMemEncoder(Encoder):
    def __init__(self, config):
        super().__init__()

        vocab = Vocab.load(config["vocab_file"])
        reg_pos_size = len(vocab.regs)
        self.src_word_embed = nn.Embedding(
            1030 + reg_pos_size, config["source_embedding_size"]
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
            "source_embedding_size": 256,
            "hidden_size": 256,
            "num_layers": 2,
            "num_heads": 1,
        }

    @classmethod
    def build(cls, config):
        params = util.update(XfmrMemEncoder.default_params(), config)

        return cls(params)

    def forward(
        self, tensor_dict: Dict[str, Union[torch.Tensor, int]]
    ) -> Dict[str, torch.Tensor]:
        """Returns the contextualized encoding for code tokens and variables

        :param tensor_dict: [description]
        :type tensor_dict: Dict[str, Union[torch.Tensor, int]]
        :return: variable_encoding: (mem_batch_size, hidden)
        :rtype: Dict[str, torch.Tensor]
        """
        mem_encoding, mem_mask = self.encode_sequence(
            tensor_dict["target_type_src_mems"][tensor_dict["target_mask"]]
        )

        # TODO: ignore the padding when averaging
        mem_encoding = mem_encoding.mean(dim=1)
        context_encoding = dict(
            variable_encoding=mem_encoding,
        )

        # context_encoding.update(tensor_dict)

        return context_encoding

    def encode_sequence(self, code_sequence):

        # (batch_size, max_code_length, embed_size)
        code_token_embedding = self.src_word_embed(code_sequence)

        # (batch_size, max_code_length)
        code_token_mask = torch.ne(code_sequence, PAD_ID)

        sorted_encodings = self.encoder(
            code_token_embedding.transpose(0, 1), ~code_token_mask
        )
        sorted_encodings = sorted_encodings.transpose(0, 1)

        # apply dropout to the last layer
        # (batch_size, seq_len, hidden_size * 2)
        sorted_encodings = self.dropout(sorted_encodings)

        return sorted_encodings, code_token_mask
