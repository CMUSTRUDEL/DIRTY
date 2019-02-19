from typing import List, Dict

import torch
import torch.nn as nn

from ast import AbstractSyntaxTree
from model.decoder import Decoder
from model.encoder import Encoder


class RenamingModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(RenamingModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source_asts: List[AbstractSyntaxTree], name_map: List[Dict[str, str]]) -> torch.Tensor:
        """
        Given a batch of decompiled abstract syntax trees, and the gold-standard renaming of variable nodes,
        compute the log-likelihood of the gold-standard renaming for training

        Arg:
            source_asts: a list of ASTs
            name_map: mapping of decompiled variable names to its renamed values

        Return:
            a tensor of size (batch_size) denoting the log-likelihood of renamings
        """

        # src_ast_encoding: (batch_size, max_ast_node_num, node_encoding_size)
        # src_ast_mask: (batch_size, max_ast_node_num)
        src_ast_encoding, src_ast_mask = self.encoder(source_asts)

        # tgt_name_ids: (batch_size, max_ast_node_num)
        # tgt_name_mask: (batch_size, max_ast_node_num)
        tgt_name_ids, tgt_name_mask = AbstractSyntaxTree.to_batched_prediction_target(source_asts, name_map)

        # (batch_size, max_ast_node_num, tgt_vocab_size)
        var_name_log_probs = self.decoder(src_ast_encoding)

        # (batch_size, max_ast_node_num)
        tgt_name_log_probs = torch.gather(var_name_log_probs, dim=1, index=tgt_name_ids.unsqueeze(-1)).squeeze(-1)
        # mask out unused probability entries
        tgt_name_log_probs = tgt_name_log_probs * tgt_name_mask

        # (batch_size)
        ast_log_probs = tgt_name_log_probs.sum(dim=-1)

        return ast_log_probs
