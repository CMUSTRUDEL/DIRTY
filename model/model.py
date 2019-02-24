from typing import List, Dict

import torch
import torch.nn as nn

from utils.ast import AbstractSyntaxTree
from model.decoder import Decoder
from model.encoder import Encoder


class RenamingModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(RenamingModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    @property
    def vocab(self):
        return self.encoder.vocab

    @property
    def device(self):
        return self.encoder.device

    def forward(self, source_asts: List[AbstractSyntaxTree], variable_name_maps: List[Dict[int, str]]) -> torch.Tensor:
        """
        Given a batch of decompiled abstract syntax trees, and the gold-standard renaming of variable nodes,
        compute the log-likelihood of the gold-standard renaming for training

        Arg:
            source_asts: a list of ASTs
            variable_name_maps: mapping of decompiled variable names to its renamed values

        Return:
            a tensor of size (batch_size) denoting the log-likelihood of renamings
        """

        # src_ast_encoding: (batch_size, max_ast_node_num, node_encoding_size)
        # src_ast_mask: (batch_size, max_ast_node_num)
        context_encoding = self.encoder(source_asts)
        src_ast_encoding = context_encoding['batch_tree_node_encoding']
        src_ast_mask = context_encoding['batch_tree_node_masks']

        prediction_target = AbstractSyntaxTree.to_batched_prediction_target(source_asts, variable_name_maps, vocab=self.vocab.target)
        # (batch_size, max_variable_node_num)
        tgt_var_node_ids = prediction_target['tgt_variable_node_ids'].to(self.device)
        # (batch_size, max_variable_node_num)
        tgt_var_node_mask = prediction_target['tgt_variable_node_mask'].to(self.device)
        # (batch_size, max_variable_node_num)
        tgt_name_ids = prediction_target['tgt_name_ids'].to(self.device)

        # (batch_size, max_ast_node_num, tgt_vocab_size)
        var_name_log_probs = self.decoder(src_ast_encoding)

        # (batch_size, max_variable_node_num, tgt_vocab_size)
        tgt_var_node_name_log_probs = torch.gather(var_name_log_probs, dim=1, index=tgt_var_node_ids.unsqueeze(-1).expand(-1, -1, var_name_log_probs.size(-1)))
        # (batch_size, max_variable_node_num)
        tgt_name_log_probs = torch.gather(tgt_var_node_name_log_probs, dim=-1, index=tgt_name_ids.unsqueeze(-1)).squeeze(-1)
        # mask out unused probability entries
        tgt_name_log_probs = tgt_name_log_probs * tgt_var_node_mask

        # (batch_size)
        ast_log_probs = tgt_name_log_probs.sum(dim=-1)

        return ast_log_probs
