from typing import List, Dict

import torch
import torch.nn as nn

from utils import nn_util
from utils.ast import AbstractSyntaxTree
from model.decoder import Decoder
from model.encoder import Encoder
from utils.dataset import BatchUtil
from utils.vocab import SAME_VARIABLE_TOKEN


class RenamingModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, config: Dict):
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

        prediction_target = BatchUtil.to_batched_prediction_target(source_asts, variable_name_maps,
                                                                   context_encoding,
                                                                   vocab=self.vocab.target)
        prediction_target = nn_util.to(prediction_target, self.device)

        # (batch_var_node_num, tgt_vocab_size)
        packed_var_name_log_probs = self.decoder(context_encoding)
        # (batch_var_node_num)
        packed_tgt_var_node_name_log_probs = torch.gather(packed_var_name_log_probs,
                                                          dim=-1,
                                                          index=prediction_target['packed_variable_tgt_name_id'].unsqueeze(-1)).squeeze(-1)
        # (batch_size, max_variable_node_num)
        tgt_name_log_probs = packed_tgt_var_node_name_log_probs[context_encoding['prediction_node_restoration_indices']]
        tgt_name_log_probs = tgt_name_log_probs * context_encoding['prediction_node_restoration_indices_mask']

        # (batch_size)
        ast_log_probs = tgt_name_log_probs.sum(dim=-1)

        return ast_log_probs

    def decode(self, source_asts: List[AbstractSyntaxTree]):
        """
        Given a batch of ASTs, predict their new variable names
        """

        context_encoding = self.encoder(source_asts)
        # (prediction_size, tgt_vocab_size)
        packed_var_name_log_probs = self.decoder(context_encoding)
        best_var_name_log_probs, best_var_name_ids = torch.max(packed_var_name_log_probs, dim=-1)

        variable_rename_results = []
        pred_node_ptr = 0
        for ast_id, ast in enumerate(source_asts):
            variable_rename_result = dict()
            for var_name, var_nodes in ast.variables.items():
                var_name_prob = best_var_name_log_probs[pred_node_ptr].item()
                token_id = best_var_name_ids[pred_node_ptr].item()
                new_var_name = self.vocab.target.id2word[token_id]

                if new_var_name == SAME_VARIABLE_TOKEN:
                    new_var_name = var_name

                variable_rename_result[var_name] = {'new_name': new_var_name,
                                                    'prob': var_name_prob}

                pred_node_ptr += 1

            variable_rename_results.append(variable_rename_result)

        return variable_rename_results

    @classmethod
    def build(cls, config):
        encoder = Encoder.build(config['encoder'])
        decoder = Decoder.build(config['decoder'])

        model = cls(encoder, decoder, config).eval()
        return model

    def save(self, model_path, **kwargs):
        params = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'kwargs': kwargs
        }

        torch.save(params, model_path)
