from typing import Dict, List

import torch
from torch import nn as nn

from model.decoder import Decoder
from utils import util, nn_util
from utils.ast import AbstractSyntaxTree
from utils.vocab import Vocab, SAME_VARIABLE_TOKEN


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

    def forward(self, src_ast_encoding: Dict[str, torch.Tensor], prediction_target: Dict[str, torch.Tensor]):
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

        # (batch_var_node_num)
        packed_tgt_var_node_name_log_probs = torch.gather(batched_p_names,
                                                          dim=-1,
                                                          index=prediction_target['variable_tgt_name_id'].unsqueeze(-1)).squeeze(-1)

        # result = dict(context_encoding=context_encoding)
        result = dict()
        with torch.no_grad():
            # compute ppl over renamed variables
            renamed_var_mask = prediction_target['var_with_new_name_mask']
            unchanged_var_mask = prediction_target['auxiliary_var_mask']
            renamed_var_avg_ll = (packed_tgt_var_node_name_log_probs * renamed_var_mask).sum(
                -1) / renamed_var_mask.sum()
            unchanged_var_avg_ll = (packed_tgt_var_node_name_log_probs * unchanged_var_mask).sum(
                -1) / unchanged_var_mask.sum()
            result['rename_ppl'] = torch.exp(-renamed_var_avg_ll).item()
            result['unchange_ppl'] = torch.exp(-unchanged_var_avg_ll).item()

        packed_tgt_var_node_name_log_probs = packed_tgt_var_node_name_log_probs * prediction_target['variable_tgt_name_weight']

        # (batch_size, max_variable_node_num)
        tgt_name_log_probs = packed_tgt_var_node_name_log_probs[src_ast_encoding['variable_encoding_restoration_indices']]
        tgt_name_log_probs = tgt_name_log_probs * src_ast_encoding['variable_encoding_restoration_indices_mask']

        # (batch_size)
        ast_log_probs = tgt_name_log_probs.sum(dim=-1) / src_ast_encoding['variable_encoding_restoration_indices_mask'].sum(-1)

        result['batch_log_prob'] = ast_log_probs

        return result

    def predict(self, source_asts: List[AbstractSyntaxTree]):
        """
        Given a batch of ASTs, predict their new variable names
        """

        tensor_dict = self.batcher.to_tensor_dict(source_asts=source_asts)
        nn_util.to(tensor_dict, self.device)
        context_encoding = self.encoder(tensor_dict)
        # (prediction_size, tgt_vocab_size)
        packed_var_name_log_probs = self.decoder(context_encoding)
        best_var_name_log_probs, best_var_name_ids = torch.max(packed_var_name_log_probs, dim=-1)

        variable_rename_results = []
        pred_node_ptr = 0
        for ast_id, ast in enumerate(source_asts):
            variable_rename_result = dict()
            for var_name in ast.variables:
                var_name_prob = best_var_name_log_probs[pred_node_ptr].item()
                token_id = best_var_name_ids[pred_node_ptr].item()
                new_var_name = self.vocab.target.id2word[token_id]

                if new_var_name == SAME_VARIABLE_TOKEN:
                    new_var_name = var_name

                variable_rename_result[var_name] = {'new_name': new_var_name,
                                                    'prob': var_name_prob}

                pred_node_ptr += 1

            variable_rename_results.append(variable_rename_result)

        assert pred_node_ptr == packed_var_name_log_probs.size(0)

        return variable_rename_results
