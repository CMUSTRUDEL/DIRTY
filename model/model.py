from typing import List, Dict, Tuple

import torch
import torch.nn as nn

from utils import nn_util, util
from utils.ast import AbstractSyntaxTree
from model.decoder import Decoder, SimpleDecoder
from model.encoder import Encoder, GraphASTEncoder
from utils.graph import PackedGraph
from utils.dataset import Batcher
from utils.vocab import SAME_VARIABLE_TOKEN


class RenamingModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(RenamingModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.config: Dict = None

    @property
    def vocab(self):
        return self.encoder.vocab

    def batcher(self):
        if not hasattr(self, '_batcher'):
            _batcher = Batcher(self.config)
            setattr(self, '_batcher', _batcher)

        return self._batcher

    @property
    def device(self):
        return self.encoder.device

    @classmethod
    def default_params(cls):
        return {
            'train': {
                'unchanged_variable_weight': 1.0
            }
        }

    @classmethod
    def build(cls, config):
        params = util.update(cls.default_params(), config)
        encoder = GraphASTEncoder.build(config['encoder'])
        decoder = SimpleDecoder.build(config['decoder'])

        model = cls(encoder, decoder)
        params = util.update(params, {'encoder': encoder.config,
                                      'decoder': decoder.config})
        model.config = params

        return model

    def forward(self, source_asts: Dict[str, torch.Tensor], prediction_target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
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

        # (batch_var_node_num, tgt_vocab_size)
        packed_var_name_log_probs = self.decoder(context_encoding)

        # (batch_var_node_num)
        packed_tgt_var_node_name_log_probs = torch.gather(packed_var_name_log_probs,
                                                          dim=-1,
                                                          index=prediction_target['variable_tgt_name_id'].unsqueeze(-1)).squeeze(-1)

        # info = dict(context_encoding=context_encoding)
        info = dict()
        with torch.no_grad():
            # compute ppl over renamed variables
            renamed_var_mask = prediction_target['var_with_new_name_mask']
            unchanged_var_mask = prediction_target['auxiliary_var_mask']
            renamed_var_avg_ll = (packed_tgt_var_node_name_log_probs * renamed_var_mask).sum(-1) / renamed_var_mask.sum()
            unchanged_var_avg_ll = (packed_tgt_var_node_name_log_probs * unchanged_var_mask).sum(-1) / unchanged_var_mask.sum()
            info['rename_ppl'] = torch.exp(-renamed_var_avg_ll).item()
            info['unchange_ppl'] = torch.exp(-unchanged_var_avg_ll).item()

        packed_tgt_var_node_name_log_probs = packed_tgt_var_node_name_log_probs * prediction_target['variable_tgt_name_weight']
        # (batch_size, max_variable_node_num)
        tgt_name_log_probs = packed_tgt_var_node_name_log_probs[source_asts['variable_master_node_restoration_indices']]
        tgt_name_log_probs = tgt_name_log_probs * source_asts['variable_master_node_restoration_indices_mask']

        # (batch_size)
        ast_log_probs = tgt_name_log_probs.sum(dim=-1) / source_asts['variable_master_node_restoration_indices_mask'].sum(-1)

        return ast_log_probs, info

    def decode(self, source_asts: List[AbstractSyntaxTree]):
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

    def save(self, model_path, **kwargs):
        params = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'kwargs': kwargs
        }

        torch.save(params, model_path)

    @classmethod
    def load(cls, model_path, use_cuda=False, new_config=None) -> 'RenamingModel':
        device = torch.device("cuda:0" if use_cuda else "cpu")
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        config = params['config']

        if new_config:
            config = util.update(config, new_config)

        kwargs = params['kwargs'] if params['kwargs'] is not None else dict()

        model = cls.build(config, **kwargs)
        model.load_state_dict(params['state_dict'])
        model = model.to(device)
        model.eval()

        return model
