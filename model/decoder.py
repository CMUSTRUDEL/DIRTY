import torch
import torch.nn as nn

from utils import nn_util


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()


class SimpleDecoder(Decoder):
    def __init__(self, ast_node_encoding_size: int, tgt_name_vocab_size: int):
        super(SimpleDecoder, self).__init__()

        self.state2names = nn.Linear(ast_node_encoding_size, tgt_name_vocab_size, bias=True)

    def forward(self, src_ast_encoding):
        """
        Given a batch of encoded ASTs, compute the log-likelihood of generating all possible renamings
        """
        # (all_var_node_num, tgt_vocab_size)
        logits = self.state2names(src_ast_encoding['prediction_node_encoding'])
        batched_p_names = torch.log_softmax(logits, dim=-1)
        # logits = self.state2names(src_ast_encoding)
        # p = torch.log_softmax(logits, dim=-1)
        
        # idx = src_ast_encoding.unpacked_variable_node_ids
        # (batch_size, max_variable_node_num, tgt_vocab_size)
        # batched_p_names.unsqueeze(-1).expand_as(src_ast_encoding.batch_size, -1, -1).scatter(idx, dim=1)
        
        return batched_p_names
