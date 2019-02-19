import torch
import torch.nn as nn


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

        return self.state2names(src_ast_encoding)
