from typing import List, Dict

import sentencepiece as spm

import torch
import torch.nn as nn

from utils.nn_util import SMALL_NUMBER


class SubTokenEmbedder(nn.Module):
    def __init__(self, bpe_model_path, embedding_size):
        super(SubTokenEmbedder, self).__init__()

        self.bpe_model = spm.SentencePieceProcessor()
        self.bpe_model.load(bpe_model_path)
        self.size = len(self.bpe_model)
        self.pad_id = self.bpe_model.pad_id()
        self.embeddings = nn.Embedding(self.size,
                                       embedding_size,
                                       padding_idx=self.pad_id)

    @property
    def device(self):
        return self.embeddings.weight.device

    @classmethod
    def to_input_tensor(cls,
                        sub_tokens_list: List[List[str]],
                        bpe_model: spm.SentencePieceProcessor,
                        pad_id=0) -> torch.Tensor:
        max_subword_num = max(len(x) for x in sub_tokens_list)
        idx_tensor = torch.zeros(len(sub_tokens_list),
                                 max_subword_num,
                                 dtype=torch.long)
        idx_tensor.fill_(pad_id)

        for i, token_list in enumerate(sub_tokens_list):
            for j, token in enumerate(token_list):
                idx_tensor[i, j] = bpe_model.piece_to_id(token)

        return idx_tensor

    def get_embedding(self, sub_tokens_list: List[List[str]]) -> torch.Tensor:
        idx_tensor = \
            self.to_input_tensor(sub_tokens_list, self.bpe_model, self.pad_id)
        embedding = self.forward(idx_tensor)

        return embedding

    def forward(self, sub_tokens_indices: torch.Tensor) -> torch.Tensor:
        # sub_tokens_indices: (batch_size, max_sub_token_num)
        sub_tokens_mask = torch.ne(sub_tokens_indices, self.pad_id).float()
        embedding = \
            self.embeddings(sub_tokens_indices) * sub_tokens_mask.unsqueeze(-1)
        embedding = \
            embedding.sum(dim=1) / sub_tokens_mask.sum(-1).unsqueeze(-1)
        return embedding


class NodeTypeEmbedder(nn.Module):
    def __init__(self, type_num, embedding_size, pad_id=0):
        super(NodeTypeEmbedder, self).__init__()

        self.size = type_num
        self.pad_id = pad_id
        self.embeddings = \
            nn.Embedding(type_num, embedding_size, padding_idx=self.pad_id)

    @property
    def device(self):
        return self.embeddings.weight.device

    @classmethod
    def to_input_tensor(cls, token_types_list: List[List[str]],
                        type2id: Dict[str, int],
                        pad_id=0) -> torch.Tensor:
        max_subword_num = max(len(x) for x in token_types_list)
        idx_tensor = torch.zeros(len(token_types_list),
                                 max_subword_num,
                                 dtype=torch.long)
        idx_tensor.fill_(pad_id)

        for i, token_list in enumerate(token_types_list):
            idx_tensor[i, :len(token_list)] = \
                torch.tensor([type2id[t] for t in token_list])
        return idx_tensor

    def forward(self, type_tokens_indices: torch.Tensor) -> torch.Tensor:
        # type_tokens_indices: (batch_size, max_sub_token_num)
        sub_tokens_mask = torch.ne(type_tokens_indices, self.pad_id).float()
        embedding = \
            self.embeddings(type_tokens_indices) \
            * sub_tokens_mask.unsqueeze(-1)
        embedding = \
            embedding.sum(dim=1) \
            / (sub_tokens_mask.sum(-1) + SMALL_NUMBER).unsqueeze(-1)

        return embedding
