import sentencepiece as spm

import torch
import torch.nn as nn


class SubTokenEmbedder(nn.Module):
    def __init__(self, bpe_model_path, embedding_size):
        super(SubTokenEmbedder, self).__init__()

        self.bpe_model = spm.SentencePieceProcessor()
        self.bpe_model.load(bpe_model_path)
        self.size = len(self.bpe_model)
        self.pad_id = self.bpe_model.pad_id()
        self.embeddings = nn.Embedding(self.size, embedding_size, padding_idx=self.pad_id)

    @property
    def device(self):
        return self.embeddings.weight.device

    def forward(self, sub_tokens_list):
        max_subword_num = max(len(x) for x in sub_tokens_list)
        idx_tensor = torch.zeros(len(sub_tokens_list), max_subword_num, dtype=torch.long)
        idx_tensor.fill_(self.pad_id)
        for i, token_list in enumerate(sub_tokens_list):
            for j, token in enumerate(token_list):
                idx_tensor[i, j] = self.bpe_model.piece_to_id(token)

        idx_tensor = idx_tensor.to(self.device)
        idx_mask = torch.ne(idx_tensor, self.pad_id).float()
        embedding = self.embeddings(idx_tensor) * idx_mask.unsqueeze(-1)
        embedding = embedding.sum(dim=1) / idx_mask.sum(-1).unsqueeze(-1)
        return embedding
