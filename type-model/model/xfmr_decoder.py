from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer, LayerNorm
from torch.nn.modules import activation
from utils import util
from utils.vocab import Vocab


class XfmrDecoder(nn.Module):
    def __init__(self, config):
        super(XfmrDecoder, self).__init__()

        self.vocab = Vocab.load(config["vocab_file"])
        self.target_embedding = nn.Embedding(
            len(self.vocab.types), config["target_embedding_size"]
        )
        self.target_transform = nn.Linear(
            config["target_embedding_size"] + config["hidden_size"],
            config["hidden_size"],
        )

        # concat variable encoding and previous target token embedding as input
        decoder_layer = TransformerDecoderLayer(
            config["hidden_size"],
            1,
            config["hidden_size"],
            config["dropout"],
            activation="gelu",
        )
        decoder_norm = LayerNorm(config["hidden_size"])
        self.decoder = TransformerDecoder(
            decoder_layer, config["num_layers"], decoder_norm
        )
        self.output = nn.Linear(config["hidden_size"], len(self.vocab.types))

        self.config: Dict = config

    @classmethod
    def default_params(cls):
        return {
            "vocab_file": None,
            "target_embedding_size": 256,
            "hidden_size": 256,
            "num_layers": 2,
        }

    @classmethod
    def build(cls, config):
        params = util.update(cls.default_params(), config)
        model = cls(params)
        return model

    def forward(
        self,
        context_encoding: Dict[str, torch.Tensor],
        target_dict: Dict[str, torch.Tensor],
    ):
        context_encoding["variable_encoding"]
        # (B, NUM_VAR) -> (B, NUM_VAR, H)
        tgt = self.target_embedding(target_dict["target_type_id"])
        # Shift 1 position to right
        tgt = torch.cat([torch.zeros_like(tgt[:, :1]), tgt[:, :-1]], dim=1)
        tgt = torch.cat([context_encoding["variable_encoding"], tgt], dim=-1)
        tgt = self.target_transform(tgt)
        # mask out attention to subsequent inputs which include the ground truth for current step
        tgt_mask = XfmrDecoder.generate_square_subsequent_mask(tgt.shape[1], tgt.device)
        # TransformerModels have batch_first=False
        hidden = self.decoder(
            tgt.transpose(0, 1),
            memory=context_encoding["code_token_encoding"].transpose(0, 1),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=~target_dict["target_mask"],
            memory_key_padding_mask=~context_encoding["code_token_mask"],
        ).transpose(0, 1)
        logits = self.output(hidden)
        return logits

    def predict(
        self,
        context_encoding: Dict[str, torch.Tensor],
        target_dict: Dict[str, torch.Tensor],
        variable_type_logits: torch.Tensor,
    ):
        """Greedy decoding"""

        batch_size, max_time_step, _ = context_encoding["variable_encoding"].shape
        tgt = torch.zeros(batch_size, 1, self.config["target_embedding_size"]).to(
            context_encoding["variable_encoding"].device
        )
        tgt = self.target_transform(
            torch.cat([context_encoding["variable_encoding"][:, :1], tgt], dim=-1)
        )
        tgt_mask = XfmrDecoder.generate_square_subsequent_mask(max_time_step, tgt.device)
        logits_list = []
        for idx in range(max_time_step):
            hidden = self.decoder(
                tgt.transpose(0, 1),
                memory=context_encoding["code_token_encoding"].transpose(0, 1),
                tgt_mask=tgt_mask[: idx + 1, : idx + 1],
                tgt_key_padding_mask=~target_dict["target_mask"][:, : idx + 1],
                memory_key_padding_mask=~context_encoding["code_token_mask"],
            ).transpose(0, 1)
            # Save logits for the current step
            logits = self.output(hidden[:, -1:])
            logits_list.append(logits)
            # Update tgt for next step with prediction at the current step
            if idx < max_time_step - 1:
                tgt_step = self.target_embedding(logits.argmax(dim=2))
                tgt_step = torch.cat(
                    [context_encoding["variable_encoding"][:, idx + 1 : idx + 2], tgt_step],
                    dim=-1,
                )
                tgt_step = self.target_transform(tgt_step)
                tgt = torch.cat([tgt, tgt_step], dim=1)
        logits = torch.cat(logits_list, dim=1)
        return logits[target_dict["target_mask"]].argmax(dim=1)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
