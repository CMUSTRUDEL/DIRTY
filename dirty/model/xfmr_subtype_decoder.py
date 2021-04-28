from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer, LayerNorm
from torch.nn.modules import activation
from utils import util
from utils.vocab import Vocab
from utils.dire_types import TypeLibCodec
from model.xfmr_decoder import XfmrDecoder


class XfmrSubtypeDecoder(XfmrDecoder):
    def __init__(self, config):
        super(XfmrDecoder, self).__init__()

        self.vocab = Vocab.load(config["vocab_file"])
        with open(config["typelib_file"]) as type_f:
            self.typelib = TypeLibCodec.decode(type_f.read())
            self.typelib = self.typelib.fix()
        self.target_embedding = nn.Embedding(
            len(self.vocab.subtypes), config["target_embedding_size"]
        )
        self.target_transform = nn.Linear(
            config["target_embedding_size"] + config["hidden_size"],
            config["hidden_size"],
        )
        # self.cached_decode_mask: Dict[int, torch.Tensor] = {}
        # self.size = torch.zeros(len(self.vocab.types), dtype=torch.long)

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
        self.output = nn.Linear(config["hidden_size"], len(self.vocab.subtypes))

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
        # (B, NUM_VAR) -> (B, NUM_VAR, H)
        tgt = self.target_embedding(target_dict["target_subtype_id"])
        # Shift 1 position to right
        tgt = torch.cat([torch.zeros_like(tgt[:, :1]), tgt[:, :-1]], dim=1)
        # Repeat variable encoding according to tgt_sizes
        tgt = torch.cat(
            [
                XfmrSubtypeDecoder.repeat_variable_encoding(
                    tgt, context_encoding, target_dict
                ),
                tgt,
            ],
            dim=-1,
        )
        tgt = self.target_transform(tgt)
        # mask out attention to subsequent inputs which include the ground truth for current step
        tgt_mask = XfmrDecoder.generate_square_subsequent_mask(tgt.shape[1], tgt.device)
        # TransformerModels have batch_first=False
        hidden = self.decoder(
            tgt.transpose(0, 1),
            memory=context_encoding["code_token_encoding"].transpose(0, 1),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=~target_dict["target_submask"],
            memory_key_padding_mask=~context_encoding["code_token_mask"],
        ).transpose(0, 1)
        logits = self.output(hidden)
        return logits

    @classmethod
    def repeat_variable_encoding(cls, tgt, context_encoding, target_dict):
        repeated_variable_encoding = torch.zeros_like(tgt)
        # TODO: is there a way to vectorize the following?
        tgt_sizes = target_dict["target_type_sizes"]
        for b in range(tgt.shape[0]):
            idx = 0
            sizes = tgt_sizes[b][(tgt_sizes[b] > 0)].tolist()
            for vid in range(len(sizes)):
                repeated_variable_encoding[
                    b, idx : idx + sizes[vid]
                ] = context_encoding["variable_encoding"][b, vid]
                idx += sizes[vid]
        return repeated_variable_encoding

    def predict(
        self,
        context_encoding: Dict[str, torch.Tensor],
        target_dict: Dict[str, torch.Tensor],
        variable_type_logits: torch.Tensor,
    ):
        """Greedy decoding"""

        batch_size, _, _ = context_encoding["variable_encoding"].shape
        max_time_step = 64
        tgt = torch.zeros(batch_size, 1, self.config["target_embedding_size"]).to(
            context_encoding["variable_encoding"].device
        )
        tgt = self.target_transform(
            torch.cat([context_encoding["variable_encoding"][:, :1], tgt], dim=-1)
        )
        tgt_mask = XfmrDecoder.generate_square_subsequent_mask(
            max_time_step, tgt.device
        )
        preds_list = []
        # Keep track of which variable is being decoded for each example in batch
        num_vars = (target_dict["target_type_sizes"] > 0).sum(dim=1).tolist()
        current_vars = [0] * batch_size
        ans = [[] for _ in range(batch_size)]
        try:
            for idx in range(max_time_step):
                hidden = self.decoder(
                    tgt.transpose(0, 1),
                    memory=context_encoding["code_token_encoding"].transpose(0, 1),
                    tgt_mask=tgt_mask[: idx + 1, : idx + 1],
                    # tgt_key_padding_mask=~target_dict["target_mask"][:, : idx + 1],
                    memory_key_padding_mask=~context_encoding["code_token_mask"],
                ).transpose(0, 1)
                # Save logits for the current step
                logits = self.output(hidden[:, -1:])
                # Make prediction for this step
                preds_step = []
                for b in range(batch_size):
                    if current_vars[b] == num_vars[b]:
                        # Already finished for this example
                        preds_step.append(
                            torch.zeros(1, dtype=torch.long, device=logits.device)
                        )
                        continue
                    scores = logits[b, 0]
                    pred = scores.argmax(dim=0, keepdim=True)
                    preds_step.append(pred)
                    subtype = self.vocab.subtypes.id2word[pred.item()]
                    ans[b].append(subtype)
                    current_vars[b] += int(subtype == "<eot>")
                pred_step = torch.cat(preds_step)
                preds_list.append(pred_step)
                # Update tgt for next step with prediction at the current step
                if sum(current_vars) < sum(num_vars):
                    tgt_step = torch.cat(
                        [
                            context_encoding["variable_encoding"][
                                range(batch_size),
                                [
                                    min(cur_var, num_var - 1)
                                    for cur_var, num_var in zip(current_vars, num_vars)
                                ],
                            ].unsqueeze(1),
                            self.target_embedding(pred_step.unsqueeze(dim=1)),
                        ],
                        dim=-1,
                    )
                    tgt_step = self.target_transform(tgt_step)
                    tgt = torch.cat([tgt, tgt_step], dim=1)
                else:
                    break
        except Exception:
            import pdb

            pdb.set_trace()
        preds = torch.stack(preds_list).transpose(0, 1)
        return list(preds)

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
