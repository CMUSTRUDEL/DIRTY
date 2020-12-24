from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer, LayerNorm
from torch.nn.modules import activation
from utils import util
from utils.vocab import Vocab
from utils.dire_types import TypeLibCodec


class XfmrDecoder(nn.Module):
    def __init__(self, config):
        super(XfmrDecoder, self).__init__()

        self.vocab = Vocab.load(config["vocab_file"])
        with open(config["typelib_file"]) as type_f:
            self.typelib = TypeLibCodec.decode(type_f.read())
        vocab_size = len(self.vocab.names) if config.get("rename", False) else len(self.vocab.types)
        self.target_id_key = "target_name_id" if config.get("rename", False) else "target_type_id"
        self.target_embedding = nn.Embedding(
            vocab_size, config["target_embedding_size"]
        )
        self.target_transform = nn.Linear(
            config["target_embedding_size"] + config["hidden_size"],
            config["hidden_size"],
        )
        self.cached_decode_mask: Dict[int, torch.Tensor] = {}
        self.size = torch.zeros(vocab_size, dtype=torch.long)

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
        self.output = nn.Linear(config["hidden_size"], vocab_size)
        self.mem_mask = config["mem_mask"]
        if config.get("rename", False):
            self.mem_mask = "none"

        self.config: Dict = config

    @classmethod
    def default_params(cls):
        return {
            "vocab_file": None,
            "target_embedding_size": 256,
            "hidden_size": 256,
            "num_layers": 2,
            "mem_mask": "none",
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
        tgt = self.target_embedding(target_dict[self.target_id_key])
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

    def pred_with_mem(self, scores: torch.Tensor, mems: Tuple[Tuple[int], Tuple[int]]) -> Tuple[torch.Tensor, Tuple[Tuple[int], Tuple[int]]]:
        rest_a, rest_s = mems
        device = scores.device
        scores = scores.cpu()
        if not rest_a or not rest_s:
            # An incorrect prediction must have been made. Ignore the mask
            mask = torch.ones(scores.shape, dtype=torch.bool)
        else:
            mask = torch.zeros(scores.shape, dtype=torch.bool)
            ret = self.typelib.get_next_replacements(rest_a, rest_s)
            for typ_set, _, _ in ret:
                if not id(typ_set) in self.cached_decode_mask:
                    t_mask = torch.zeros(scores.shape, dtype=torch.bool)
                    for typ in typ_set:
                        if str(typ) in self.vocab.types:
                            typ_id = self.vocab.types[str(typ)]
                            t_mask[typ_id] = 1
                            self.size[typ_id] = typ.size
                    self.cached_decode_mask[id(typ_set)] = t_mask
                mask |= self.cached_decode_mask[id(typ_set)]
            # also incorrect prediction 
            if mask.sum() == 0:
                mask = torch.ones(scores.shape, dtype=torch.bool)
        scores[~mask] = float('-inf')
        pred = scores.argmax(dim=0, keepdim=True)
        if rest_a:
            start = rest_a[0]
            size = self.size[pred].item()
            rest_a = tuple(s for s in rest_a if s >= (size + start))
            rest_s = tuple(s for s in rest_s if s >= (size + start))
        return pred.to(device), (rest_a, rest_s)

    def predict(
        self,
        context_encoding: Dict[str, torch.Tensor],
        input_dict: Dict[str, torch.Tensor],
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
        preds_list = []
        if self.mem_mask == "hard":
            raise NotImplementedError
            # tgt_mems = target_dict["target_mems"]
        if self.mem_mask == "soft":
            mem_encoding = self.mem_encoder(input_dict)
            mem_logits = self.mem_decoder(mem_encoding, target_dict=None)
            mem_logits_list = []
            idx = 0
            for b in range(batch_size):
                nvar = input_dict["target_mask"][b].sum().item()
                mem_logits_list.append(mem_logits[idx: idx + nvar])
                idx += nvar
            assert idx == mem_logits.shape[0]
        
        for idx in range(max_time_step):
            hidden = self.decoder(
                tgt.transpose(0, 1),
                memory=context_encoding["code_token_encoding"].transpose(0, 1),
                tgt_mask=tgt_mask[: idx + 1, : idx + 1],
                tgt_key_padding_mask=~input_dict["target_mask"][:, : idx + 1],
                memory_key_padding_mask=~context_encoding["code_token_mask"],
            ).transpose(0, 1)
            # Save logits for the current step
            logits = self.output(hidden[:, -1:])
            # Make prediction for this step
            preds_step = []
            for b in range(batch_size):
                if not input_dict["target_mask"][b, idx]:
                    preds_step.append(torch.zeros(1, dtype=torch.long, device=logits.device))
                    continue
                scores = logits[b, 0]
                if self.mem_mask == "hard":
                    pred, tgt_mems[b] = self.pred_with_mem(scores, tgt_mems[b])
                elif self.mem_mask == "none":
                    pred = scores.argmax(dim=0, keepdim=True)
                elif self.mem_mask == "soft":
                    scores += mem_logits_list[b][idx]
                    pred = scores.argmax(dim=0, keepdim=True)
                    pass
                else:
                    raise NotImplementedError
                preds_step.append(pred)
            pred_step = torch.cat(preds_step)
            preds_list.append(pred_step)
            # Update tgt for next step with prediction at the current step
            if idx < max_time_step - 1:
                tgt_step = torch.cat(
                    [context_encoding["variable_encoding"][:, idx + 1 : idx + 2], self.target_embedding(pred_step.unsqueeze(dim=1))],
                    dim=-1,
                )
                tgt_step = self.target_transform(tgt_step)
                tgt = torch.cat([tgt, tgt_step], dim=1)
        preds = torch.stack(preds_list).transpose(0, 1)
        return preds[input_dict["target_mask"]]

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
