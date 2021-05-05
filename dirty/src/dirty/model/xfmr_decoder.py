from typing import Dict

import torch
import torch.nn as nn
from csvnpm.binary.dire_types import TypeLibCodec
from torch.nn import LayerNorm, TransformerDecoder, TransformerDecoderLayer

from dirty.utils import util
from dirty.utils.vocab import Vocab

from .beam import Beam


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    x = x.repeat(count, *(1,) * x.dim()).transpose(0, 1).contiguous().view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


class XfmrDecoder(nn.Module):
    def __init__(self, config):
        super(XfmrDecoder, self).__init__()

        self.vocab = Vocab.load(config["vocab_file"])
        with open(config["typelib_file"]) as type_f:
            self.typelib = TypeLibCodec.decode(type_f.read())
        vocab_size = (
            len(self.vocab.names)
            if config.get("rename", False)
            else len(self.vocab.types)
        )
        self.target_id_key = (
            "target_name_id" if config.get("rename", False) else "target_type_id"
        )
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
            config["num_heads"],
            4 * config["hidden_size"],
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
            "num_heads": 1,
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
        # mask out attention to subsequent inputs which include the ground
        # truth for current step
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
        input_dict: Dict[str, torch.Tensor],
        variable_type_logits: torch.Tensor,
        beam_size: int = 0,
    ):
        if beam_size == 0:
            return self.greedy_decode(
                context_encoding, input_dict, variable_type_logits
            )
        else:
            return self.beam_decode(
                context_encoding, input_dict, variable_type_logits, beam_size
            )

    def greedy_decode(
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
        tgt_mask = XfmrDecoder.generate_square_subsequent_mask(
            max_time_step, tgt.device
        )
        preds_list = []
        if self.mem_mask == "soft":
            mem_encoding = self.mem_encoder(input_dict)
            mem_logits = self.mem_decoder(mem_encoding, target_dict=None)
            mem_logits_list = []
            idx = 0
            for b in range(batch_size):
                nvar = input_dict["target_mask"][b].sum().item()
                mem_logits_list.append(mem_logits[idx : idx + nvar])
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
                    preds_step.append(
                        torch.zeros(1, dtype=torch.long, device=logits.device)
                    )
                    continue
                scores = logits[b, 0]
                if self.mem_mask == "none":
                    pred = scores.argmax(dim=0, keepdim=True)
                elif self.mem_mask == "soft":
                    scores += mem_logits_list[b][idx]
                    pred = scores.argmax(dim=0, keepdim=True)
                else:
                    raise NotImplementedError
                preds_step.append(pred)
            pred_step = torch.cat(preds_step)
            preds_list.append(pred_step)
            # Update tgt for next step with prediction at the current step
            if idx < max_time_step - 1:
                tgt_step = torch.cat(
                    [
                        context_encoding["variable_encoding"][:, idx + 1 : idx + 2],
                        self.target_embedding(pred_step.unsqueeze(dim=1)),
                    ],
                    dim=-1,
                )
                tgt_step = self.target_transform(tgt_step)
                tgt = torch.cat([tgt, tgt_step], dim=1)
        preds = torch.stack(preds_list).transpose(0, 1)
        return preds[input_dict["target_mask"]]

    def beam_decode(
        self,
        context_encoding: Dict[str, torch.Tensor],
        input_dict: Dict[str, torch.Tensor],
        variable_type_logits: torch.Tensor,
        beam_size: int = 5,
        length_norm: bool = True,
    ):
        """Beam search decoding"""

        batch_size, max_time_step, _ = context_encoding["variable_encoding"].shape
        tgt = torch.zeros(batch_size, 1, self.config["target_embedding_size"]).to(
            context_encoding["variable_encoding"].device
        )
        tgt = self.target_transform(
            torch.cat([context_encoding["variable_encoding"][:, :1], tgt], dim=-1)
        )
        tgt_mask = XfmrDecoder.generate_square_subsequent_mask(
            max_time_step, tgt.device
        )
        if self.mem_mask == "soft":
            mem_encoding = self.mem_encoder(input_dict)
            mem_logits = self.mem_decoder(mem_encoding, target_dict=None)
            mem_logits_list = []
            idx = 0
            for b in range(batch_size):
                nvar = input_dict["target_mask"][b].sum().item()
                mem_logits_list.append(mem_logits[idx : idx + nvar])
                idx += nvar
            assert idx == mem_logits.shape[0]

        beams = [
            Beam(
                beam_size,
                n_best=1,
                cuda=tgt.device.type == "cuda",
                length_norm=length_norm,
            )
            for _ in range(batch_size)
        ]

        # Tensor shapes
        # tgt: batch, time, hidden
        # context_encoding["code_token_encoding"]: batch, len, hidden
        # input_dict["target_mask"]: batch, max_time
        # tgt_mask: max_time, max_time
        # context_encoding["code_token_mask"]: batch, len
        # context_encoding["variable_encoding"]: batch, max_time, hidden

        tgt = tile(tgt, beam_size, dim=0)
        target_mask = tile(input_dict["target_mask"], beam_size, dim=0)
        code_token_encoding = tile(
            context_encoding["code_token_encoding"], beam_size, dim=0
        )
        code_token_mask = tile(context_encoding["code_token_mask"], beam_size, dim=0)
        variable_encoding = tile(
            context_encoding["variable_encoding"], beam_size, dim=0
        )

        for idx in range(max_time_step):
            hidden = self.decoder(
                tgt.transpose(0, 1),
                memory=code_token_encoding.transpose(0, 1),
                tgt_mask=tgt_mask[: idx + 1, : idx + 1],
                tgt_key_padding_mask=~target_mask[:, : idx + 1],
                memory_key_padding_mask=~code_token_mask,
            ).transpose(0, 1)
            # Save logits for the current step
            logits = self.output(hidden[:, -1:])
            scores = logits[:, 0].view(batch_size, beam_size, -1)
            select_indices_array = []
            for b, bm in enumerate(beams):
                if not input_dict["target_mask"][b, idx]:
                    select_indices_array.append(
                        torch.arange(beam_size).to(tgt.device) + b * beam_size
                    )
                    continue
                if self.mem_mask == "soft" and input_dict["target_mask"][b, idx]:
                    scores[b] += mem_logits_list[b][idx]
                bm.advance(torch.log_softmax(scores[b], dim=1))
                select_indices_array.append(bm.getCurrentOrigin() + b * beam_size)
            select_indices = torch.cat(select_indices_array)
            tgt = tgt[select_indices]
            pred_step = torch.stack(
                [
                    bm.getCurrentState()
                    if input_dict["target_mask"][b, idx]
                    else torch.zeros(beam_size, dtype=torch.long).to(tgt.device)
                    for b, bm in enumerate(beams)
                ]
            ).view(-1)
            # Update tgt for next step with prediction at the current step
            if idx < max_time_step - 1:
                tgt_step = torch.cat(
                    [
                        variable_encoding[:, idx + 1 : idx + 2],
                        self.target_embedding(pred_step.unsqueeze(dim=1)),
                    ],
                    dim=-1,
                )
                tgt_step = self.target_transform(tgt_step)
                tgt = torch.cat([tgt, tgt_step], dim=1)

        all_hyps, all_scores = [], []
        for j in range(batch_size):
            b = beams[j]
            scores, ks = b.sortFinished(minimum=beam_size)
            times, k = ks[0]
            hyp = b.getHyp(times, k)
            all_hyps.append(torch.tensor(hyp))
            all_scores.append(scores[0])

        return torch.cat(all_hyps)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        r"""
        Generate a square mask for the sequence. The masked positions are filled with
          float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


class XfmrInterleaveDecoder(XfmrDecoder):
    def __init__(self, config):
        super(XfmrDecoder, self).__init__()

        self.vocab = Vocab.load(config["vocab_file"])
        with open(config["typelib_file"]) as type_f:
            self.typelib = TypeLibCodec.decode(type_f.read())

        retype_vocab_size = len(self.vocab.types)
        rename_vocab_size = len(self.vocab.names)
        self.target_embedding = nn.Embedding(
            retype_vocab_size + rename_vocab_size,
            config["target_embedding_size"],
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
        self.output = nn.Linear(
            config["hidden_size"], retype_vocab_size + rename_vocab_size
        )
        self.mem_mask = config["mem_mask"]
        self.config: Dict = config
        self.retype_vocab_size = retype_vocab_size

    @staticmethod
    def interleave_2d(tensor1, tensor2):
        return torch.cat((tensor1.view(-1, 1), tensor2.view(-1, 1)), dim=1).view(
            tensor1.shape[0], tensor1.shape[1] * 2
        )

    @staticmethod
    def interleave_3d(tensor1, tensor2):
        d3 = tensor1.shape[2]
        return torch.cat(
            (tensor1.view(-1, 1, d3), tensor2.view(-1, 1, d3)), dim=1
        ).view(tensor1.shape[0], tensor1.shape[1] * 2, d3)

    @staticmethod
    def devinterleave_2d(tensor):
        d1, d2 = tensor.shape[0], tensor.shape[1] // 2
        tensor = tensor.view(-1, 2)
        return tensor[:, 0].view(d1, d2), tensor[:, 1].tensor2.view(d1, d2)

    @staticmethod
    def devinterleave_3d(tensor):
        d1, d2, d3 = tensor.shape[0], tensor.shape[1] // 2, tensor.shape[2]
        tensor = tensor.view(-1, 2, d3)
        return tensor[:, 0].view(d1, d2, d3), tensor[:, 1].view(d1, d2, d3)

    def forward(
        self,
        context_encoding: Dict[str, torch.Tensor],
        target_dict: Dict[str, torch.Tensor],
    ):
        # Interleave type1, name1, type2, name2
        # interleave target
        # (B, NUM_VAR) -> (B, NUM_VAR * 2)
        tgt = XfmrInterleaveDecoder.interleave_2d(
            target_dict["target_type_id"],
            self.retype_vocab_size + target_dict["target_name_id"],
        )
        # (B, NUM_VAR) -> (B, NUM_VAR, H)
        tgt = self.target_embedding(tgt)
        # interleave variable encoding
        # (B, NUM_VAR, H) -> (B, NUM_VAR * 2, H)
        variable_encoding = XfmrInterleaveDecoder.interleave_3d(
            context_encoding["variable_encoding"],
            context_encoding["variable_encoding"],
        )

        # Shift 1 position to right
        tgt = torch.cat([torch.zeros_like(tgt[:, :1]), tgt[:, :-1]], dim=1)
        tgt = torch.cat([variable_encoding, tgt], dim=-1)
        tgt = self.target_transform(tgt)
        # mask out attention to subsequent inputs which include the ground
        # truth for current step
        tgt_mask = XfmrDecoder.generate_square_subsequent_mask(tgt.shape[1], tgt.device)
        # TransformerModels have batch_first=False
        tgt_padding_mask = XfmrInterleaveDecoder.interleave_2d(
            target_dict["target_mask"], target_dict["target_mask"]
        )
        hidden = self.decoder(
            tgt.transpose(0, 1),
            memory=context_encoding["code_token_encoding"].transpose(0, 1),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=~tgt_padding_mask,
            memory_key_padding_mask=~context_encoding["code_token_mask"],
        ).transpose(0, 1)
        logits = self.output(hidden)
        type_logits, name_logits = XfmrInterleaveDecoder.devinterleave_3d(logits)
        return (
            type_logits[:, :, : self.retype_vocab_size],
            name_logits[:, :, self.retype_vocab_size :],
        )

    def predict(
        self,
        context_encoding: Dict[str, torch.Tensor],
        input_dict: Dict[str, torch.Tensor],
        variable_type_logits: torch.Tensor,
        beam_size: int = 0,
    ):
        if beam_size == 0:
            return self.greedy_decode(
                context_encoding, input_dict, variable_type_logits
            )
        else:
            return self.beam_decode(
                context_encoding, input_dict, variable_type_logits, beam_size
            )

    def greedy_decode(
        self,
        context_encoding: Dict[str, torch.Tensor],
        input_dict: Dict[str, torch.Tensor],
        variable_type_logits: torch.Tensor,
    ):
        """Greedy decoding"""

        variable_encoding = XfmrInterleaveDecoder.interleave_3d(
            context_encoding["variable_encoding"],
            context_encoding["variable_encoding"],
        )
        tgt_padding_mask = XfmrInterleaveDecoder.interleave_2d(
            input_dict["target_mask"], input_dict["target_mask"]
        )
        batch_size, max_time_step, _ = variable_encoding.shape
        tgt = torch.zeros(batch_size, 1, self.config["target_embedding_size"]).to(
            variable_encoding.device
        )
        tgt = self.target_transform(torch.cat([variable_encoding[:, :1], tgt], dim=-1))
        tgt_mask = XfmrDecoder.generate_square_subsequent_mask(
            max_time_step, tgt.device
        )
        if self.mem_mask == "soft":
            mem_encoding = self.mem_encoder(input_dict)
            mem_logits = self.mem_decoder(mem_encoding, target_dict=None)
            mem_logits_list = []
            idx = 0
            for b in range(batch_size):
                nvar = input_dict["target_mask"][b].sum().item()
                mem_logits_list.append(mem_logits[idx : idx + nvar])
                idx += nvar
            assert idx == mem_logits.shape[0]

        type_preds_list = []
        name_preds_list = []
        for idx in range(max_time_step):
            hidden = self.decoder(
                tgt.transpose(0, 1),
                memory=context_encoding["code_token_encoding"].transpose(0, 1),
                tgt_mask=tgt_mask[: idx + 1, : idx + 1],
                tgt_key_padding_mask=~tgt_padding_mask[:, : idx + 1],
                memory_key_padding_mask=~context_encoding["code_token_mask"],
            ).transpose(0, 1)
            # Save logits for the current step
            logits = self.output(hidden[:, -1:])
            # Make prediction for this step
            if idx % 2 == 0:
                # pred type
                preds_step = []
                for b in range(batch_size):
                    if not tgt_padding_mask[b, idx]:
                        preds_step.append(
                            torch.zeros(1, dtype=torch.long, device=logits.device)
                        )
                        continue
                    scores = logits[b, 0, : self.retype_vocab_size]
                    if self.mem_mask == "none":
                        pred = scores.argmax(dim=0, keepdim=True)
                    elif self.mem_mask == "soft":
                        scores += mem_logits_list[b][idx // 2]
                        pred = scores.argmax(dim=0, keepdim=True)
                    preds_step.append(pred)
                pred_step = torch.cat(preds_step)
                type_preds_list.append(pred_step)
            else:
                # pred type
                preds_step = []
                for b in range(batch_size):
                    if not tgt_padding_mask[b, idx]:
                        preds_step.append(
                            torch.zeros(1, dtype=torch.long, device=logits.device)
                        )
                        continue
                    scores = logits[b, 0, self.retype_vocab_size :]
                    pred = scores.argmax(dim=0, keepdim=True)
                    preds_step.append(pred)
                pred_step = torch.cat(preds_step)
                name_preds_list.append(pred_step)
            # Update tgt for next step with prediction at the current step
            if idx < max_time_step - 1:
                tgt_step = torch.cat(
                    [
                        variable_encoding[:, idx + 1 : idx + 2],
                        self.target_embedding(
                            (
                                pred_step
                                if idx % 2 == 0
                                else pred_step + self.retype_vocab_size
                            ).unsqueeze(dim=1)
                        ),
                    ],
                    dim=-1,
                )
                tgt_step = self.target_transform(tgt_step)
                tgt = torch.cat([tgt, tgt_step], dim=1)
        type_preds = torch.stack(type_preds_list).transpose(0, 1)
        name_preds = torch.stack(name_preds_list).transpose(0, 1)
        return (
            type_preds[input_dict["target_mask"]],
            name_preds[input_dict["target_mask"]],
        )

    def beam_decode(
        self,
        context_encoding: Dict[str, torch.Tensor],
        input_dict: Dict[str, torch.Tensor],
        variable_type_logits: torch.Tensor,
        beam_size: int = 5,
        length_norm: bool = True,
    ):
        """Beam search decoding"""

        variable_encoding = XfmrInterleaveDecoder.interleave_3d(
            context_encoding["variable_encoding"],
            context_encoding["variable_encoding"],
        )
        tgt_padding_mask = XfmrInterleaveDecoder.interleave_2d(
            input_dict["target_mask"], input_dict["target_mask"]
        )
        batch_size, max_time_step, _ = variable_encoding.shape
        tgt = torch.zeros(batch_size, 1, self.config["target_embedding_size"]).to(
            variable_encoding.device
        )
        tgt = self.target_transform(torch.cat([variable_encoding[:, :1], tgt], dim=-1))
        tgt_mask = XfmrDecoder.generate_square_subsequent_mask(
            max_time_step, tgt.device
        )
        if self.mem_mask == "soft":
            mem_encoding = self.mem_encoder(input_dict)
            mem_logits = self.mem_decoder(mem_encoding, target_dict=None)
            mem_logits_list = []
            idx = 0
            for b in range(batch_size):
                nvar = input_dict["target_mask"][b].sum().item()
                mem_logits_list.append(mem_logits[idx : idx + nvar])
                idx += nvar
            assert idx == mem_logits.shape[0]

        beams = [
            Beam(
                beam_size,
                n_best=1,
                cuda=tgt.device.type == "cuda",
                length_norm=length_norm,
            )
            for _ in range(batch_size)
        ]

        # Tensor shapes
        # tgt: batch, time, hidden
        # context_encoding["code_token_encoding"]: batch, len, hidden
        # input_dict["target_mask"]: batch, max_time
        # tgt_mask: max_time, max_time
        # context_encoding["code_token_mask"]: batch, len
        # context_encoding["variable_encoding"]: batch, max_time, hidden

        tgt = tile(tgt, beam_size, dim=0)
        tiled_target_mask = tile(tgt_padding_mask, beam_size, dim=0)
        code_token_encoding = tile(
            context_encoding["code_token_encoding"], beam_size, dim=0
        )
        code_token_mask = tile(context_encoding["code_token_mask"], beam_size, dim=0)
        variable_encoding = tile(variable_encoding, beam_size, dim=0)

        for idx in range(max_time_step):
            hidden = self.decoder(
                tgt.transpose(0, 1),
                memory=code_token_encoding.transpose(0, 1),
                tgt_mask=tgt_mask[: idx + 1, : idx + 1],
                tgt_key_padding_mask=~tiled_target_mask[:, : idx + 1],
                memory_key_padding_mask=~code_token_mask,
            ).transpose(0, 1)
            # Save logits for the current step
            logits = self.output(hidden[:, -1:])
            scores = logits[:, 0].view(batch_size, beam_size, -1)
            select_indices_array = []
            for b, bm in enumerate(beams):
                if not tgt_padding_mask[b, idx]:
                    select_indices_array.append(
                        torch.arange(beam_size).to(tgt.device) + b * beam_size
                    )
                    continue
                if idx % 2 == 0:
                    s = scores[b, :, : self.retype_vocab_size]
                    if self.mem_mask == "soft" and tgt_padding_mask[b, idx]:
                        s += mem_logits_list[b][idx // 2]
                else:
                    s = scores[b, :, self.retype_vocab_size :]
                bm.advance(torch.log_softmax(s, dim=1))
                select_indices_array.append(bm.getCurrentOrigin() + b * beam_size)
            select_indices = torch.cat(select_indices_array).long()
            tgt = tgt[select_indices]
            pred_step = (
                torch.stack(
                    [
                        bm.getCurrentState()
                        if tgt_padding_mask[b, idx]
                        else torch.zeros(beam_size, dtype=torch.long).to(tgt.device)
                        for b, bm in enumerate(beams)
                    ]
                )
                .view(-1)
                .long()
            )
            # Update tgt for next step with prediction at the current step
            if idx < max_time_step - 1:
                tgt_step = torch.cat(
                    [
                        variable_encoding[:, idx + 1 : idx + 2],
                        self.target_embedding(
                            (
                                pred_step
                                if idx % 2 == 0
                                else pred_step + self.retype_vocab_size
                            ).unsqueeze(dim=1)
                        ),
                    ],
                    dim=-1,
                )
                tgt_step = self.target_transform(tgt_step)
                tgt = torch.cat([tgt, tgt_step], dim=1)

        all_type_hyps, all_name_hyps, all_scores = [], [], []
        for j in range(batch_size):
            b = beams[j]
            scores, ks = b.sortFinished(minimum=beam_size)
            times, k = ks[0]
            hyp = b.getHyp(times, k)
            hyp = torch.tensor(hyp).view(-1, 2).t()
            all_type_hyps.append(hyp[0])
            all_name_hyps.append(hyp[1])
            all_scores.append(scores[0])

        return torch.cat(all_type_hyps), torch.cat(all_name_hyps)
