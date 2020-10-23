from typing import Dict
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def forward(
        self,
        context_encoding: Dict[str, torch.Tensor],
        target_dict: Dict[str, torch.Tensor],
    ):
        raise NotImplementedError

    def predict(
        self,
        context_encoding: Dict[str, torch.Tensor],
        target_dict: Dict[str, torch.Tensor],
        variable_type_logits: torch.Tensor,
    ):
        raise NotImplementedError

    @staticmethod
    def build(config):
        from .simple_decoder import SimpleDecoder
        from .xfmr_decoder import XfmrDecoder
        from .xfmr_mem_decoder import XfmrMemDecoder

        return {
            "SimpleDecoder": SimpleDecoder,
            "XfmrDecoder": XfmrDecoder,
            "XfmrMemDecoder": XfmrMemDecoder,
        }[config["type"]](config)
