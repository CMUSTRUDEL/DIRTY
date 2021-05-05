from typing import Dict, Union

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def forward(
        self, tensor_dict: Dict[str, Union[torch.Tensor, int]]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def build(config):
        from .xfmr_mem_encoder import XfmrMemEncoder
        from .xfmr_sequential_encoder import XfmrSequentialEncoder

        return {
            "XfmrSequentialEncoder": XfmrSequentialEncoder,
            "XfmrMemEncoder": XfmrMemEncoder,
        }[config["type"]](config)
