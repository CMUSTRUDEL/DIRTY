from typing import Dict, List

import torch
from torch import nn as nn
from utils import util
from utils.vocab import Vocab


class SimpleDecoder(nn.Module):
    def __init__(self, config):
        super(SimpleDecoder, self).__init__()

        self.vocab = vocab = Vocab.load(config["vocab_file"])
        self.output = nn.Linear(
            config["hidden_size"],
            len(vocab.names) if config.get("rename", False) else len(vocab.types),
            bias=True,
        )
        self.config: Dict = config

    @classmethod
    def default_params(cls):
        return {
            "vocab_file": None,
            "hidden_size": 128,
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
        logits = self.output(context_encoding["variable_encoding"])
        return logits

    def predict(
        self,
        context_encoding: Dict[str, torch.Tensor],
        target_dict: Dict[str, torch.Tensor],
        variable_type_logits: torch.Tensor,
    ):
        return variable_type_logits[target_dict["target_mask"]].argmax(dim=1)
