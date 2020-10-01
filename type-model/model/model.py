import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torchvision import transforms

from model.simple_decoder import SimpleDecoder
from model.xfmr_sequential_encoder import XfmrSequentialEncoder


class TypeReconstructionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.encoder = XfmrSequentialEncoder.build(config["encoder"])
        self.decoder = SimpleDecoder.build(config["decoder"])
        self.config = config

    def forward(self, x_dict):
        embedding = self.encoder(x_dict)
        return embedding

    def training_step(
        self,
        batch: Tuple[Dict[str, Union[torch.Tensor, int]], Dict[str, torch.Tensor]],
        batch_idx,
    ):
        input_dict, target_dict = batch
        context_encoding = self.encoder(input_dict)
        # (batch_size, max_variable_memtion_num, num_types)
        variable_type_logits = self.decoder(context_encoding)
        # (batch_size, max_variable_memtion_num)
        loss = F.cross_entropy(
            variable_type_logits.transpose(1, 2),
            target_dict["target_type_id"],
            reduce=False,
        )
        loss = loss[target_dict["target_mask"]]
        loss = loss.mean()
        result = pl.TrainResult(loss)
        result.log("train_loss", loss)

        pred = variable_type_logits[target_dict["target_mask"]].argmax(dim=1)
        target = target_dict["target_type_id"][target_dict["target_mask"]]
        result.log("train_acc", accuracy(pred, target))

        return result

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(
        self,
        batch: Tuple[Dict[str, Union[torch.Tensor, int]], Dict[str, torch.Tensor]],
        batch_idx,
        prefix: str,
    ):
        input_dict, target_dict = batch
        context_encoding = self.encoder(input_dict)
        # (batch_size, max_variable_memtion_num, num_types)
        variable_type_logits = self.decoder(context_encoding)
        # (batch_size, max_variable_memtion_num)
        loss = F.cross_entropy(
            variable_type_logits.transpose(1, 2),
            target_dict["target_type_id"],
            reduce=False,
        )
        loss = loss[target_dict["target_mask"]]
        loss = loss.mean()
        result = pl.EvalResult()
        result.log(f"{prefix}_loss", loss)

        pred = variable_type_logits[target_dict["target_mask"]].argmax(dim=1)
        target = target_dict["target_type_id"][target_dict["target_mask"]]
        result.log(f"{prefix}_acc", accuracy(pred, target))
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["train"]["lr"])
        return optimizer
