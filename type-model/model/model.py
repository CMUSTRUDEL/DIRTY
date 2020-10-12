import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.metrics.sklearns import F1
from torch import nn
from torchvision import transforms
from utils.vocab import Vocab

from model.encoder import Encoder
from model.decoder import Decoder


class TypeReconstructionModel(pl.LightningModule):
    def __init__(self, config, config_load=None):
        super().__init__()
        if config_load is not None:
            config = config_load
        self.encoder = Encoder.build(config["encoder"])
        self.decoder = Decoder.build(config["decoder"])
        self.config = config
        self.vocab = Vocab.load(config["data"]["vocab_file"])
        self.f1_macro = F1(average="macro")
        self._preprocess_udt_idxs()

    def _preprocess_udt_idxs(self):
        self.vocab.types.struct_set = set()
        for idx, type_str in self.vocab.types.id2word.items():
            if type_str.startswith("struct"):
                self.vocab.types.struct_set.add(idx)

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
        variable_type_logits = self.decoder(context_encoding, target_dict)
        loss = F.cross_entropy(
            # cross_entropy requires num_classes at the second dimension
            variable_type_logits.transpose(1, 2),
            target_dict["target_type_id"],
            reduce=False,
        )
        loss = loss[target_dict["target_mask"]]
        loss = loss.mean()
        result = pl.TrainResult(loss)
        result.log("train_loss", loss)

        pred = self.decoder.predict(variable_type_logits, target_dict)
        target = target_dict["target_type_id"][target_dict["target_mask"]]
        result.log("train_acc", accuracy(pred, target))

        return result

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def _shared_eval_step(
        self,
        batch: Tuple[Dict[str, Union[torch.Tensor, int]], Dict[str, torch.Tensor]],
        batch_idx,
    ):
        input_dict, target_dict = batch
        context_encoding = self.encoder(input_dict)
        variable_type_logits = self.decoder(context_encoding, target_dict)
        loss = F.cross_entropy(
            variable_type_logits.transpose(1, 2),
            target_dict["target_type_id"],
            reduce=False,
        )
        loss = loss[target_dict["target_mask"]]
        preds = self.decoder.predict(variable_type_logits, target_dict)
        targets = target_dict["target_type_id"][target_dict["target_mask"]]
        ret = {
            "loss": loss.detach().cpu(),
            "preds": preds.detach().cpu(),
            "targets": targets.detach().cpu(),
        }
        return ret

    def validation_epoch_end(self, outputs):
        return self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        return self._shared_epoch_end(outputs, "test")

    def _shared_epoch_end(self, outputs, prefix):
        preds = torch.cat([x["preds"] for x in outputs])
        targets = torch.cat([x["targets"] for x in outputs])
        loss = torch.cat([x["loss"] for x in outputs]).mean()
        result = pl.EvalResult(early_stop_on=loss, checkpoint_on=loss)
        result.log(f"{prefix}_loss", loss, prog_bar=True)
        result.log(f"{prefix}_acc", accuracy(preds, targets))
        result.log(f"{prefix}_f1_macro", self.f1_macro(preds, targets))
        struc_mask = torch.zeros(len(targets), dtype=torch.bool)
        for idx, target in enumerate(targets):
            if target.item() in self.vocab.types.struct_set:
                struc_mask[idx] = 1
        if struc_mask.sum() > 0:
            result.log(
                f"{prefix}_struc_acc", accuracy(preds[struc_mask], targets[struc_mask])
            )
            result.log(
                f"{prefix}_struc_f1_macro",
                self.f1_macro(preds[struc_mask], targets[struc_mask]),
            )
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["train"]["lr"])
        return optimizer
