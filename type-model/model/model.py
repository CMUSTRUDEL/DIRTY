import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional.classification import accuracy, f1_score
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
            reduction='none',
        )
        loss = loss[target_dict["target_mask"]]
        loss = loss.mean()
        self.log('train_loss', loss)
        return loss

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
            reduction='none',
        )
        loss = loss[target_dict["target_mask"]]
        preds = self.decoder.predict(context_encoding, target_dict, variable_type_logits)
        targets = target_dict["target_type_id"][target_dict["target_mask"]]
        return dict(
            loss=loss.detach().cpu(),
            preds=preds.detach().cpu(),
            targets=targets.detach().cpu(),
            targets_nums=target_dict["target_mask"].sum(dim=1).detach().cpu()
        )

    def validation_epoch_end(self, outputs):
        return self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        return self._shared_epoch_end(outputs, "test")

    def _shared_epoch_end(self, outputs, prefix):
        preds = torch.cat([x["preds"] for x in outputs])
        targets = torch.cat([x["targets"] for x in outputs])
        loss = torch.cat([x["loss"] for x in outputs]).mean()
        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_acc", accuracy(preds, targets))
        self.log(f"{prefix}_f1_macro", f1_score(preds, targets, class_reduction='macro'))
        # func acc
        num_correct, num_funcs, pos = 0, 0, 0
        for target_num in map(lambda x: x["targets_nums"], outputs):
            for num in target_num.tolist():
                num_correct += all(preds[pos:pos + num] == targets[pos:pos + num])
                pos += num
            num_funcs += len(target_num)
        assert pos == sum(x["targets_nums"].sum() for x in outputs), (pos, sum(x["targets_nums"].sum() for x in outputs))
        self.log(f"{prefix}_func_acc", num_correct / num_funcs)

        struc_mask = torch.zeros(len(targets), dtype=torch.bool)
        for idx, target in enumerate(targets):
            if target.item() in self.vocab.types.struct_set:
                struc_mask[idx] = 1
        if struc_mask.sum() > 0:
            self.log(f"{prefix}_struc_acc", accuracy(preds[struc_mask], targets[struc_mask]))
            # adjust for the number of classes
            self.log(f"{prefix}_struc_f1_macro", f1_score(preds[struc_mask], targets[struc_mask], class_reduction='macro') * len(self.vocab.types) / len(self.vocab.types.struct_set))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["train"]["lr"])
        return optimizer
