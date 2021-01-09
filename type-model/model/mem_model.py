from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy

from model.encoder import Encoder
from model.decoder import Decoder


class MemReconstructionModel(pl.LightningModule):
    def __init__(self, config, config_load=None):
        super().__init__()
        if config_load is not None:
            config = config_load
        self.encoder = Encoder.build(config["encoder"])
        self.decoder = Decoder.build(config["decoder"])
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
        variable_type_logits = self.decoder(context_encoding, target_dict)
        loss = F.cross_entropy(
            # cross_entropy requires num_classes at the second dimension
            variable_type_logits.transpose(1, 2),
            target_dict["tgt_starts"],
            reduction='none',
        )
        loss = loss[target_dict["tgt_starts_mask"]]
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
            target_dict["tgt_starts"],
            reduction='none',
        )
        loss = loss[target_dict["tgt_starts_mask"]]
        preds = self.decoder.predict(context_encoding, target_dict, variable_type_logits)
        targets = target_dict["tgt_starts"][target_dict["tgt_starts_mask"]]
        # pos = 0
        # for num in target_dict["tgt_starts_mask"].sum(dim=1).detach().cpu().tolist():
        #     print(preds[pos:pos + num], targets[pos:pos + num])
        #     pos += num
        return dict(
            loss=loss.detach().cpu(),
            preds=preds.detach().cpu(),
            targets=targets.detach().cpu(),
            targets_nums=target_dict["tgt_starts_mask"].sum(dim=1).detach().cpu()
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
        self.log(f"{prefix}_acc_macro", accuracy(preds, targets, class_reduction='macro'))
        # func acc
        num_correct, num_funcs, pos = 0, 0, 0
        for target_num in map(lambda x: x["targets_nums"], outputs):
            for num in target_num.tolist():
                num_correct += all(preds[pos:pos + num] == targets[pos:pos + num])
                pos += num
            num_funcs += len(target_num)
        assert pos == sum(x["targets_nums"].sum() for x in outputs), (pos, sum(x["targets_nums"].sum() for x in outputs))
        self.log(f"{prefix}_func_acc", num_correct / num_funcs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["train"]["lr"])
        return optimizer
