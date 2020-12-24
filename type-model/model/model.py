import json
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torchvision import transforms
from utils.vocab import Vocab
from utils.dire_types import TypeInfo, TypeLibCodec

from model.encoder import Encoder
from model.decoder import Decoder


class RenamingDecodeModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder.build({**config["decoder"], "rename": True})

    def training_step(self, input_dict, context_encoding, target_dict):
        variable_type_logits = self.decoder(context_encoding, target_dict)
        loss = F.cross_entropy(
            # cross_entropy requires num_classes at the second dimension
            variable_type_logits.transpose(1, 2),
            target_dict["target_name_id"],
            reduction='none',
        )
        return loss[target_dict["target_mask"]].mean()

    def shared_eval_step(self, context_encoding, input_dict, target_dict):
        variable_type_logits = self.decoder(context_encoding, target_dict)
        loss = F.cross_entropy(
            # cross_entropy requires num_classes at the second dimension
            variable_type_logits.transpose(1, 2),
            target_dict["target_name_id"],
            reduction='none',
        )
        loss = loss[input_dict["target_mask"]]
        targets = target_dict["target_name_id"][input_dict["target_mask"]].detach().cpu()
        preds = self.decoder.predict(context_encoding, input_dict, variable_type_logits).detach().cpu()

        return dict(
            rename_loss=loss.detach().cpu(),
            rename_preds=preds,
            rename_targets=targets
        )

class RetypingDecodeModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder.build({**config["decoder"]})
        self.subtype = config["decoder"]["type"] in ['XfmrSubtypeDecoder']
        self.soft_mem_mask = config["decoder"]["mem_mask"] == "soft"
        if self.soft_mem_mask:
            self.mem_encoder = Encoder.build(config["mem_encoder"])
            self.mem_decoder = Decoder.build(config["mem_decoder"])
            self.decoder.mem_encoder = self.mem_encoder
            self.decoder.mem_decoder = self.mem_decoder
            
    def training_step(self, input_dict, context_encoding, target_dict):
        variable_type_logits = self.decoder(context_encoding, target_dict)
        if self.soft_mem_mask:
            variable_type_logits = variable_type_logits[target_dict["target_mask"]]
            mem_encoding = self.mem_encoder(input_dict)
            mem_type_logits = self.mem_decoder(mem_encoding, target_dict)
            loss = F.cross_entropy(
                variable_type_logits + mem_type_logits,
                target_dict["target_type_id"][target_dict["target_mask"]],
                reduction='none',
            )
        else:
            loss = F.cross_entropy(
                variable_type_logits.transpose(1, 2),
                target_dict["target_subtype_id"] if self.subtype else target_dict["target_type_id"],
                reduction='none',
            )
            loss = loss[target_dict["target_submask"] if self.subtype else target_dict["target_mask"]]

        return loss.mean()

    def shared_eval_step(self, context_encoding, input_dict, target_dict):
        variable_type_logits = self.decoder(context_encoding, target_dict)
        if self.soft_mem_mask:
            variable_type_logits = variable_type_logits[input_dict["target_mask"]]
            mem_encoding = self.mem_encoder(input_dict)
            mem_type_logits = self.mem_decoder(mem_encoding, target_dict)
            loss = F.cross_entropy(
                # cross_entropy requires num_classes at the second dimension
                variable_type_logits + mem_type_logits,
                target_dict["target_type_id"][input_dict["target_mask"]],
                reduction='none',
            )
        else:
            loss = F.cross_entropy(
                # cross_entropy requires num_classes at the second dimension
                variable_type_logits.transpose(1, 2),
                target_dict["target_subtype_id"] if self.subtype else target_dict["target_type_id"],
                reduction='none',
            )
            loss = loss[target_dict["target_submask"] if self.subtype else target_dict["target_mask"]]
        targets = target_dict["target_type_id"][input_dict["target_mask"]].detach().cpu()
        preds = self.decoder.predict(context_encoding, input_dict, variable_type_logits).detach().cpu()

        return dict(
            retype_loss=loss.detach().cpu(),
            retype_preds=preds,
            retype_targets=targets,
        )

class TypeReconstructionModel(pl.LightningModule):
    def __init__(self, config, config_load=None):
        super().__init__()
        if config_load is not None:
            config = config_load
        self.encoder = Encoder.build(config["encoder"])
        self.retype = config["data"].get("retype", False)
        self.rename = config["data"].get("rename", False)
        if self.retype:
            self.retyping_module = RetypingDecodeModule(config)
        if self.rename:
            self.renaming_module = RenamingDecodeModule(config)
        self.config = config
        self.vocab = Vocab.load(config["data"]["vocab_file"])
        self._preprocess()
        self.soft_mem_mask = config["decoder"]["mem_mask"] == "soft"

    def _preprocess(self):
        self.vocab.types.struct_set = set()
        for idx, type_str in self.vocab.types.id2word.items():
            if type_str.startswith("struct"):
                self.vocab.types.struct_set.add(idx)
        with open(self.config["data"]["typelib_file"]) as type_f:
            typelib = TypeLibCodec.decode(type_f.read())
            self.typstr_to_piece = {}
            for size in typelib:
                for _, tp in typelib[size]:
                    self.typstr_to_piece[str(tp)] = tp.tokenize()[:-1]
        self.typstr_to_piece["<unk>"] = ["<unk>"]

    def training_step(
        self,
        batch: Tuple[Dict[str, Union[torch.Tensor, int]], Dict[str, torch.Tensor]],
        batch_idx,
    ):
        input_dict, target_dict = batch
        total_loss = 0
        context_encoding = self.encoder(input_dict)
        if self.retype:
            loss = self.retyping_module.training_step(input_dict, context_encoding, target_dict)
            self.log('train_retype_loss', loss)
            total_loss += loss
        if self.rename:
            loss = self.renaming_module.training_step(input_dict, context_encoding, target_dict)
            self.log('train_rename_loss', loss)
            total_loss += loss
        self.log('train_loss', total_loss)
        return total_loss

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
        ret_dict = {}
        if self.retype:
            ret = self.retyping_module.shared_eval_step(context_encoding, input_dict, target_dict)
            ret_dict = {**ret, **ret_dict}
        if self.rename:
            ret = self.renaming_module.shared_eval_step(context_encoding, input_dict, target_dict)
            ret_dict = {**ret, **ret_dict}

        return dict(
            **ret_dict,
            targets_nums=input_dict["target_mask"].sum(dim=1).detach().cpu(),
            test_meta=target_dict["test_meta"],
            index=input_dict["index"],
            tgt_var_names=target_dict["tgt_var_names"]
        )

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        indexes, tgt_var_names, preds, targets, body_in_train_mask = self._shared_epoch_end(outputs, "test")
        return 
        raise NotImplementedError
        if "pred_file" in self.config["test"]:
            results, refs = {}, {}
            for (binary, func_name, decom_var_name), target_var_name, pred, target, body_in_train in zip(indexes, tgt_var_names, preds.tolist(), targets.tolist(), body_in_train_mask.tolist()):
                results.setdefault(binary, {}).setdefault(func_name, []).append((decom_var_name, self.output_vocab.id2word[pred]))
                refs.setdefault(binary, {}).setdefault(func_name, []).append((target_var_name, self.output_vocab.id2word[target], body_in_train))
            pred_file = self.config["test"]["pred_file"]
            ref_file = os.path.splitext(pred_file)[0] + "_ref.json"
            json.dump(results, open(pred_file, "w"))
            json.dump(refs, open(ref_file, "w"))

    def _shared_epoch_end(self, outputs, prefix):
        if self.retype:
            indexes, tgt_var_names, preds, targets, body_in_train_mask = self._shared_epoch_end_task(outputs, prefix, "retype")
        if self.rename:
            indexes, tgt_var_names, preds, targets, body_in_train_mask = self._shared_epoch_end_task(outputs, prefix, "rename")
        return indexes, tgt_var_names, preds, targets, body_in_train_mask

    def _shared_epoch_end_task(self, outputs, prefix, task):
        indexes = sum([x["index"] for x in outputs], [])
        tgt_var_names = sum([x["tgt_var_names"] for x in outputs], [])
        preds = torch.cat([x[f"{task}_preds"] for x in outputs])
        targets = torch.cat([x[f"{task}_targets"] for x in outputs])
        loss = torch.cat([x[f"{task}_loss"] for x in outputs]).mean()
        self.log(f"{prefix}_{task}_loss", loss)
        self.log(f"{prefix}_{task}_acc", accuracy(preds, targets))
        self.log(f"{prefix}_{task}_acc_macro", accuracy(preds, targets, num_classes=len(self.vocab.types), class_reduction='macro'))
        # func acc
        num_correct, num_funcs, pos = 0, 0, 0
        body_in_train_mask = []
        name_in_train_mask = []
        for target_num, test_metas in map(lambda x: (x["targets_nums"], x["test_meta"]), outputs):
            for num, test_meta in zip(target_num.tolist(), test_metas):
                num_correct += all(preds[pos:pos + num] == targets[pos:pos + num])
                pos += num
                body_in_train_mask += [test_meta["function_body_in_train"]] * num
                name_in_train_mask += [test_meta["function_name_in_train"]] * num
            num_funcs += len(target_num)
        body_in_train_mask = torch.tensor(body_in_train_mask)
        name_in_train_mask = torch.tensor(name_in_train_mask)
        self.log(f"{prefix}_{task}_body_in_train_acc", accuracy(preds[body_in_train_mask], targets[body_in_train_mask]))
        self.log(f"{prefix}_{task}_body_not_in_train_acc", accuracy(preds[~body_in_train_mask], targets[~body_in_train_mask]))
        assert pos == sum(x["targets_nums"].sum() for x in outputs), (pos, sum(x["targets_nums"].sum() for x in outputs))
        self.log(f"{prefix}_{task}_func_acc", num_correct / num_funcs)

        struc_mask = torch.zeros(len(targets), dtype=torch.bool)
        for idx, target in enumerate(targets):
            if target.item() in self.vocab.types.struct_set:
                struc_mask[idx] = 1
        task_str = "" if task == "retype" else f"_{task}"
        if struc_mask.sum() > 0:
            self.log(f"{prefix}{task_str}_struc_acc", accuracy(preds[struc_mask], targets[struc_mask]))
            # adjust for the number of classes
            self.log(f"{prefix}{task_str}_struc_acc_macro", accuracy(preds[struc_mask], targets[struc_mask], num_classes=len(self.vocab.types), class_reduction='macro') * len(self.vocab.types) / len(self.vocab.types.struct_set))
        if (struc_mask & body_in_train_mask).sum() > 0:
            self.log(f"{prefix}{task_str}_body_in_train_struc_acc", accuracy(preds[struc_mask & body_in_train_mask], targets[struc_mask & body_in_train_mask]))
        if (~body_in_train_mask & struc_mask).sum() > 0:
            self.log(f"{prefix}{task_str}_body_not_in_train_struc_acc", accuracy(preds[~body_in_train_mask & struc_mask], targets[~body_in_train_mask & struc_mask]))
        return indexes, tgt_var_names, preds, targets, body_in_train_mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["train"]["lr"])
        return optimizer
