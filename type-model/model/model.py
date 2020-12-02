import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional.classification import accuracy, f1_score
from torch import nn
from torchvision import transforms
from utils.vocab import Vocab
from utils.dire_types import TypeInfo, TypeLibCodec

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
        self.subtype = config["decoder"]["type"] in ['XfmrSubtypeDecoder']
        self._preprocess()
        self.soft_mem_mask = config["decoder"]["mem_mask"] == "soft"
        if self.soft_mem_mask:
            self.mem_encoder = Encoder.build(config["mem_encoder"])
            self.mem_decoder = Decoder.build(config["mem_decoder"])
            # Used for decoding
            self.decoder.mem_encoder = self.mem_encoder
            self.decoder.mem_decoder = self.mem_decoder

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
        if self.soft_mem_mask:
            variable_type_logits = variable_type_logits[target_dict["target_mask"]]
            mem_encoding = self.mem_encoder(input_dict)
            mem_type_logits = self.mem_decoder(mem_encoding, target_dict)
            loss = F.cross_entropy(
                # cross_entropy requires num_classes at the second dimension
                variable_type_logits + mem_type_logits,
                target_dict["target_type_id"][target_dict["target_mask"]],
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
        if self.soft_mem_mask:
            variable_type_logits = variable_type_logits[target_dict["target_mask"]]
            mem_encoding = self.mem_encoder(input_dict)
            mem_type_logits = self.mem_decoder(mem_encoding, target_dict)
            loss = F.cross_entropy(
                # cross_entropy requires num_classes at the second dimension
                variable_type_logits + mem_type_logits,
                target_dict["target_type_id"][target_dict["target_mask"]],
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
        targets = target_dict["target_type_id"].detach().cpu()
        preds = self.decoder.predict(context_encoding, target_dict, variable_type_logits)
        if self.subtype:
            preds_out = []
            f1 = []
            for pred, target in zip(preds, targets):
                target = target[target > 0]
                # Compute piece-wise f1
                pred_pieces = [self.vocab.subtypes.id2word[pred_id] for pred_id in pred[pred > 0].tolist()]
                target_strs = [self.vocab.types.id2word[tp] for tp in target.tolist()]
                pred_piece_list = []
                # Group pieces separated by <eot>
                current = []
                for piece in pred_pieces:
                    if piece == "<eot>":
                        pred_piece_list.append(current)
                        current = []
                    else:
                        current.append(piece)
                pred_piece_list += ["<unk>"] * (len(target_strs) - len(pred_piece_list))
                for pred_piece, target_str in zip(pred_piece_list, target_strs):
                    target_pieces = self.typstr_to_piece[target_str]
                    f1.append(2 * len(set(pred_piece) & set(target_pieces)) / (1e-12 + len(pred_piece) + len(target_pieces)))
                pred_detok = TypeInfo.detokenize(pred_pieces)
                if len(pred_detok) < target.shape[0]:
                    pred_detok += ["<unk>"] * (target.shape[0] - len(pred_detok))
                elif len(pred_detok) > target.shape[0]:
                    import pdb
                    pdb.set_trace()
                preds_out.append(torch.tensor([self.vocab.types[pred_id] for pred_id in pred_detok]))
            targets = targets[target_dict["target_mask"]]
            assert len(f1) == targets.shape[0]
            preds = torch.cat(preds_out)
        else:
            targets = targets[target_dict["target_mask"]]
            preds = preds.detach().cpu()
            f1 = torch.zeros(targets.shape)
            for i, (pred_id, target_id) in enumerate(zip(preds.tolist(), targets.tolist())):
                pred_str = self.vocab.types.id2word[pred_id]
                target_str = self.vocab.types.id2word[target_id]
                pred_pieces = self.typstr_to_piece[pred_str]
                target_pieces = self.typstr_to_piece[target_str]
                f1[i] = (2 * len(set(pred_pieces) & set(target_pieces)) / (1e-12 + len(pred_pieces) + len(target_pieces)))

        return dict(
            loss=loss.detach().cpu(),
            preds=preds,
            f1=torch.tensor(f1),
            targets=targets,
            targets_nums=target_dict["target_mask"].sum(dim=1).detach().cpu(),
            test_meta=target_dict["test_meta"]
        )

    def validation_epoch_end(self, outputs):
        return self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        return self._shared_epoch_end(outputs, "test")

    def _shared_epoch_end(self, outputs, prefix):
        preds = torch.cat([x["preds"] for x in outputs])
        targets = torch.cat([x["targets"] for x in outputs])
        f1 = torch.cat([x["f1"] for x in outputs])
        print(preds.shape, f1.shape)
        assert f1.shape == preds.shape
        loss = torch.cat([x["loss"] for x in outputs]).mean()
        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_acc", accuracy(preds, targets))
        self.log(f"{prefix}_f1_macro", f1_score(preds, targets, class_reduction='macro'))
        self.log(f"{prefix}_F1", f1.mean())
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
        self.log(f"{prefix}_body_in_train_acc", accuracy(preds[body_in_train_mask], targets[body_in_train_mask]))
        self.log(f"{prefix}_body_not_in_train_acc", accuracy(preds[~body_in_train_mask], targets[~body_in_train_mask]))
        # self.log(f"{prefix}_name_in_train_acc", accuracy(preds[name_in_train_mask], targets[name_in_train_mask]))
        self.log(f"{prefix}_body_in_train_f1_macro", f1_score(preds[body_in_train_mask], targets[body_in_train_mask], class_reduction='macro'))
        self.log(f"{prefix}_body_not_in_train_f1_macro", f1_score(preds[~body_in_train_mask], targets[~body_in_train_mask], class_reduction='macro'))
        # self.log(f"{prefix}_name_in_train_f1_macro", f1_score(preds[name_in_train_mask], targets[name_in_train_mask], class_reduction='macro'))
        assert pos == sum(x["targets_nums"].sum() for x in outputs), (pos, sum(x["targets_nums"].sum() for x in outputs))
        self.log(f"{prefix}_func_acc", num_correct / num_funcs)

        struc_mask = torch.zeros(len(targets), dtype=torch.bool)
        for idx, target in enumerate(targets):
            if target.item() in self.vocab.types.struct_set:
                struc_mask[idx] = 1
        if struc_mask.sum() > 0:
            self.log(f"{prefix}_struc_acc", accuracy(preds[struc_mask], targets[struc_mask]))
            self.log(f"{prefix}_body_in_train_struc_acc", accuracy(preds[struc_mask & body_in_train_mask], targets[struc_mask & body_in_train_mask]))
            self.log(f"{prefix}_body_not_in_train_struc_acc", accuracy(preds[~body_in_train_mask & struc_mask], targets[~body_in_train_mask & struc_mask]))
            # adjust for the number of classes
            self.log(f"{prefix}_struc_f1_macro", f1_score(preds[struc_mask], targets[struc_mask], class_reduction='macro') * len(self.vocab.types) / len(self.vocab.types.struct_set))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["train"]["lr"])
        return optimizer
