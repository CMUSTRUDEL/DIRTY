"""
Variable renaming

Usage:
    exp.py train [options] CONFIG_FILE
    exp.py test [options] MODEL_FILE TEST_DATA_FILE

Options:
    -h --help                                   Show this screen
    --cuda                                      Use GPU
    --debug                                     Debug mode
    --seed=<int>                                Seed [default: 0]
    --work-dir=<dir>                            work dir [default: data/exp_runs/]
    --extra-config=<str>                        extra config [default: {}]
    --ensemble                                  Use ensemble
    --save-to=<str>                             Save decode results to path
"""
import json
import os
import random
import sys
from typing import Dict, Iterable, List, Tuple

import _jsonnet
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from docopt import docopt
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from model.model import TypeReconstructionModel
from utils import util
from utils.dataset import Dataset


def train(args):
    work_dir = args["--work-dir"]
    config = json.loads(_jsonnet.evaluate_file(args["CONFIG_FILE"]))
    config["work_dir"] = work_dir

    if not os.path.exists(work_dir):
        print(f"creating work dir [{work_dir}]", file=sys.stderr)
        os.makedirs(work_dir)

    if args["--extra-config"]:
        extra_config = args["--extra-config"]
        extra_config = json.loads(extra_config)
        config = util.update(config, extra_config)

    json.dump(config, open(os.path.join(work_dir, "config.json"), "w"), indent=2)

    # dataloaders
    batch_size = config["train"]["batch_size"]
    train_set = Dataset(config["data"]["train_file"], config["data"])
    dev_set = Dataset(config["data"]["dev_file"], config["data"])
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=Dataset.collate_fn,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dev_set,
        batch_size=batch_size,
        collate_fn=Dataset.collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    # model
    model = TypeReconstructionModel(config)

    wandb_logger = WandbLogger(name=work_dir, project="dire")
    wandb_logger.log_hyperparams(config)
    trainer = pl.Trainer(
        max_epochs=config["train"]["max_epoch"],
        logger=wandb_logger,
        gpus=1 if args["--cuda"] else None,
        auto_select_gpus=True,
        val_check_interval=5000,
        gradient_clip_val=1,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    cmd_args = docopt(__doc__)
    print(f"Main process id {os.getpid()}", file=sys.stderr)

    # seed the RNG
    seed = int(cmd_args["--seed"])
    print(f"use random seed {seed}", file=sys.stderr)
    torch.manual_seed(seed)

    use_cuda = cmd_args["--cuda"]
    if use_cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    random.seed(seed * 17 // 7)

    if cmd_args["train"]:
        train(cmd_args)
