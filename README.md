<p align="center">
  <img src="fig.png" width="800">
  <br />
  <br />
  <a href="https://github.com/CMUSTRUDEL/DIRTY/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/CMUSTRUDEL/DIRTY" /></a>
  <a href="https://github.com/ambv/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
</p>

-------------------------------------

# DIRTY: Augmenting Decompiler Output with Learned Variable Names and Types

[**Code**](https://github.com/CMUSTRUDEL/DIRTY) | [**arXiv**](#common-issues) | [**Demo**](https://dirtdirty.github.io/explorer.html)

Original implementation for paper [Augmenting Decompiler Output with Learned Variable Names and Types](#common-issues).

**DIRTY** is a Transformer-based model which **improves the quality of decompiler outputs** by automatically generating meaningful variable names and types, assigning variable names that agree with those written by developers 66.4% of the time and types 75.8% of the time.
We also release a large real-world dataset **DIRT** for this task, which consists of 75K+ programs and 1M+ human-written C functions mined from GitHub paired with their decompiler outputs.

- [DIRTY: Augmenting Decompiler Output with Learned Variable Names and Types](#dirty-augmenting-decompiler-output-with-learned-variable-names-and-types)
  - [Installation](#installation)
    - [Requirements](#requirements)
  - [Quick Start](#quick-start)
    - [Training](#training)
      - [Download DIRT](#download-dirt)
      - [Train DIRTY](#train-dirty)
    - [Inference](#inference)
      - [Download Trained Model](#download-trained-model)
      - [Test DIRTY](#test-dirty)
  - [Common Issues](#common-issues)
  - [Citing DIRTY](#citing-dirty)

## Installation

### Requirements

- Linux with Python ≥ 3.6
- [PyTorch ≥ 1.5.1](https://pytorch.org/)
- `pip install -r requirements.txt`

## Quick Start

### Training

#### Download DIRT

The first step to train DIRTY is to download the preprocessed DIRT dataset.
If you wish to obtain the original unpreprocessed dataset, please open an issue.

```bash
cd dirty/
python ../scripts/download.py --url https://drive.google.com/open?id=1JWRkIlzdBPhpeSHe1KsJNuRid7KWmggk --path . --fname dirt.tar.gz
```

The command would automatically download and decompress the dataset from Google Drive.
If your machine does not have access to Google, please manually download from the above link and untar it to `data1/`.

#### Train DIRTY

We have setup configuration files for different models reported in our paper:

| file                      | model           | time (estimated hours) |
| ------------------------- | --------------- | ---------------------- |
| multitask.xfmr.jsonnet    | DIRTY-Multitask | 120                    |
| rename.xfmr.jsonnet       | DIRTY-Rename    | 80                     |
| retype.xfmr.jsonnet       | DIRTY-Retype    | 80                     |
| retype_nomem.xfmr.jsonnet | DIRTY_NDL       | 80                     |
| retype_base.xfmr.jsonnet  | DIRTY_S         | 40                     |

Training a models is as easy as specifying the name of the experimental run and the config file.
Suppose we want to reproduce the Multi-task model in Table~7 in the paper:

```bash
cd dirty/
python exp.py train --cuda --expname=dirty_mt multitask.xfmr.jsonnet
```

Then, please watch for the line `wandb: Run data is saved locally in ...` in the output.
This is where the logs and models are to be saved.
You can also monitor the automatically uploaded training and validation status (e.g., losses, accuracy) in your browser in real-time with the link printed after `wandb: 🚀 View run at ...`.

Feel free to adjust the hyperparameters in `*.jsonnet` config files to train your own model.

### Inference

#### Download Trained Model

As an alternative to train the model by yourself, you can download our trained DIRTY model (coming soon).

#### Test DIRTY

First, run your trained/downloaded model to produce predictions on the DIRE test set.

```
python exp.py train --cuda --expname=eval_dirty_mt multitask.xfmr.jsonnet --eval-ckpt wandb/run-YYYYMMDD_HHMMSS-XXXXXXXX/files/dire/XXXXXXXX/checkpoints/epoch=N.ckpt
```

The predictions will be saved to `pred_XXX.json`.
This filename is different for different models and can be modified in config files.

Then, use our standalone benchmark script:

```
python -m utils.evaluate --pred-file pred_mt.json --config-file multitask.xfmr.jsonnet
```

## Common Issues

<details>
<summary>
Where do I find the DIRTY paper?
</summary>
<br/>
We apologize for the inconvenience. It is currently under peer review.
</details>

## Citing DIRTY

If you use DIRTY/DIRT in your research or wish to refer to the baseline results, please use the following BibTeX.

```
Not available yet.
```
