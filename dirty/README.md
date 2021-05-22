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
  - [Structure](#structure)
    - [`dirty/`](#dirty)
      - [`dirty/exp.py`](#dirtyexppy)
      - [`dirty/*.xfmr.jsonnet`](#dirtyxfmrjsonnet)
      - [`dirty/model`](#dirtymodel)
      - [`dirty/utils`](#dirtyutils)
      - [`dirty/baselines`](#dirtybaselines)
    - [`binary/`](#binary)
    - [`idastubs/`](#idastubs)
    - [`dataset-gen/`](#dataset-gen)
    - [`dire/`](#dire)
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

As an alternative to train the model by yourself, you can download our trained DIRTY model.

```bash
cd dirty/
mkdir exp_runs/
python ../scripts/download.py --url https://drive.google.com/open?id=1iDxlF9nsU4fgy2DRDbGg0WLosABspdHg --path . --fname exp_runs/dirty_mt.ckpt
```

#### Test DIRTY

First, run your trained/downloaded model to produce predictions on the DIRE test set.

```
python exp.py train --cuda --expname=eval_dirty_mt multitask.xfmr.jsonnet --eval-ckpt <ckpt_path>
```

`<ckpt_path>` is either `exp_runs/dirty_mt.ckpt` if you download our trained model,
or saved during training at `wandb/run-YYYYMMDD_HHMMSS-XXXXXXXX/files/dire/XXXXXXXX/checkpoints/epoch=N.ckpt`.

We sugguest changing `beam_size` in config files to `0` to switch to greedy decoding, which is significantly faster.
The default configuration of `beam_size = 5` can take hours.

The predictions will be saved to `pred_XXX.json`.
This filename depends on models and can be modified in config files.
You can inspect the prediction results, which is in the following format.

```python
{
  binary: {
    func_name: {
      var1: [var1_retype, var1_rename], ...
    }, ...
  }, ...
}
```

Finally, use our standalone benchmark script:

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

## Structure

Here is a walk-through of the code files of this repo.

### `dirty/`

The `dirty/` folder contains the main code for the DIRTY model and DIRT dataset.

#### `dirty/exp.py`

The entry point for running DIRTY experiments.
It loads a configuration file, constructs a dataset instance, a model instance, and launches into a Trainer which runs training or inference according to configuration, and save logs and results in wandb.

#### `dirty/*.xfmr.jsonnet`

Configuration files for running DIRTY experiments.

#### `dirty/model`


This folder contains neural models consisting of the DIRTY model.

```
├── dirty
│   ├── model
│   │   ├── beam.py                     # Beam search
│   │   ├── decoder.py                  # factory class for building Decoders from configs
│   │   ├── encoder.py                  # factory class for building Encoders from configs
│   │   ├── model.py                    # training and evaluation step and metric logging
│   │   ├── simple_decoder.py           # A `decoder' consists of a linear layer,
                                        # used for producing a soft mask from Data Layout Encoder
│   │   ├── xfmr_decoder.py             # Type/Multitask Decoder
│   │   ├── xfmr_mem_encoder.py         # Data Layout Encoder
│   │   ├── xfmr_sequential_encoder.py  # Code Encoder
│   │   └── xfmr_subtype_decoder.py     # Not used in the current version
```

#### `dirty/utils`

This folder contains code for the DIRT dataset, data preprocessing, evaluation, helper functions, and demos in the paper.

```
├── dirty
│   └── utils
│       ├── case_study.py           # Generate results for Table 3 and Table 6 in the paper
│       ├── code_processing.py      # Code canonicalization such as converting literals
│       ├── compute_mi.py           # Compute the mutual information between variables and types as a proof-of-concept for MT
│       ├── dataset.py              # A parallelized data loading class for preparing batched samples from DIRT for DIRTY
│       ├── dataset_statistics.py   # Compute dataset statistics
│       ├── dire_types.py -> ../../binary/dire_types.py
│       ├── evaluate.py             # Evaluate final scores from json files saved from different methods for fair comparison
│       ├── function.py -> ../../binary/function.py
│       ├── ida_ast.py -> ../../binary/ida_ast.py
│       ├── lexer.py
│       ├── preprocess.py           # Preprocess data produced from `dataset-gen/` into the DIRT dataset
│       ├── util.py
│       ├── variable.py -> ../../binary/variable.py
│       └── vocab.py
```

#### `dirty/baselines`

Empirical baselines included in the paper.
Use `python -m baselines.<xxxxxx>` to run.
Results are saved to corresponding json files and can be evaluated with `python -m utils.evaluate`.

```
├── dirty
│   ├── baselines
│   │   ├── copy_decompiler.py
│   │   ├── most_common.py
│   │   └── most_common_decomp.py
```

### `binary/`

The `binary/` folder contains definitions for classes, including types, variables, and functions, constructed from decompiler outputs from binaries.

```
├── binary
│   ├── __init__.py     
│   ├── dire_types.py   # constructing types and a type library
│   ├── function.py     # definition and serialization for function instances
│   ├── ida_ast.py      # constructing ASTs from IDA-Pro outputs
│   └── variable.py     # definition and serialization for variable instances
```

### `idastubs/`

The `idastubs/` folder contains helper functions used by the `ida_ast.py` file.

### `dataset-gen/`

The `dataset-gen/` folder contains producing unpreprocessed data from binaries using IDA-Pro (required).

### `dire/`

Legacy code for the DIRE paper.

## Citing DIRTY

If you use DIRTY/DIRT in your research or wish to refer to the baseline results, please use the following BibTeX.

```
Not available yet.
```
