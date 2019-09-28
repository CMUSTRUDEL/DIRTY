# Neural Variable Renaming

This repository contains the neural variable renaming model `DIRE` from our ASE 2019 paper *DIRE: A Neural Approach to Decompiled Identifier Renaming*.

## Conda Environment

First, download all supporting files:

```
wget http://www.cs.cmu.edu/~pengchey/dire_models.zip
unzip dire_models.zip
```

To install and activate the conda environment:

```
conda env install -f data/env.yml
conda activate var_rename
```

## Dataset and Preprocessing

We created a corpus of 164,632 unique x86-64 binaries generated from C projects mined from GitHub. Each binary is decompiled by `hexray`. To download the full dataset, please visit [here](https://doi.org/10.5281/zenodo.3403077).

### Pre-process the Github Binaries Dataset for DIRE

To train and test DIRE model using the collected binaries dataset, first run the following pre-process script `utils.preprocess` to (1) filter invalid examples (e.g., code with too-large ASTs), and (2) randomly partition the entire dataset into training/development/test sets:

```bash
python -m utils.preprocess \
    "path/to/binary/dataset/*.tar.gz" \   # use wild-card to match all tar files
    data/preprocessed_data
```

All scripts are documented using [`docopt`](http://docopt.org/), please refer to the docstring of `utils/preprocess.py` for its complete usage.

You may also download our pre-processed dataset along with the training/testing partitions from [here](mailto:pcyin@cs.cmu.edu). The pre-processing scripts also support fixing the testing set to be a pre-defined partition. For example, to use the same testing partition as the one used in our paper during pre-processing, you may run:

```bash
python -m utils.preprocess \
    --no-filtering \                                # optional: do not perform filtering 
    --test-file=path/to/predefined/test_file.tar \
    "path/to/binary/dataset/*.tar.gz" \
    data/preprocessed_data
```

Next, to create the vocabulary files:

```bash
python -m utils.vocab \
    --use-bpe \
    --size=10000 \
    "data/preprocessed_data/train-shard-*.tar" \
    data/preprocessed_data/vocab.bpe10000
```

Again, please refer to the script file's docstring for its complete usage. We've also included the vocabulary file in the release (under `data/vocab.bpe10000`).

## Running DIRE

`exp.py` is the entry script for training and evaluating the DIRE model. Below is an example training script:

```bash
python exp.py \
    train \
    --cuda \
    --work-dir=path/to/the/work/folder \
    --extra-config='{ "data": {"train_file": "data/preprocessed_data/train-shard-*.tar" }, "decoder": { "input_feed": false, "tie_embedding": true }, "train": { "evaluate_every_nepoch": 5, "max_epoch": 60 } }' \
    data/config/model.hybrid.jsonnet
```

`DIRE` uses [`json.net`]() for programmable configuration. Extra configs could be specified using the `--extra-config` argument.

To evaluate a saved or pretrained model, run the following command. 
 
 ```bash
python exp.py \
    test \
    --cuda \
    --extra-config='{"decoder": {"remove_duplicates_in_prediction": true} }' \
    path/to/work/folder/model.iter_number.bin \
    data/preprocessed_data/test.tar
```

### Pretrained Models

We also provide pre-trained DIRE models used in our paper, located under `data/saved_models/`.

## Reference

```
@inproceedings{lacomis19ase,
    title = {{DIRE}: A Neural Approach to Decompiled Identifier Renaming},
    author = {Jeremy Lacomis and Pengcheng Yin and Edward J. Schwartz and Miltiadis Allamanis and Claire Le Goues and Graham Neubig and Bogdan Vasilescu},
    booktitle = {34th IEEE/ACM International Conference on Automated Software Engineering (ASE)},
    address = {San Diego, California},
    month = {November},
    year = {2019}
}
```