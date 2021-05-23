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
conda env create -f data/env.yml
conda activate var_rename
```

## Dataset and Preprocessing

We created a corpus of 164,632 unique x86-64 binaries generated from C projects mined from GitHub. Each binary is decompiled by `hexray`. To download the full dataset, please visit [here](https://doi.org/10.5281/zenodo.3403077).

### Pre-process the Github Binaries Dataset for DIRE

**Clearning Binary Data** To train and test DIRE model using the collected binaries dataset, first run the following pre-process script `utils.preprocess` to (1) filter invalid examples (e.g., code with too-large ASTs), and (2) randomly partition the entire dataset into training/development/test sets:

```bash
mkdir -p data/preprocessed_data

python -m utils.preprocess \
    "path/to/binary/dataset/*.tar.gz" \   # use wild-card to match all tar files
    data/preprocessed_data
```

All scripts are documented using [`docopt`](http://docopt.org/), please refer to the docstring of `utils/preprocess.py` for its complete usage.

**Our Preprocessed Splits** You may also download our pre-processed dataset along with the training/testing splits from [here](https://drive.google.com/drive/folders/19Rf7NtW56r6fz-ycldZq9hjxNr5osAJW?usp=sharing). The pre-processing scripts also support fixing the testing set to be a pre-defined partition. For example, to use the same testing partition as the one used in our paper during pre-processing, you may run:

```bash
python -m utils.preprocess \
    --no-filtering \                                # optional: do not perform filtering
    --test-file=path/to/predefined/test_file.tar \
    "path/to/binary/dataset/*.tar.gz" \
    data/preprocessed_data
```

**Vocabulary Files** We've included the vocabulary file in the release (under `data/vocab.bpe10000`). If you would like to create your own vocabulary (e.g., to try a different BPE vocabulary size), simply run:

```bash
python -m utils.vocab \
    --use-bpe \
    --size=10000 \
    "data/preprocessed_data/train-shard-*.tar" \
    data/vocab.bpe10000
```

Again, please refer to the script file's docstring for its complete usage.

## Running DIRE

`exp.py` is the entry script for training and evaluating the DIRE model. Below is an example training script:

```bash
mkdir -p exp_runs/dire.hybrid   # create a work directory

python exp.py \
    train \
    --cuda \
    --work-dir=exp_runs/dire.hybrid \
    --extra-config='{ "data": {"train_file": "data/preprocessed_data/train-shard-*.tar" }, "decoder": { "input_feed": false, "tie_embedding": true }, "train": { "evaluate_every_nepoch": 5, "max_epoch": 60 } }' \
    data/config/model.hybrid.jsonnet
```

`DIRE` uses [`json.net`]() for programmable configuration. Extra configs could be specified using the `--extra-config` argument.

To evaluate a saved or pretrained model, run the following command.

 ```bash
python exp.py \
    test \
    --cuda \
    --extra-config='{"data": {"vocab_file": "data/vocab.bpe10000/vocab"}, "decoder": {"remove_duplicates_in_prediction": true} }' \
    data/saved_models/model.hybrid.bin \   # path to the pretrained models at `data/saved_models` or the saved model under the user-specified work directory
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
