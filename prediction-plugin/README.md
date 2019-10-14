# Prediction Plugin

This code loads a provided model and generates predictions from Hex-Rays ASTs.

## Conda Environment

To install and activate the conda environment:

```
conda env create -f env.yml
conda activate dire_prediction
```

## Download the pretrained models

```
wget -O pretrained_model.tar.gz https://www.dropbox.com/s/2b4c9ba2g0nhe7q/pretrained_model.tar.gz\?dl\=1
tar -xf pretrained_model.tar.gz
```

## Create a working directory

I start by creating a working directory with subfolders for each step of the
process and copying the binary/binaries to a dedicated folder:

```bash
mkdir -p workdir/{binaries,trees,preprocessed}
cp /path/to/some/binary workdir/binaries
```

## Collect trees from target binary

Generate the trees using the decompiler, and use `tar` to create an input file (for
compatibility with the preprocessing script).

```bash
python decompiler/run_decompiler.py \
    --ida /path/to/idat64 \
    workdir/binaries \
    workdir/trees
tar -cf workdir/trees.tar -C workdir trees
```

See `README.md` in the `decompiler` subdirectory for more details.

## Pre-process collected tress

Preprocessing takes the output of the decompiler dump and converts it into a
format expected by the neural model. It also removes ASTs with >300 nodes and
functions without any variables to rename.

```bash
python -m utils.preprocess workdir/trees.tar workdir/preprocessed
```

## Running DIRE

`exp.py` is the entry script for the DIRE model.
To predict using the  pretrained model, run the following command.

 ```bash
python exp.py \
    --cuda \
    data/saved_models/model.hybrid.bin \
    workdir/preprocessed/preprocessed.tar
```
