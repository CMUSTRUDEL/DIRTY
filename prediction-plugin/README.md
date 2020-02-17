# Prediction Plugin

This code loads a provided model and generates predictions from Hex-Rays ASTs.

## Python Environment

To install and activate the conda environment:

```
conda env create -f env.yml
conda activate dire_prediction
```

You can also setup the environment using `pip`:

```
pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html
```

Note: If you are using virtual environments with python3, you should
use `venv` instead of `virtualenv` to avoid (this
bug)[https://github.com/pypa/virtualenv/issues/737].

## Hex-Rays Python Environment

Install `jsonlines` so that the IDApython script can see it:
```
sudo pip install jsonlines
```

## Download the pretrained models

```
wget -O pretrained_model.tar.gz https://www.dropbox.com/s/2b4c9ba2g0nhe7q/pretrained_model.tar.gz\?dl\=1
tar -xzf pretrained_model.tar.gz
```

## Use Hex-Rays script/plugin

### Plugin

To install the script as a plugin, create a symbolic link to
`prediction-plugin/decompiler/decompiler-scripts/predict_names.py`
from your Hex-Rays `plugins` directory.  For example:

```bash
ln -s /path/to/prediction-plugin/decompiler/decompiler-scripts/predict_names.py /path/to/idapro-7.3/plugins/predict_names.py
```

You only need to do this once.

### Script

To load the script without installing it as a plugin, open Hex-Rays,
select "Script file" from the "File" menu, and then select
`prediction-plugin/decompiler/decompiler-scripts/predict_names.py`.

### Usage

Once installed as plugin or loaded as a script, open a Pseudocode
window for any function (e.g., by pressing F5) inside Hex-Rays.  Use
the "Predict variable names" action that is available from the
right-click context menu of the Pseudocode window.

Note that Hex-Rays must always be loaded from the `dire_prediction`
Conda environment.

# Running the scripts manually

For debugging or development purposes, you may also wish to run the
prediction plugin through scripts.  Before starting, you must first
follow the directions above to [setup the `dire_prediction` conda
environment](#conda-environment) and [download the pretrained
models](#download-the-pretrained-models).

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
