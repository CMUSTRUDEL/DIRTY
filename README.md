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


# Development and Requirements

This repository contains three packages.

- `csvnom-utils` is CMU-STRUDEL Variable Name Prediction Model Utilities
- `dire` is Legacy code for the dire project
- `dirty` is the type and variable prediction model

### Requires
- Linux with Python ≥ 3.6
- [PyTorch ≥ 1.5.1](https://pytorch.org/)

### Development

This code leverages `pre-commit` hooks to mantain strict coding standards, and `tox` to automate testing.

```
$ pip install tox
$ pip install pre-commit
$ pre-commit init
```

A `tox` is run from the package level, and creates the appropriate testing environments. Tests are currently empty, but coverage is displayed as part of the tests.
