Dumping ASTs
=================

This directory contains the files needed to generate ASTs from binaries.

Prerequisites
=============

The current implementation takes the path to a directory of x86_64 binary
files. The directory structure must be flat, and the binaries must be uniquely
named: their names will be used as a prefix to the output files.

A copy of Hex-Rays (and, implicitly, IDA Pro) is also required.

Use
===

Use is fairly simple, given a directory of binaries and an existing output
directory, just run the [run_decompiler.py](run_decompiler.py) script with
Python 3:
`python3 run_decompiler.py --ida /path/to/idat64 BINARIES_DIR OUTPUT_DIR`

This generates a `.jsonl` file for each binary in `BINARIES_DIR`. The file is in
the [JSON Lines](http://jsonlines.org) format, and each entry corresponds to a
function in the binary.

Output Format
=============

The output format is the same as format for the training corpus for tool
compatibility. Since there are no new names, the old name is just used twice
instead.

Each line in the output is a JSON value corresponding to a function in the
binary. At the moment there are three fields:
* `function`: The name of the function.
* `raw_code`: The pseudocode output by the decompiler, with placeholders for
variable names. Variable names are replaced with a token of the format
`@@VAR_[id]@@[old_name]@@[old_name]`, where `[id]` identifies all positions of
the same variable, `[old_name]` is the name assigned by the decompiler.
* `ast` holds a JSON-serialized representation of the internal Hex-Rays AST.
