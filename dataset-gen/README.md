Building A Corpus
=================

This directory contains the files needed to generate a corpus from a set of
binaries containing debug information (i.e., compiled with `gcc -g`).

Prerequisites
=============

The current implementation takes the path to a directory of x86_64 binary
files. The directory structure must be flat, and the binaries must be uniquely
named: their names will be used as a prefix to the output files. My current
strategy was to simply rename all of the binaries that I collected to their
SHA-256 hash. I did this in my initial approach to filter out identical
binaries, and kept them organized this way. I further broke the binaries down
into sixteen directories, each holding all files that began with a particular
hexadecimal digit. This made them much easier to work with.

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

How It Works
============

For each binary, there are two stages of decompilation. First, each binary is
decompiled and the decompiler is allowed to use DWARF debugging information to
generate pseudocode using the original, human-written variable names. This stage
collects information about the addresses in the binary corresponding to each
variable. This information is passed to the second stage, which decompiles the
same binary with its debugging information stripped, and maps the original
variable names to variables that correspond to the same addresses.

Output Format
=============

Each line in the output is a JSON value corresponding to a function in the
binary. At the moment there are three fields:
* `function`: The name of the function.
* `raw_code`: The pseudocode output by the decompiler, with placeholders for
variable names. Variable names are replaced with a token of the format
`@@VAR_[id]@@[old_name]@@[new_name]`, where `[id]` identifies all positions of
the same variable, `[old_name]` is the name assigned by the decompiler when it
does not have debugging information, and `[new_name]` is the name mined from the
debugging information contained in the binary.
* `ast` holds a JSON-serialized representation of the internal Hex-Rays AST.
