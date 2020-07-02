# Runs the decompiler to collect variable names from binaries containing
# debugging information, then strips the binaries and injects the collected
# names into that decompilation output.
# This generates an aligned, parallel corpus for training translation models.

# Requires Python 3

import argparse
import os

from runner import Runner

dire_dir = os.path.dirname(os.path.abspath(__file__))
COLLECT = os.path.join(dire_dir, 'decompiler', 'debug.py')
DUMP_TREES = os.path.join(dire_dir, 'decompiler', 'dump_trees.py')

parser = argparse.ArgumentParser(
    description="Run the decompiler to generate a corpus.")
parser.add_argument('--ida',
                    metavar='IDA',
                    help="location of the idat64 binary",
                    default='/home/jlacomis/bin/ida/idat64',
                    )
parser.add_argument('--type-lib',
                    metavar='LIB',
                    help="name of type library",
                    default='types',
                    )
parser.add_argument('binaries_dir',
                    metavar='BINARIES_DIR',
                    help="directory containing binaries",
                    )
parser.add_argument('output_dir',
                    metavar='OUTPUT_DIR',
                    help="output directory",
                    )
parser.add_argument('--verbose', '-v', action='store_true')


args = parser.parse_args()
Runner(args.ida, args.type_lib, args.binaries_dir, args.output_dir, args.verbose, COLLECT, DUMP_TREES)
