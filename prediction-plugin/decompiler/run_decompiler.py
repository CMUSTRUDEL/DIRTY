# Runs the decompiler to dump ASTs for a binary
# Requires Python 3

import argparse
import datetime
import errno
import pickle
import os
import subprocess
import sys
import tempfile

dire_dir = os.path.dirname(os.path.abspath(__file__))
DUMP_TREES = os.path.join(dire_dir, 'decompiler_scripts', 'dump_trees.py')

parser = argparse.ArgumentParser(description="Run the decompiler and dump AST.")
parser.add_argument('--ida',
                    metavar='IDA',
                    help="location of the idat64 binary",
                    default='/home/jlacomis/bin/ida/idat64',
)
parser.add_argument('binaries_dir',
                    metavar='BINARIES_DIR',
                    help="directory containing binaries",
)
parser.add_argument('output_dir',
                    metavar='OUTPUT_DIR',
                    help="output directory",
)

args = parser.parse_args()
env = os.environ.copy()
env['IDALOG'] = '/dev/stdout'

# Check for/create output directories
output_dir = os.path.abspath(args.output_dir)
env['OUTPUT_DIR'] = output_dir

def make_directory(dir_path):
    """Make a directory, with clean error messages."""
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"'{dir_path}' is not a directory")
        if e.errno != errno.EEXIST:
            raise

make_directory(output_dir)

def run_decompiler(file_name, env, script, timeout=None):
    """Run a decompiler script.

    Keyword arguments:
    file_name -- the binary to be decompiled
    env -- an os.environ mapping, useful for passing arguments
    script -- the script file to run
    timeout -- timeout in seconds (default no timeout)
    """
    idacall = [args.ida, '-B', f'-S{script}', file_name]
    output = ''
    try:
        output = subprocess.check_output(idacall, env=env, timeout=timeout)
    except subprocess.CalledProcessError as e:
        output = e.output
        subprocess.call(['rm', '-f', f'{file_name}.i64'])
    return output

# Create a temporary directory, since the decompiler makes a lot of additional
# files that we can't clean up from here
with tempfile.TemporaryDirectory() as tempdir:
    tempfile.tempdir = tempdir

    # File counts for progress output
    num_files = sum(1 for x in os.listdir(args.binaries_dir) \
                    if os.path.isfile(os.path.join(args.binaries_dir, x)))
    file_count = 1

    for binary in os.listdir(args.binaries_dir):
        print(f"File {file_count} of {num_files}")
        file_count += 1
        start = datetime.datetime.now()
        print(f"Started: {start}")
        env['PREFIX'] = binary
        file_path = os.path.join(args.binaries_dir, binary)

        with tempfile.NamedTemporaryFile() as orig:
            subprocess.check_output(['cp', file_path, orig.name])
            # Timeout after 10 minutes
            try:
                output = run_decompiler(orig.name, env, DUMP_TREES, timeout=600)
                if not os.path.isfile(os.path.join(args.output_dir, binary+".jsonl")):
                    print("Something bad happened: ", output)
                    assert False
            except subprocess.TimeoutExpired:
                print("Timed out\n")
                continue

        end = datetime.datetime.now()
        duration = end-start
        print(f"Duration: {duration}\n")
