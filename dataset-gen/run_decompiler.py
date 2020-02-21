# Runs the decompiler to collect variable names from binaries containing
# debugging information, then strips the binaries and injects the collected
# names into that decompilation output.
# This generates an aligned, parallel corpus for training translation models.

# Requires Python 3

import argparse
import datetime
import errno
import pickle
import os
import subprocess
import sys
import tempfile

statyre_dir = os.path.dirname(os.path.abspath(__file__))
COLLECT = os.path.join(statyre_dir, 'decompiler_scripts', 'collect.py')
DUMP_TREES = os.path.join(statyre_dir, 'decompiler_scripts', 'dump_trees.py')

parser = argparse.ArgumentParser(description="Run the decompiler to generate a corpus.")
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

# Use RAM-backed memory for tmp if available
if os.path.exists('/dev/shm'):
    tempfile.tempdir = '/dev/shm'

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
        print(f"Collecting from {file_path}")
        with tempfile.NamedTemporaryFile() as collected_vars:
            # First collect variables
            env['COLLECTED_VARS'] = collected_vars.name
            with tempfile.NamedTemporaryFile() as orig:
                subprocess.check_output(['cp', file_path, orig.name])
                # Timeout after 30 seconds for first run
                try:
                    run_decompiler(orig.name, env, COLLECT, timeout=30)
                except subprocess.TimeoutExpired:
                    print("Timed out\n")
                    continue
                try:
                    if not pickle.load(collected_vars):
                        print("No variables collected\n")
                        continue
                except:
                    print("No variables collected\n")
                    continue
            # Make a new stripped copy and pass it the collected vars
            with tempfile.NamedTemporaryFile() as stripped:
                subprocess.call(['cp', file_path, stripped.name])
                subprocess.call(['strip', '--strip-debug', stripped.name])
                print(f"{binary} stripped")
                # Dump the trees.
                # No timeout here, we know it'll run in a reasonable amount of
                # time and don't want mismatched files
                run_decompiler(stripped.name, env, DUMP_TREES)
        end = datetime.datetime.now()
        duration = end-start
        print(f"Duration: {duration}\n")
