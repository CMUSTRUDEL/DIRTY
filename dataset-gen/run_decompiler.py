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
import tempfile

statyre_dir = os.path.dirname(os.path.abspath(__file__))
COLLECT = os.path.join(statyre_dir, 'decompiler_scripts', 'collect.py')
DUMP_TREES = os.path.join(statyre_dir, 'decompiler_scripts', 'dump_trees.py')

parser = argparse.ArgumentParser(
    description="Run the decompiler to generate a corpus.")
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
parser.add_argument('--verbose', '-v', action='store_true')

args = parser.parse_args()
env = os.environ.copy()
env['IDALOG'] = '/dev/stdout'

# Check for/create output directories
output_dir = os.path.abspath(args.output_dir)
env['OUTPUT_DIR'] = output_dir
env['TYPE_DBASE'] = os.path.abspath(args.type_dbase)

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
    if args.verbose:
        print(output.decode('unicode_escape'))


class NoVarsError(Exception):
    """Exception raised when no variables are defined"""
    pass


class Timer:
    def __init__(self, num_files):
        self.total_time = datetime.timedelta(0)
        self.runs = 0
        self.num_files = num_files

    def average_time(self):
        if self.runs == 0:
            return datetime.timedelta(0)
        return self.total_time / self.runs

    def remaining_time(self):
        return self.average_time() * (self.num_files - self.runs)

    def update(self, start, end):
        self.runs += 1
        duration = end - start
        self.total_time = self.total_time + duration
        finish_time = datetime.datetime.now() + self.remaining_time()
        print(f"Duration: {duration}\n"
              f"Total time: {self.total_time}\n"
              f"Average: {self.average_time()}\n"
              f"Remaining: {self.remaining_time()}\n"
              f"Projected finish time: {finish_time}\n")


class TimedRun:
    def __init__(self, timer):
        self.start_time = datetime.datetime.now()
        self.timer = timer
        print(f"File {timer.runs + 1} of {timer.num_files}")
        print(f"Started: {self.start_time}")

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        if type is KeyboardInterrupt:
            return False
        if type is subprocess.TimeoutExpired:
            print("Timed out")
        if type in (pickle.UnpicklingError, NoVarsError):
            print("No variables collected")
        timer.update(self.start_time, datetime.datetime.now())
        return True


# Create a temporary directory, since the decompiler makes a lot of additional
# files that we can't clean up from here
with tempfile.TemporaryDirectory() as tempdir:
    tempfile.tempdir = tempdir

    # File counts for progress output
    num_files = sum(1 for x in os.listdir(args.binaries_dir)
                    if os.path.isfile(os.path.join(args.binaries_dir, x)))
    timer = Timer(num_files)

    for binary in os.listdir(args.binaries_dir):
        env['PREFIX'] = binary
        file_path = os.path.join(args.binaries_dir, binary)
        print(f"Collecting from {file_path}")
        with tempfile.NamedTemporaryFile() as collected_vars, \
             tempfile.NamedTemporaryFile() as fun_locals, \
             TimedRun(timer):
            # First collect variables
            env['COLLECTED_VARS'] = collected_vars.name
            env['FUN_LOCALS'] = fun_locals.name
            with tempfile.NamedTemporaryFile() as orig:
                subprocess.check_output(['cp', file_path, orig.name])
                # Timeout after 30 seconds for first run
                run_decompiler(orig.name, env, COLLECT, timeout=30)
                if not pickle.load(collected_vars):
                    raise NoVarsError
            # Make a new stripped copy and pass it the collected vars
            with tempfile.NamedTemporaryFile() as stripped:
                subprocess.call(['cp', file_path, stripped.name])
                subprocess.call(['strip', '--strip-debug', stripped.name])
                print(f"{binary} stripped")
                # Dump the trees.
                # No timeout here, we know it'll run in a reasonable amount of
                # time and don't want mismatched files
                run_decompiler(stripped.name, env, DUMP_TREES)
