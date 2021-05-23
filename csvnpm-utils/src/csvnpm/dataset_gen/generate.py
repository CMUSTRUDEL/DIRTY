# Runs the decompiler to collect variable names from binaries containing
# debugging information, then strips the binaries and injects the collected
# names into that decompilation output.
# This generates an aligned, parallel corpus for training translation models.

# Requires Python 3

import argparse
import errno
import hashlib
import os
import subprocess
import tempfile as tf
from multiprocessing import Pool
from typing import Iterable, Tuple

from tqdm import tqdm  # type: ignore

# from runner import Runner

# COLLECT = os.path.join(dire_dir, "decompiler", "debug.py")
# DUMP_TREES = os.path.join(dire_dir, "decompiler", "dump_trees.py")


class Runner:
    dire_dir = os.path.dirname(os.path.abspath(__file__))
    COLLECT = os.path.join(dire_dir, "decompiler", "debug.py")
    DUMP_TREES = os.path.join(dire_dir, "decompiler", "dump_trees.py")

    def __init__(self, args: argparse.Namespace):
        self.ida = args.ida
        self.binaries_dir = args.binaries_dir
        self.output_dir = args.output_dir
        self._num_files = args.num_files
        self.verbose = args.verbose
        self.num_threads = args.num_threads

        self.env = os.environ.copy()
        self.env["IDALOG"] = "/dev/stdout"
        self.env["OUTPUT_DIR"] = self.output_dir

        self.make_dir(self.output_dir)
        self.make_dir(os.path.join(self.output_dir, "types"))
        self.make_dir(os.path.join(self.output_dir, "bins"))

        # Use RAM-backed memory for tmp if available
        if os.path.exists("/dev/shm"):
            tf.tempdir = "/dev/shm"
        try:
            self.run()
        except KeyboardInterrupt:
            pass

    @property
    def binaries(self) -> Iterable[Tuple[str, str]]:
        """Readable 64-bit ELFs in the binaries_dir and their paths"""

        def is_elf64(root: str, path: str) -> bool:
            file_path = os.path.join(root, path)
            try:
                with open(file_path, "rb") as f:
                    header = f.read(5)
                    # '\x7fELF' means it's an ELF file, '\x02' means 64-bit
                    return header == b"\x7fELF\x02"
            except IOError:
                return False

        return (
            (root, f)
            for root, _, files in os.walk(self.binaries_dir)
            for f in files
            if is_elf64(root, f)
        )

    @property
    def num_files(self) -> int:
        """The number of files in the binaries directory. Note that this is not
        the total number of binaries because it does not check file headers. The
        number of binary files could be lower."""
        if self._num_files is None:
            self._num_files = 0
            for _, _, files in os.walk(self.binaries_dir):
                self._num_files += len(files)
        return self._num_files

    @staticmethod
    def make_dir(dir_path):
        """Make a directory, with clean error messages."""

        try:
            os.makedirs(dir_path)
        except OSError as e:
            if not os.path.isdir(dir_path):
                raise NotADirectoryError(f"'{dir_path}' is not a directory")
            if e.errno != errno.EEXIST:
                raise

    def run_decompiler(self, env, file_name, script, timeout=None):
        """Run a decompiler script.

        Keyword arguments:
        file_name -- the binary to be decompiled
        env -- an os.environ mapping, useful for passing arguments
        script -- the script file to run
        timeout -- timeout in seconds (default no timeout)
        """
        idacall = [self.ida, "-B", f"-S{script}", file_name]
        output = ""
        try:
            output = subprocess.check_output(idacall, env=env, timeout=timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            output = e.output
            subprocess.call(["rm", "-f", f"{file_name}.i64"])
        if self.verbose:
            print(output.decode("unicode_escape"))

    def run_one(self, args: Tuple[str, str]) -> None:
        path, binary = args
        new_env = self.env.copy()
        with tf.TemporaryDirectory() as tempdir:
            with tf.NamedTemporaryFile(dir=tempdir) as functions, tf.NamedTemporaryFile(
                dir=tempdir
            ) as orig, tf.NamedTemporaryFile(dir=tempdir) as stripped:
                file_path = os.path.join(path, binary)
                new_env["FUNCTIONS"] = functions.name
                # Build up hash string in 4k blocks
                file_hash = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        file_hash.update(byte_block)
                prefix = f"{file_hash.hexdigest()}_{binary}"
                new_env["PREFIX"] = prefix
                # Try stripping first, if it fails return
                subprocess.call(["cp", file_path, stripped.name])
                try:
                    subprocess.call(["strip", "--strip-debug", stripped.name])
                except subprocess.CalledProcessError:
                    if self.verbose:
                        print(f"Could not strip {prefix}, skipping.")
                    return
                if os.path.exists(
                    os.path.join(self.output_dir, "bins", prefix + ".jsonl.gz")
                ):
                    if self.verbose:
                        print(f"{prefix} already collected, skipping")
                    return
                if os.path.exists(
                    os.path.join(self.output_dir, "types", prefix + ".jsonl.gz")
                ):
                    if self.verbose:
                        print(f"{prefix} types already collected, skipping")
                else:
                    # Collect from original
                    subprocess.check_output(["cp", file_path, orig.name])
                    # Timeout after 30s for the collect run
                    self.run_decompiler(new_env, orig.name, self.COLLECT, timeout=30)
                # Dump trees
                self.run_decompiler(
                    new_env, stripped.name, self.DUMP_TREES, timeout=120
                )

    def run(self):
        # File counts for progress output

        # Create a temporary directory, since the decompiler makes a lot of
        # additional files that we can't clean up from here
        with Pool(self.num_threads) as pool:
            for p in tqdm(
                pool.imap_unordered(self.run_one, self.binaries),
                total=self.num_files,
                leave=True,
                dynamic_ncols=True,
                unit="bin",
                smoothing=0.1,
            ):
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Run the decompiler to generate a corpus."
    )
    parser.add_argument(
        "--ida",
        metavar="IDA",
        help="location of the idat64 binary",
        default="/home/jlacomis/bin/ida/idat64",
    )
    parser.add_argument(
        "-t",
        "--num-threads",
        metavar="N",
        help="number of threads to use",
        default=4,
        type=int,
    )
    parser.add_argument(
        "-n",
        "--num-files",
        metavar="N",
        help="number of binary files",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--binaries_dir",
        metavar="BINARIES_DIR",
        help="directory containing binaries",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        metavar="OUTPUT_DIR",
        help="output directory",
        required=True,
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    Runner(args)


if __name__ == "__main__":
    main()
