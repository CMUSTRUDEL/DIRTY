import datetime
import errno
import pickle
import os
import subprocess
import tempfile


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
    def __init__(self, timer, env):
        self.timer = timer
        self.env = env
        self.collected_vars = None
        self.fun_locals = None
        self.orig = None
        self.stripped = None

    def __enter__(self):
        self.start_time = datetime.datetime.now()
        self.collected_vars = tempfile.NamedTemporaryFile()
        self.fun_locals = tempfile.NamedTemporaryFile()
        self.orig = tempfile.NamedTemporaryFile()
        self.stripped = tempfile.NamedTemporaryFile()
        self.env['COLLECTED_VARS'] = self.collected_vars.name
        self.env['FUN_LOCALS'] = self.fun_locals.name
        print(f"File {self.timer.runs + 1} of {self.timer.num_files}")
        print(f"Started at {self.start_time}")
        return self

    def __exit__(self, type, value, traceback):
        self.collected_vars.close()
        self.fun_locals.close()
        self.orig.close()
        self.stripped.close()
        if type is KeyboardInterrupt:
            return False
        elif type is subprocess.TimeoutExpired:
            print("Timed out")
        elif type in (pickle.UnpicklingError, NoVarsError):
            print("No variables collected")
        elif type is not None:
            print(f"{Type}: value")
            return False
        self.timer.update(self.start_time, datetime.datetime.now())
        return True


class Runner:
    def __init__(self, ida, binaries_dir, output_dir, verbose, COLLECT, DUMP_TREES):
        self.ida = ida
        self.binaries_dir = binaries_dir
        self.output_dir = output_dir
        self.verbose = verbose
        self.COLLECT = COLLECT
        self.DUMP_TREES = DUMP_TREES

        self.env = os.environ.copy()
        self.env['IDALOG'] = '/dev/stdout'
        self.env['OUTPUT_DIR'] = self.output_dir

        self.binaries = os.listdir(binaries_dir)

        self.num_files = \
            len([name for name in os.listdir(self.binaries_dir)
                 if os.path.isfile(os.path.join(self.binaries_dir, name))])

        self.make_dir(output_dir)

        # Use RAM-backed memory for tmp if available
        if os.path.exists('/dev/shm'):
            tempfile.tempdir = '/dev/shm'
        self.run()

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

    def run_decompiler(self, file_name, script, timeout=None):
        """Run a decompiler script.

        Keyword arguments:
        file_name -- the binary to be decompiled
        env -- an os.environ mapping, useful for passing arguments
        script -- the script file to run
        timeout -- timeout in seconds (default no timeout)
        """
        idacall = [self.ida, '-B', f'-S{script}', file_name]
        output = ''
        try:
            output = subprocess.check_output(idacall, env=self.env, timeout=timeout)
        except subprocess.CalledProcessError as e:
            output = e.output
            subprocess.call(['rm', '-f', f'{file_name}.i64'])
        if self.verbose:
            print(output.decode('unicode_escape'))

    def run(self):
        # File counts for progress output
        timer = Timer(self.num_files)
        # Create a temporary directory, since the decompiler makes a lot of
        # additional files that we can't clean up from here
        with tempfile.TemporaryDirectory() as tempdir:
            tempfile.tempdir = tempdir
            for binary in self.binaries:
                self.env['PREFIX'] = binary
                file_path = os.path.join(self.binaries_dir, binary)
                print(f"Collecting from {file_path}")

                with TimedRun(timer, self.env) as r:
                    # Collect from original
                    subprocess.check_output(['cp', file_path, r.orig.name])
                    # Timeout after 30s for the collect run
                    self.run_decompiler(r.orig.name, self.COLLECT, timeout=30)
                    if not pickle.load(r.collected_vars):
                        raise NoVarsError

                    # Dump trees
                    subprocess.call(['cp', file_path, r.stripped.name])
                    subprocess.call(['strip', '--strip-debug', r.stripped.name])
                    self.run_decompiler(r.stripped.name, self.DUMP_TREES)
