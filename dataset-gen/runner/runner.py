import datetime
import errno
import pickle
import os
import subprocess
import tempfile
from tqdm import tqdm


class Timer:
    def __init__(self, num_files):
        self._binary = ''
        self.total_time = datetime.timedelta(0)
        self.runs = 0
        self.num_files = num_files
        self.file_log = tqdm(total=0, bar_format='{desc}')
        self.file_log.set_description_str(f'Current file:')
        self.progress_log = tqdm(total=num_files, unit='file')
        self.last_log = tqdm(total=0, bar_format='{desc}')
        self.total_time_log = tqdm(total=0, bar_format='{desc}')
        self.average_time_log = tqdm(total=0, bar_format='{desc}')
        self.remaining_time_log = tqdm(total=0, bar_format='{desc}')
        self.projected_log = tqdm(total=0, bar_format='{desc}')
        self._message = tqdm(total=0, bar_format='{desc}')
        self._message.set_description_str('Message:')
        self.update_logs()

    @property
    def binary(self):
        return self._binary

    @binary.setter
    def binary(self, new_binary):
        self._binary = new_binary
        self.file_log.set_description_str(f'Current file: {self._binary}')

    def average_time(self):
        if self.runs == 0:
            return datetime.timedelta(0)
        return self.total_time / self.runs

    def remaining_time(self):
        return self.average_time() * (self.num_files - self.runs)

    def update_logs(self, last=''):
        self.progress_log.update(1)
        self.last_log.set_description_str(f'Last file time: {last}')
        self.total_time_log.set_description_str(
            f'Total time: {self.total_time}')
        self.average_time_log.set_description_str(
            f'Average time: {self.average_time()}')
        self.remaining_time_log.set_description_str(
            f'Remaining time: {self.remaining_time()}')
        if self.runs == 0:
            finish_time = ''
        else:
            finish_time = datetime.datetime.now() + self.remaining_time()
        self.projected_log.set_description_str(
            f'Projected finish time: {finish_time}')

    def message(self, msg):
        self._message.set_description_str(f'Message: {msg}')

    def update(self, start, end):
        self.runs += 1
        duration = end - start
        self.total_time = self.total_time + duration
        finish_time = datetime.datetime.now() + self.remaining_time()
        self.update_logs(duration)


class TimedRun:
    def __init__(self, timer, binary, env):
        self.timer = timer
        timer.binary = binary
        self.env = env
        self.functions = None
        self.orig = None
        self.stripped = None

    def __enter__(self):
        self.start_time = datetime.datetime.now()
        self.functions = tempfile.NamedTemporaryFile()
        self.orig = tempfile.NamedTemporaryFile()
        self.stripped = tempfile.NamedTemporaryFile()
        self.env['FUNCTIONS'] = self.functions.name
        return self

    def __exit__(self, type, value, traceback):
        self.functions.close()
        self.orig.close()
        self.stripped.close()
        if type is KeyboardInterrupt:
            return False
        elif type is subprocess.TimeoutExpired:
            self.timer.message("Timed out")
        elif type is pickle.UnpicklingError:
            self.timer.message("Unpickling error")
        elif type is not None:
            self.timer.message(f"{Type}: value")
            return False
        self.timer.update(self.start_time, datetime.datetime.now())
        return True


class Runner:
    def __init__(self, ida, type_lib, binaries_dir, output_dir, verbose, COLLECT, DUMP_TREES):
        self.ida = ida
        self.type_lib = type_lib
        self.binaries_dir = binaries_dir
        self.output_dir = output_dir
        self.verbose = verbose
        self.COLLECT = COLLECT
        self.DUMP_TREES = DUMP_TREES

        self.env = os.environ.copy()
        self.env['IDALOG'] = '/dev/stdout'
        self.env['OUTPUT_DIR'] = self.output_dir
        self.env['TYPE_LIB'] = self.type_lib

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
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(e)
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

                with TimedRun(timer, binary, self.env) as r:
                    # Collect from original
                    subprocess.check_output(['cp', file_path, r.orig.name])
                    # Timeout after 30s for the collect run
                    self.run_decompiler(r.orig.name, self.COLLECT, timeout=10)
                    # Dump trees
                    subprocess.call(['cp', file_path, r.stripped.name])
                    subprocess.call(['strip', '--strip-debug', r.stripped.name])
                    self.run_decompiler(r.stripped.name, self.DUMP_TREES, timeout=10)
