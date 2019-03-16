from docopt import docopt
import os

from utils.dataset import Example, Dataset


def main():
    args = dict()
    tgt_folder = args['--data_folder']
    tar_files = args['--tar-files']

    os.system('mkdir -p {tgt_folder}')

    for tar_file in tar_files.split(','):
        print(f'read {tar_file}')
        tar_dataset = Dataset(tar_file)
        for example in tar_dataset.get_iterator(progress=True, num_workers=5):

