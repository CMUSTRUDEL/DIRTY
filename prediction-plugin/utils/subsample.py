import os, sys
import glob
from sh import tar
import numpy as np


def main():
    tar_files = sys.argv[1]
    sample_ratio = float(sys.argv[2])

    np.random.seed(1234)

    tar_files = glob.glob(tar_files)
    print(tar_files)

    work_dir = os.path.dirname(tar_files[0])
    os.system(f'mkdir -p {work_dir}/binary_files/')

    for tar_file in tar_files:
        print(f'extracting {tar_file}')
        tar('-C', f'{work_dir}/binary_files/', '-xvf', tar_file)

    all_files = glob.glob(f'{work_dir}/binary_files/*.jsonl')
    all_files.sort()

    print(f'{len(all_files)} in total')
    sampled_files = np.random.choice(all_files, replace=False, size=int(sample_ratio * len(all_files)))
    print(f'{len(sampled_files)} sampled files')

    os.chdir(work_dir)
    with open(f'sampled_binaries.txt', 'w') as f:
        for fname in sampled_files:
            fname = os.path.basename(fname)
            f.write(fname + '\n')

    print('creating tar file')
    os.chdir('binary_files/')
    tar('-cf', f'../sampled_binaries_{sample_ratio}.tar', '-T', '../sampled_binaries.txt')


if __name__ == '__main__':
    main()
