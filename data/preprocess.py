import sh
from sh import ls, grep, cat, tar, mkdir, rm, mv
import glob
import os

tree_files = list(map(lambda x: x.strip(), grep(ls('-1', _tty_out=False), '-e', '\w-trees.tar.gz')))
mkdir('tmp')
for tree_file in tree_files:
    tar('xzf', tree_file, '-C', 'tmp/')

mkdir('trees')
for tree_file in tree_files:
    tree_prefix = tree_file[:tree_file.index('.')]
    # mv(os.path.join('tmp/', tree_prefix) + '/*', 'trees/')
    os.system(f'mv tmp/{tree_prefix}/* trees/')
os.chdir('trees')
tar('cf', '../trees.tar', '.')

# cat(tree_files, _out='trees.tar')
# tar(tar(cat(tree_files), '-izxf', '-'), '-cf', 'trees.tar', '-T', '-')
# print(tree_files)
# tar(cat(*tree_files), '-cf', 'trees.tar')
