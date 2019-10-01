# compute training data statistics
from collections import Counter
import gc
import numpy as np

from utils.dataset import Dataset

gc.collect()


def compute_dataset_stat():
    train_path = 'data/all_trees_tokenized_0410/train-shard-*.tar'
    train_set = Dataset(train_path)

    num_train_examples = 0
    num_variables = []
    num_variables_with_new_name = []
    ast_sizes = []
    var_name_freq = Counter()

    dev_set = Dataset('data/all_trees_tokenized_0410/dev.tar')
    test_set = Dataset('data/all_trees_tokenized_0410/test.tar')

    for dataset_id, dataset in enumerate([train_set, dev_set, test_set]):
        for example in dataset.get_iterator(num_workers=5):
            example_num_variables = len(example.variable_name_map)

            n_var_new_name = 0
            for old_var_name, new_var_name in example.variable_name_map.items():
                if old_var_name != new_var_name:
                    n_var_new_name += 1

                    if dataset_id == 0:
                        var_name_freq[new_var_name] += 1

            num_variables.append(example_num_variables)
            num_variables_with_new_name.append(n_var_new_name)
            ast_sizes.append(example.ast.size)

            del example

    with open('data/all_trees_tokenized_0410/train_var_name_freq.txt', 'w') as f:
        for var_name, freq in var_name_freq.most_common():
            f.write(f'{var_name}\t{freq}\n')

    with open('data/all_trees_tokenized_0410/stat.txt', 'w') as f:
        print(len(train_set), len(dev_set), len(test_set), file=f)
        print('num total functions: ', num_train_examples, file=f)
        print('avg. size of AST:', np.average(ast_sizes), file=f)
        print('avg. num of variables:', np.average(num_variables), file=f)
        print('avg. num of variables with new name:', np.average(num_variables_with_new_name), file=f)


if __name__ == '__main__':
    compute_dataset_stat()
