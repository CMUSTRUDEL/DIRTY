import numpy as np

from utils.dataset import Dataset
from model.model import RenamingModel


class Evaluator(object):
    @staticmethod
    def decode_and_evaluate(model: RenamingModel, dataset: Dataset, batch_size=2048):
        data_iter = dataset.batch_iter_from_compressed_file(batch_size=batch_size, shuffle=False)

        was_training = model.training
        model.eval()
        example_acc_list = []
        variable_acc_list = []
        for examples in data_iter:
            rename_results = model.predict(examples)
            for example, rename_result in zip(examples, rename_results):
                tree_acc = []
                for old_name, gold_new_name in example.variable_name_map:
                    pred_new_name = rename_result[old_name]
                    is_correct = pred_new_name == gold_new_name
                    tree_acc.append(is_correct)

                variable_acc_list.extend(tree_acc)
                tree_acc = np.average(tree_acc)
                example_acc_list(tree_acc)

        num_variables = len(variable_acc_list)
        corpus_acc = np.average(variable_acc_list)
        tree_acc = np.average(example_acc_list)

        if was_training:
            model.train()

        return dict(corpus_acc=corpus_acc, tree_acc=tree_acc, num_variables=num_variables)
