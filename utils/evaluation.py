import numpy as np

from utils.dataset import Dataset
from model.model import RenamingModel


class Evaluator(object):
    @staticmethod
    def decode_and_evaluate(model: RenamingModel, dataset: Dataset, batch_size=2048):
        data_iter = dataset.batch_iterator(batch_size=batch_size, shuffle=False, progress=False, config=model.config)

        was_training = model.training
        model.eval()
        example_acc_list = []
        variable_acc_list = []
        need_rename_cases = []
        for batch in data_iter:
            examples = batch.examples
            rename_results = model.decode([e.ast for e in examples])
            for example, rename_result in zip(examples, rename_results):
                tree_acc = []
                if len(example.variable_name_map) == 0:
                    continue

                for old_name, gold_new_name in example.variable_name_map.items():
                    pred = rename_result[old_name]
                    pred_new_name = pred['new_name']
                    is_correct = pred_new_name == gold_new_name
                    tree_acc.append(is_correct)

                    if gold_new_name != old_name:  # and gold_new_name in model.vocab.target:
                        need_rename_cases.append(is_correct)

                variable_acc_list.extend(tree_acc)
                tree_acc = np.average(tree_acc)
                example_acc_list.append(tree_acc)

        num_variables = len(variable_acc_list)
        corpus_acc = np.average(variable_acc_list)
        corpus_need_rename_acc = np.average(need_rename_cases)
        valid_example_num = len(example_acc_list)
        tree_acc = np.average(example_acc_list)

        if was_training:
            model.train()

        return dict(corpus_acc=corpus_acc, corpus_need_rename_acc=corpus_need_rename_acc,
                    tree_acc=tree_acc,
                    num_variables=num_variables, num_valid_examples=valid_example_num)
