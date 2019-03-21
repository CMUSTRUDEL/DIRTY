from typing import Dict, List

import numpy as np
import torch

import editdistance

from utils.dataset import Dataset
from model.model import RenamingModel


class Evaluator(object):
    @staticmethod
    def get_soft_metrics(pred_name: str, gold_name: str) -> Dict:
        edit_distance = float(editdistance.eval(pred_name, gold_name))
        cer = float(edit_distance / len(gold_name))
        acc = float(pred_name == gold_name)

        return dict(edit_distance=edit_distance,
                    ref_len=len(gold_name),
                    cer=cer,
                    accuracy=acc)

    @staticmethod
    def average(metrics_list: List[Dict]) -> Dict:
        agg_results = dict()
        for metrics in metrics_list:
            for key, val in metrics.items():
                agg_results.setdefault(key, []).append(val)

        avg_results = dict()
        avg_results['corpus_cer'] = sum(agg_results['edit_distance']) / sum(agg_results['ref_len'])

        for key, val in agg_results.items():
            val = avg_results[key]
            avg_results[key] = np.average(val)

        return avg_results

    @staticmethod
    def decode_and_evaluate(model: RenamingModel, dataset: Dataset, config: Dict, return_results=False):
        data_iter = dataset.batch_iterator(batch_size=config['train']['batch_size'],
                                           train=False, progress=True,
                                           config=model.config,
                                           num_readers=config['train']['num_readers'],
                                           num_batchers=config['train']['num_batchers'])

        was_training = model.training
        model.eval()
        example_acc_list = []
        variable_acc_list = []
        need_rename_cases = []

        func_name_in_train_acc_list = []
        func_name_not_in_train_acc_list = []
        func_body_in_train_acc_list = []
        func_body_not_in_train_acc_list = []

        all_examples = []

        with torch.no_grad():
            for batch in data_iter:
                examples = batch.examples
                rename_results = model.decoder.predict([e.ast for e in examples], model.encoder)
                for example, rename_result in zip(examples, rename_results):
                    example_pred_accs = []
                    if len(example.variable_name_map) == 0:
                        continue

                    for old_name, gold_new_name in example.variable_name_map.items():
                        pred = rename_result[old_name]
                        pred_new_name = pred['new_name']
                        var_metric = Evaluator.get_soft_metrics(pred_new_name, gold_new_name)
                        # is_correct = pred_new_name == gold_new_name
                        example_pred_accs.append(var_metric)

                        if gold_new_name != old_name:  # and gold_new_name in model.vocab.target:
                            need_rename_cases.append(var_metric)

                            if example.test_meta['function_name_in_train']:
                                func_name_in_train_acc_list.append(var_metric)
                            else:
                                func_name_not_in_train_acc_list.append(var_metric)

                            if example.test_meta['function_body_in_train']:
                                func_body_in_train_acc_list.append(var_metric)
                            else:
                                func_body_not_in_train_acc_list.append(var_metric)

                    variable_acc_list.extend(example_pred_accs)
                    example_acc_list.append(example_pred_accs)

                    if return_results:
                        all_examples.append((example, rename_result, example_pred_accs))

        valid_example_num = len(example_acc_list)
        num_variables = len(variable_acc_list)
        corpus_acc = Evaluator.average(variable_acc_list)

        if was_training:
            model.train()

        eval_results = dict(corpus_acc=corpus_acc,
                            corpus_need_rename_acc=Evaluator.average(need_rename_cases),
                            func_name_in_train_acc=Evaluator.average(func_name_in_train_acc_list),
                            func_name_not_in_train_acc=Evaluator.average(func_name_not_in_train_acc_list),
                            func_body_in_train_acc=Evaluator.average(func_body_in_train_acc_list),
                            func_body_not_in_train_acc=Evaluator.average(func_body_not_in_train_acc_list),
                            num_variables=num_variables,
                            num_valid_examples=valid_example_num)

        if return_results:
            return eval_results, all_examples
        return eval_results


if __name__ == '__main__':
    print(Evaluator.get_soft_metrics('file_name', 'filename'))
