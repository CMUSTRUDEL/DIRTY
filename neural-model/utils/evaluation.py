import json
import time
from typing import Dict, List, Any

import numpy as np
import torch

import editdistance

from utils import nn_util
from utils.dataset import Dataset
from model.model import RenamingModel


class Evaluator(object):
    @staticmethod
    def get_soft_metrics(pred_name: str, gold_name: str) -> Dict:
        edit_distance = float(editdistance.eval(pred_name, gold_name))
        cer = float(edit_distance / (max(len(gold_name), 1)))
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
        avg_results['corpus_cer'] = sum(agg_results['edit_distance']) / (max(sum(agg_results['ref_len']), 1))

        for key, val in agg_results.items():
            avg_results[key] = np.average(val)

        return avg_results

    @staticmethod
    def evaluate_ppl(model: RenamingModel, dataset: Dataset, config: Dict, predicate: Any = None):
        if predicate is None:
            predicate = lambda e: True

        eval_batch_size = config['train']['batch_size']
        data_iter = dataset.batch_iterator(batch_size=eval_batch_size,
                                           train=False, progress=True,
                                           return_examples=False,
                                           return_prediction_target=True,
                                           config=model.config,
                                           num_readers=config['train']['num_readers'],
                                           num_batchers=config['train']['num_batchers'])

        was_training = model.training
        model.eval()
        cum_log_probs = 0.
        cum_num_examples = 0
        with torch.no_grad():
            for batch in data_iter:
                nn_util.to(batch.tensor_dict, model.device)
                result = model(batch.tensor_dict, batch.tensor_dict['prediction_target'])
                log_probs = result['batch_log_prob'].cpu().tolist()
                for e_id, test_meta in enumerate(batch.tensor_dict['test_meta']):
                    if predicate(test_meta):
                        log_prob = log_probs[e_id]
                        cum_log_probs += log_prob
                        cum_num_examples += 1

        ppl = np.exp(-cum_log_probs / (cum_num_examples + 1e-12))

        if was_training:
            model.train()

        return ppl

    @staticmethod
    def decode_and_evaluate(model: RenamingModel, dataset: Dataset, config: Dict, return_results=False, eval_batch_size=None):
        if eval_batch_size is None:
            eval_batch_size = config['train']['eval_batch_size'] if 'eval_batch_size' in config['train'] else config['train']['batch_size']
        data_iter = dataset.batch_iterator(batch_size=eval_batch_size,
                                           train=False, progress=True,
                                           return_examples=True,
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

        all_examples = dict()

        results = {}
        with torch.no_grad():
            for batch in data_iter:
                examples = batch.examples
                rename_results = model.predict(examples)
                for example, rename_result in zip(examples, rename_results):
                    example_pred_accs = []
                    binary = example.binary_file['file_name'][:example.binary_file['file_name'].index("_")]
                    func_name = example.ast.compilation_unit

                    top_rename_result = rename_result[0]
                    for old_name, gold_new_name in example.variable_name_map.items():
                        pred = top_rename_result[old_name]
                        pred_new_name = pred['new_name']
                        results.setdefault(binary, {}).setdefault(func_name, {})[old_name] = "", pred_new_name
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
                        all_examples[example.binary_file['file_name'] + '_' + str(example.binary_file['line_num'])] = (rename_result, Evaluator.average(example_pred_accs))
                        # all_examples.append((example, rename_result, example_pred_accs))

        json.dump(results, open(f"pred_dire_{time.strftime('%d%H%M')}.json", "w"))

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
