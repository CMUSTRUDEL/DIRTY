# evaluate debin
import json
from utils.dataset import get_json_iterator_from_tar_file, Example, Dataset
from utils.evaluation import *


def evaluate_debin(debin_output_path, test_example_path):
    debin_predicted_data = Dataset(debin_output_path)

    debin_prediction_dict = dict()

    for example in debin_predicted_data.get_iterator(num_workers=5):
        example_key = example.binary_file['file_name'].split('/')[-1] + '_' + str(example.ast.compilation_unit)

        assert example_key not in debin_prediction_dict

        pred_result = dict(example.variable_name_map.items())
        debin_prediction_dict[example_key] = pred_result

        del example

    #     num_examples_without_pred = 0
    #     for example in test_examples:
    #         example_key = example.binary_file['file_name'] + '_' + str(example.ast.compilation_unit)
    #         if example_key not in debin_prediction_dict:
    #             num_examples_without_pred += 1
    #             debin_prediction_dict[example_key].variable_name_map

    #     print(f'Num. examples without prediction: {num_examples_without_pred}')

    need_rename_cases = []
    func_name_in_train_acc_list = []
    func_name_not_in_train_acc_list = []
    func_body_in_train_acc_list = []
    func_body_not_in_train_acc_list = []
    debin_predicted_func_num = 0

    test_set = Dataset(test_example_path)

    for example in test_set.get_iterator(num_workers=5):
        example_key = example.binary_file['file_name'].split('/')[-1] + '_' + str(example.ast.compilation_unit)
        if example_key in debin_prediction_dict:
            rename_result = debin_prediction_dict[example_key]
            debin_predicted_func_num += 1
        else:
            rename_result = {k: k for k in example.variable_name_map}  # use identity predictions

        example_pred_accs = []

        for old_name, gold_new_name in example.variable_name_map.items():
            pred_new_name = rename_result[old_name]
            var_metric = Evaluator.get_soft_metrics(pred_new_name, gold_new_name)

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

    eval_results = dict(corpus_need_rename_acc=Evaluator.average(need_rename_cases),
                        func_name_in_train_acc=Evaluator.average(func_name_in_train_acc_list),
                        func_name_not_in_train_acc=Evaluator.average(func_name_not_in_train_acc_list),
                        func_body_in_train_acc=Evaluator.average(func_body_in_train_acc_list),
                        func_body_not_in_train_acc=Evaluator.average(func_body_not_in_train_acc_list))

    print('Num. functions debin predicted: ', debin_predicted_func_num)
    print(eval_results)


if __name__ == '__main__':
    evaluate_debin('data/debin.predictions.0.01.tar', 'data/all_trees_tokenized_0410/test.tar')
