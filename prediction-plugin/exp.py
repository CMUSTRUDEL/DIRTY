"""
Variable renaming

Usage:
    exp.py [options] MODEL_FILE TEST_DATA_FILE

Options:
    -h --help                Show this screen
    --cuda                   Use GPU
    --seed=<int>             Seed [default: 0]
    --extra-config=<str>     extra config [default: {}]
    --save-to=<str>          Save decode results to path
"""
import json
import numpy as np
import os
import pickle
import pprint
import random
import sys
import torch

from docopt import docopt
from model.model import RenamingModel

from utils.dataset import Dataset
from utils.evaluation import Evaluator

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['GPU_DEBUG'] = '2'
# from gpu_profile import gpu_profile

cmd_args = docopt(__doc__)
print(f'Main process id {os.getpid()}', file=sys.stderr)

# seed the RNG
seed = int(cmd_args['--seed'])
print(f'use random seed {seed}', file=sys.stderr)
torch.manual_seed(seed)

if cmd_args['--cuda'] is not None:
    torch.cuda.manual_seed(seed)
np.random.seed(seed * 13 // 7)
random.seed(seed * 17 // 7)
sys.setrecursionlimit(7000)

if cmd_args['--extra-config'] is not None:
    extra_config = json.loads(cmd_args['--extra-config'])
else:
    default_config = '{"decoder": {"remove_duplicates_in_prediction": true} }'
    extra_config = json.loads(default_config)

model_path = cmd_args['MODEL_FILE']
print(f'loading model from [{model_path}]', file=sys.stderr)
model = RenamingModel.load(model_path,
                           use_cuda=cmd_args['--cuda'],
                           new_config=extra_config)
model.eval()

test_set_path = cmd_args['TEST_DATA_FILE']
test_set = Dataset(test_set_path)
decode_results = \
    Evaluator.decode(model, test_set, model.config)
pp = pprint.PrettyPrinter(stream=sys.stderr)
pp.pprint(decode_results)

if cmd_args['--save-to'] is not None:
    save_to = cmd_args['--save-to']
else:
    test_name = test_set_path.split("/")[-1]
    save_to = \
        f'{cmd_args["MODEL_FILE"]}.{test_name}.decode_results.bin'

print(f'Saved decode results to {save_to}', file=sys.stderr)
pickle.dump(decode_results, open(save_to, 'wb'))
