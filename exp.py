import sys
from typing import List, Tuple, Dict

import numpy as np
import os

import torch

from model.decoder import SimpleDecoder
from model.encoder import GraphASTEncoder
from model.gnn import AdjacencyList, GatedGraphNeuralNetwork
from model.model import RenamingModel
from utils import nn_util
from utils.ast import AbstractSyntaxTree
from utils.dataset import Dataset


def train(time=None):
    # a grammar defines the syntax types of nodes on ASTs
    grammar = Grammar()
    # a vocabulary defines the collection of all source and target variable names
    vocab = Vocab()
    encoder = GraphASTEncoder(ast_node_encoding_size=128, )
    decoder = SimpleDecoder(ast_node_encoding_size=128,
                            tgt_name_vocab_size=len(vocab.tgt))
    model = RenamingModel(encoder, decoder)

    # load training dataset, which is a collection of ASTs and maps of gold-standard renamings
    train_set: List[Tuple[AbstractSyntaxTree, Dict]] = Dataset.load()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)
    nn_util.glorot_init(params)

    # training loop
    train_iter = 0.
    log_every = 10
    cum_loss = cum_examples = 0.
    t1 = time.time()

    while True:
        for examples in nn_util.batch_iter(train_set, batch_size=32, shuffle=True):
            train_iter += 1
            optimizer.zero_grad()

            src_asts = [e.ast for e in examples]
            rename_maps = [e.rename_map for e in examples]

            tgt_log_probs = model(src_asts, rename_maps)

            loss = -tgt_log_probs.mean()

            cum_loss += loss.item()
            cum_examples += len(examples)

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 5.)

            optimizer.step()

        if train_iter % log_every == 0:
            print(f'[Learner] train_iter={train_iter} avg. loss={cum_loss / cum_examples}, '
                  f'{cum_examples} examples ({cum_examples / (time.time() - t1)} examples/s)', file=sys.stderr)

            cum_loss = cum_examples = 0.
            t1 = time.time()
