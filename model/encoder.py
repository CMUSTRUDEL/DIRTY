import numpy as np
from typing import List, Dict
import pickle

import torch
import torch.nn as nn
from sentencepiece import SentencePieceProcessor

from model.embedding import SubTokenEmbedder
from utils import util
from utils.ast import AbstractSyntaxTree
from model.gnn import GatedGraphNeuralNetwork, AdjacencyList
from utils.grammar import Grammar
from utils.graph import PackedGraph
from utils.vocab import Vocab


class Encoder(nn.Module):
    pass


class GraphASTEncoder(Encoder):
    """An encoder based on gated recurrent graph neural networks"""
    def __init__(self,
                 gnn: GatedGraphNeuralNetwork,
                 connections: List[str],
                 sub_token_embedder: SubTokenEmbedder,
                 vocab: Vocab):
        super(GraphASTEncoder, self).__init__()

        self.connections = connections
        self.gnn = gnn

        self.vocab = vocab
        self.grammar = grammar = vocab.grammar
        self.node_type_embedding = nn.Embedding(len(grammar.syntax_types) + 2, gnn.hidden_size)
        self.var_node_name_embedding = nn.Embedding(len(vocab.source), gnn.hidden_size, padding_idx=0)
        self.variable_master_node_type_idx = len(grammar.syntax_types)
        self.master_node_type_idx = self.variable_master_node_type_idx + 1
        self.type_and_content_hybrid = nn.Linear(2 * gnn.hidden_size, gnn.hidden_size)

        self.sub_token_embedder = sub_token_embedder

        self.config: Dict = None

    @property
    def device(self):
        return self.node_type_embedding.weight.device

    @classmethod
    def default_params(cls):
        return {
            'gnn': {
                'hidden_size': 128,
                'layer_timesteps': [8],
                'residual_connections': {0: [0]}
            },
            'connections': {'top_down', 'bottom_up', 'variable_master_nodes', 'terminals', 'master_node'},
            'vocab_file': None,
            'bpe_model_path': None
        }

    @classmethod
    def build(cls, config):
        params = util.update(GraphASTEncoder.default_params(), config)

        connections = params['connections']
        connection2edge_type = {
            'top_down': 1,
            'bottom_up': 1,
            'variable_master_nodes': 2,
            'terminals': 2,
            'master_node': 2
        }
        num_edge_types = sum(connection2edge_type[key] for key in connections)
        gnn = GatedGraphNeuralNetwork(hidden_size=params['gnn']['hidden_size'],
                                      layer_timesteps=params['gnn']['layer_timesteps'],
                                      residual_connections=params['gnn']['residual_connections'],
                                      num_edge_types=num_edge_types)

        vocab = torch.load(params['vocab_file'])
        sub_token_embedder = SubTokenEmbedder(params['bpe_model_path'], gnn.hidden_size)

        model = cls(gnn,
                    params['connections'],
                    sub_token_embedder,
                    vocab)
        model.config = params

        return model

    def forward(self, tensor_dict: Dict[str, torch.Tensor]):
        tree_node_init_encoding = self.get_batched_tree_init_encoding(tensor_dict)

        # (batch_size, node_encoding_size)
        tree_node_encoding = self.gnn.compute_node_representations(initial_node_representation=tree_node_init_encoding,
                                                                   adjacency_lists=tensor_dict['adj_lists'])

        variable_master_node_ids = tensor_dict['variable_master_node_ids']

        variable_master_node_encoding = tree_node_encoding[variable_master_node_ids]

        context_encoding = dict(
            tree_node_init_encoding=tree_node_init_encoding,
            packed_tree_node_encoding=tree_node_encoding,
            variable_master_node_encoding=variable_master_node_encoding,
        )

        return context_encoding

    @classmethod
    def to_packed_graph(cls, asts: List[AbstractSyntaxTree], connections: List) -> PackedGraph:
        packed_graph = PackedGraph(asts)

        node_adj_list = []
        terminal_nodes_adj_list = []
        master_node_adj_list = []
        var_master_nodes_adj_list = []

        max_variable_num = max(len(ast.variables) for ast in asts)
        var_master_node_restoration_indices = torch.zeros(len(asts), max_variable_num, dtype=torch.long)
        var_master_node_restoration_indices_mask = torch.zeros(len(asts), max_variable_num)

        for ast_id, ast in enumerate(asts):
            for prev_node, succ_node in ast.adjacency_list:
                prev_node_packed_id = packed_graph.get_packed_node_id(ast_id, prev_node)
                succ_node_packed_id = packed_graph.get_packed_node_id(ast_id, succ_node)

                node_adj_list.append((prev_node_packed_id, succ_node_packed_id))

            if 'terminals' in connections:
                for i in range(len(ast.terminal_nodes) - 1):
                    terminal_i = ast.terminal_nodes[i]
                    terminal_ip1 = ast.terminal_nodes[i + 1]

                    terminal_nodes_adj_list.append((
                        packed_graph.get_packed_node_id(ast_id, terminal_i),
                        packed_graph.get_packed_node_id(ast_id, terminal_ip1)
                    ))

            # add master node
            if 'master_node' in connections:
                master_node_id = packed_graph.register_node(ast_id, 'master_node', group='master_nodes')

                master_node_adj_list.extend([
                    (master_node_id, packed_graph.get_packed_node_id(ast_id, node))
                    for node in ast
                ])

            # add prediction node to packed graph
            for i, (var_name, var_nodes) in enumerate(ast.variables.items()):
                var_master_node_id, node_id_in_group = packed_graph.register_node(ast_id, var_name,
                                                                                  group='variable_master_nodes',
                                                                                  return_node_index_in_group=True)

                var_master_node_restoration_indices[ast_id, i] = node_id_in_group

                var_master_nodes_adj_list.extend([(
                    var_master_node_id, packed_graph.get_packed_node_id(ast_id, node))
                    for node in var_nodes
                ])

            var_master_node_restoration_indices_mask[ast_id, :len(ast.variables)] = 1.

        adj_lists = []
        if 'top_down' in connections:
            adj_lists.append(node_adj_list)
        if 'bottom_up' in connections:
            reversed_node_adj_list = [(n2, n1) for n1, n2 in node_adj_list]
            adj_lists.append(reversed_node_adj_list)
        if 'variable_master_nodes' in connections:
            reversed_var_master_nodes_adj_list = [(n2, n1) for n1, n2 in var_master_nodes_adj_list]
            adj_lists.append(master_node_adj_list)
            adj_lists.append(reversed_var_master_nodes_adj_list)
        if 'terminals' in connections:
            reversed_terminal_nodes_adj_list = [(n2, n1) for n1, n2 in terminal_nodes_adj_list]
            adj_lists.append(terminal_nodes_adj_list)
            adj_lists.append(reversed_terminal_nodes_adj_list)
        if 'master_node' in connections:
            reversed_master_node_adj_list = [(n2, n1) for n1, n2 in master_node_adj_list]
            adj_lists.append(master_node_adj_list)
            adj_lists.append(reversed_master_node_adj_list)

        adj_lists = [
            AdjacencyList(adj_list=adj_list, node_num=packed_graph.size) for adj_list in adj_lists
        ]

        packed_graph.adj_lists = adj_lists
        packed_graph.variable_master_node_restoration_indices = var_master_node_restoration_indices
        packed_graph.variable_master_node_restoration_indices_mask = var_master_node_restoration_indices_mask

        return packed_graph

    @classmethod
    def to_tensor_dict(cls, packed_graph: PackedGraph,
                       bpe_model: SentencePieceProcessor,
                       bpe_pad_idx: int,
                       grammar: Grammar,
                       vocab: Vocab) -> Dict[str, torch.Tensor]:
        # predefined index
        variable_master_node_type_idx = len(grammar.syntax_types)
        master_node_type_idx = variable_master_node_type_idx + 1

        node_type_indices = torch.zeros(packed_graph.size, dtype=torch.long)
        var_node_name_indices = torch.zeros(packed_graph.size, dtype=torch.long)

        sub_tokens_list = []
        node_with_subtokens_indices = []

        for i, (ast_id, group, node_key) in enumerate(packed_graph.nodes.values()):
            ast = packed_graph.trees[ast_id]
            if group == 'ast_nodes':
                node = ast.id_to_node[node_key]
                type_idx = grammar.syntax_type_to_id[node.node_type]
                if node.is_variable_node:
                    var_node_name_indices[i] = vocab.source[node.old_name]

                if node.node_type == 'obj':
                    # compute variable embedding
                    node_sub_tokes = bpe_model.encode_as_ids(node.name)
                    sub_tokens_list.append(node_sub_tokes)
                    node_with_subtokens_indices.append(i)
            elif group == 'variable_master_nodes':
                type_idx = variable_master_node_type_idx
                var_node_name_indices[i] = vocab.source[node_key]
            elif group == 'master_nodes':
                type_idx = master_node_type_idx
            else:
                raise ValueError()

            node_type_indices[i] = type_idx

        sub_tokens_indices = None
        if node_with_subtokens_indices:
            max_subtoken_num = max(len(x) for x in sub_tokens_list)
            sub_tokens_indices = np.zeros((len(sub_tokens_list), max_subtoken_num), dtype=np.int64)
            sub_tokens_indices.fill(bpe_pad_idx)
            for i, token_ids in enumerate(sub_tokens_list):
                sub_tokens_indices[i, :len(token_ids)] = token_ids

            sub_tokens_indices = torch.from_numpy(sub_tokens_indices)

        return dict(
            node_type_indices=torch.tensor(node_type_indices, dtype=torch.long),
            var_node_name_indices=torch.tensor(var_node_name_indices, dtype=torch.long),
            node_with_subtokens_indices=torch.tensor(node_with_subtokens_indices, dtype=torch.long),
            sub_tokens_indices=sub_tokens_indices
        )

    def get_batched_tree_init_encoding(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_type_embedding = self.node_type_embedding(tensor_dict['node_type_indices'])
        node_content_embedding = self.var_node_name_embedding(tensor_dict['var_node_name_indices']) * \
            torch.ne(tensor_dict['var_node_name_indices'], 0.).float().unsqueeze(-1)

        if tensor_dict['node_with_subtokens_indices'].size(0) > 0:
            obj_node_content_embedding = self.sub_token_embedder(tensor_dict['sub_tokens_indices'])
            node_content_embedding = node_content_embedding.scatter(0, tensor_dict['node_with_subtokens_indices'].unsqueeze(-1).expand_as(obj_node_content_embedding),
                                                                    obj_node_content_embedding)

        tree_node_embedding = self.type_and_content_hybrid(torch.cat([node_type_embedding, node_content_embedding], dim=-1))

        return tree_node_embedding

    def unpack_batch_encoding(self, flattened_node_encodings: torch.Tensor,
                              batch_syntax_trees: List[AbstractSyntaxTree],
                              example_node2batch_node_map: Dict):
        batch_size = len(batch_syntax_trees)
        max_node_num = max(tree.size for tree in batch_syntax_trees)

        index = np.zeros((batch_size, max_node_num), dtype=np.int64)
        batch_tree_node_masks = torch.zeros(batch_size, max_node_num, device=self.device)
        for e_id, syntax_tree in enumerate(batch_syntax_trees):
            example_nodes_with_batch_id = [(example_node_id, batch_node_id)
                                           for (_e_id, example_node_id), batch_node_id
                                           in example_node2batch_node_map.items()
                                           if _e_id == e_id]
            # example_nodes_batch_id = list(map(lambda x: x[1], sorted(example_nodes_with_batch_id, key=lambda t: t[0])))
            sorted_example_nodes_with_batch_id = sorted(example_nodes_with_batch_id, key=lambda t: t[0])
            example_nodes_batch_id = [t[1] for t in sorted_example_nodes_with_batch_id]

            index[e_id, :len(example_nodes_batch_id)] = example_nodes_batch_id
            batch_tree_node_masks[e_id, :len(example_nodes_batch_id)] = 1.

        # (batch_size, max_node_num, node_encoding_size)
        batch_node_encoding = flattened_node_encodings[torch.from_numpy(index).to(flattened_node_encodings.device)]
        batch_node_encoding.data.masked_fill_((1. - batch_tree_node_masks).byte().unsqueeze(-1), 0.)

        return dict(batch_tree_node_encoding=batch_node_encoding,
                    batch_tree_node_masks=batch_tree_node_masks)
