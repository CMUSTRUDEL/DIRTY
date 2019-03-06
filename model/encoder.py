from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple

import torch
import torch.nn as nn

from model.embedding import SubTokenEmbedder
from utils.ast import AbstractSyntaxTree, SyntaxNode, TerminalNode
from model.gnn import GatedGraphNeuralNetwork, main, AdjacencyList
from utils.grammar import Grammar
from utils.vocab import Vocab


class Encoder(nn.Module):
    pass


class PackedGraph(object):
    def __init__(self, asts: List[AbstractSyntaxTree]):
        self.trees = []
        self.node_groups: List[Dict] = []
        self._group_node_count = defaultdict(int)
        self.ast_node_to_packed_node: List[Dict] = []
        self.packed_node_to_ast_node: Dict = OrderedDict()
        self._nodes = OrderedDict()

        for ast in asts:
            self.register_tree(ast)

    def register_tree(self, ast: AbstractSyntaxTree):
        ast_id = self.tree_num
        self.trees.append(ast)
        self.node_groups.append(dict())

        for node in ast:
            self.register_node(ast_id, node.node_id)

    def register_node(self, tree_id, node, group='ast_nodes', return_node_index_in_group=False):
        if group not in self.node_groups[tree_id]:
            self.node_groups[tree_id][group] = OrderedDict()

        node_group = self.node_groups[tree_id][group]
        packed_node_id = self.size
        node_group[node] = packed_node_id
        self._group_node_count[group] += 1
        self._nodes[packed_node_id] = (tree_id, group, node)

        if return_node_index_in_group:
            node_index_in_group = self._group_node_count[group] - 1
            return packed_node_id, node_index_in_group

        return packed_node_id

    def get_packed_node_id(self, tree_id, node, group='ast_nodes'):
        if isinstance(node, SyntaxNode):
            node = node.node_id
        return self.node_groups[tree_id][group][node]

    @property
    def size(self):
        return len(self._nodes)

    @property
    def nodes(self):
        return self._nodes

    def get_nodes_by_group(self, group):
        for i in range(self.tree_num):
            for node, packed_node_id in self.node_groups[i][group].items():
                yield node, packed_node_id

    @property
    def tree_num(self):
        return len(self.trees)


class GraphASTEncoder(Encoder):
    """An encoder based on gated recurrent graph neural networks"""
    def __init__(self,
                 ast_node_encoding_size: int,
                 grammar: Grammar,
                 vocab: Vocab,
                 bpe_model_path: str):
        super(GraphASTEncoder, self).__init__()

        self.connections = ['master_node']

        self.gnn = GatedGraphNeuralNetwork(hidden_size=ast_node_encoding_size,
                                           layer_timesteps=[8, 2, 8, 2],
                                           residual_connections={1: [0], 3: [2]},
                                           num_edge_types=8)

        self.src_node_embedding = nn.Embedding(len(vocab.source) + len(grammar.syntax_types) + 1, ast_node_encoding_size)
        self.master_node_embed_idx = self.src_node_embedding.num_embeddings - 1
        self.sub_token_embedder = SubTokenEmbedder(bpe_model_path, ast_node_encoding_size)
        self.vocab = vocab
        self.grammar = grammar

    @property
    def device(self):
        return self.src_node_embedding.weight.device

    def forward(self, asts: List[AbstractSyntaxTree]):
        batch_size = len(asts)

        packed_graph = self.to_packed_graph(asts)

        tree_node_init_encoding = self.get_batched_tree_init_encoding(packed_graph)

        # (batch_size, node_encoding_size)
        tree_node_encoding = self.gnn.compute_node_representations(initial_node_representation=tree_node_init_encoding,
                                                                   adjacency_lists=packed_graph.adj_lists)

        variable_master_node_ids = [_id for node, _id in packed_graph.get_nodes_by_group('variable_master_nodes')]

        variable_master_node_encoding = tree_node_encoding[variable_master_node_ids]

        context_encoding = dict(
            packed_tree_node_encoding=tree_node_encoding,
            variable_master_node_encoding=variable_master_node_encoding,
            packed_graph=packed_graph
        )

        return context_encoding

    def to_packed_graph(self, asts: List[AbstractSyntaxTree]) -> PackedGraph:
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

            for i in range(len(ast.terminal_nodes) - 1):
                terminal_i = ast.terminal_nodes[i]
                terminal_ip1 = ast.terminal_nodes[i + 1]

                terminal_nodes_adj_list.append((
                    packed_graph.get_packed_node_id(ast_id, terminal_i),
                    packed_graph.get_packed_node_id(ast_id, terminal_ip1)
                ))

            # add master node
            if 'master_node' in self.connections:
                master_node_id = packed_graph.register_node(ast_id, 'master_nodes', group='master_nodes')

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

        reversed_node_adj_list = [(n2, n1) for n1, n2 in node_adj_list]
        reversed_terminal_nodes_adj_list = [(n2, n1) for n1, n2 in terminal_nodes_adj_list]
        reversed_var_master_nodes_adj_list = [(n2, n1) for n1, n2 in var_master_nodes_adj_list]
        adj_lists = [
            node_adj_list, reversed_node_adj_list,
            terminal_nodes_adj_list, reversed_terminal_nodes_adj_list,
            var_master_nodes_adj_list, reversed_var_master_nodes_adj_list
        ]

        if 'master_node' in self.connections:
            adj_lists.append(master_node_adj_list)
            reversed_master_node_adj_list = [(n2, n1) for n1, n2 in master_node_adj_list]
            adj_lists.append(reversed_master_node_adj_list)

        adj_lists = [
            AdjacencyList(adj_list=adj_list, node_num=packed_graph.size, device=self.device) for adj_list in adj_lists
        ]

        packed_graph.adj_lists = adj_lists
        packed_graph.variable_master_node_restoration_indices = var_master_node_restoration_indices.to(self.device)
        packed_graph.variable_master_node_restoration_indices_mask = var_master_node_restoration_indices_mask.to(self.device)

        return packed_graph

    def get_batched_tree_init_encoding(self, packed_graph: PackedGraph) -> torch.Tensor:
        indices = torch.zeros(packed_graph.size, dtype=torch.long)
        sub_tokens_list = []
        sub_tokens_indices = []
        for i, (ast_id, group, node_on_graph) in enumerate(packed_graph.nodes.values()):
            ast = packed_graph.trees[ast_id]
            idx = 0
            if group == 'ast_nodes':
                node = ast.id_to_node[node_on_graph]

                if node.is_variable_node:
                    idx = self.vocab.source[node.var_id]
                else:
                    idx = self.grammar.syntax_type_to_id[node.node_type] + len(self.vocab.source)

                if node.node_type == 'obj':
                    # compute variable embedding
                    sub_tokens_list.append(node.sub_tokens)
                    sub_tokens_indices.append(i)
            elif group == 'variable_master_nodes':
                idx = self.grammar.syntax_type_to_id['var'] + len(self.vocab.source)
            elif group == 'master_nodes':
                idx = self.master_node_embed_idx
            else:
                raise ValueError()

            indices[i] = idx

        tree_node_embedding = self.src_node_embedding(indices.to(self.device))
        if sub_tokens_indices:
            obj_tokens_embedding = self.sub_token_embedder(sub_tokens_list)
            tree_node_embedding[sub_tokens_indices] = obj_tokens_embedding

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
