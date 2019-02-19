from collections import OrderedDict

import torch
import torch.nn as nn

from typing import List, Dict, Tuple

from utils.ast import AbstractSyntaxTree, AbstractSyntaxNode, TerminalNode
from model.gnn import GatedGraphNeuralNetwork, main, AdjacencyList


class Encoder(nn.Module):
    pass


class GraphASTEncoder(Encoder):
    """An encoder based on gated recurrent graph neural networks"""
    def __init__(self,
                 ast_node_encoding_size: int,
                 grammar: Grammar,
                 vocab: Vocab):
        super(GraphASTEncoder, self).__init__()

        self.gnn = GatedGraphNeuralNetwork(hidden_size=ast_node_encoding_size,
                                           num_edge_types=2)

        self.src_node_embedding = nn.Embedding(len(vocab) + len(grammar.syntax_types), ast_node_encoding_size)
        self.vocab = vocab

    def forward(self, asts: List[AbstractSyntaxTree]):
        adj_lists, tree_node2batch_graph_node, batch_graph_node2tree_node = self.get_batched_adjacency_lists(asts)

        tree_node_init_encoding = self.get_batched_tree_init_encoding(asts, tree_node2batch_graph_node, batch_graph_node2tree_node)

        # (batch_size, node_encoding_size)
        tree_node_encoding = self.gnn.compute_node_representations(initial_node_representation=tree_node_init_encoding,
                                                                   adjacency_lists=adj_lists)

        return tree_node_encoding

    def get_batched_adjacency_lists(self, asts: List[AbstractSyntaxTree]):
        tree_node2batch_graph_node = OrderedDict()

        ast_adj_list = []
        reversed_ast_adj_list = []
        terminal_nodes_adj_list = []
        reversed_terminal_nodes_adj_list = []

        for ast_id, syntax_tree in enumerate(asts):
            for prev_node_id, succ_node_id in syntax_tree.adjacency_list:
                # an edge from preceding to succeeding node
                node_s_batch_id = tree_node2batch_graph_node.setdefault((ast_id, prev_node_id), len(tree_node2batch_graph_node))
                node_t_batch_id = tree_node2batch_graph_node.setdefault((ast_id, succ_node_id), len(tree_node2batch_graph_node))

                ast_adj_list.append((node_s_batch_id, node_t_batch_id))
                reversed_ast_adj_list.append((node_t_batch_id, node_s_batch_id))

            # add bi-directional connection between adjacent terminal nodes
            for node_id, succ_node_id in syntax_tree.adjacent_terminal_nodes:
                cur_token_batch_id = tree_node2batch_graph_node[(ast_id, node_id)]
                next_token_batch_id = tree_node2batch_graph_node[(ast_id, succ_node_id)]

                terminal_nodes_adj_list.append((cur_token_batch_id, next_token_batch_id))
                reversed_terminal_nodes_adj_list.append((next_token_batch_id, cur_token_batch_id))

        all_nodes_num = len(tree_node2batch_graph_node)
        adj_lists = [AdjacencyList(node_num=all_nodes_num, adj_list=ast_adj_list),
                     AdjacencyList(node_num=all_nodes_num, adj_list=reversed_ast_adj_list)]

        if terminal_nodes_adj_list:
            adj_lists.extend([
                AdjacencyList(node_num=all_nodes_num, adj_list=terminal_nodes_adj_list),
                AdjacencyList(node_num=all_nodes_num, adj_list=reversed_terminal_nodes_adj_list)
            ])

        batch_graph_node2tree_node = OrderedDict([(v, k) for k, v in tree_node2batch_graph_node.items()])

        return adj_lists, tree_node2batch_graph_node, batch_graph_node2tree_node

    def get_batched_tree_init_encoding(self, asts: List[AbstractSyntaxTree],
                                       tree_node2batch_graph_node: Dict[Tuple[int, int], int],
                                       batch_graph_node2tree_node: Dict[int, Tuple[int, int]]):
        indices = []

        for i, (batch_node_id, (tree_id, tree_node_id)) in enumerate(batch_graph_node2tree_node.items()):
            node = asts[tree_id].id2node[tree_node_id]

            if isinstance(node, AbstractSyntaxNode):
                idx = self.grammar.type2id[node.type] + len(self.vocab)

            elif isinstance(node, TerminalNode):
                idx = self.vocab[node.value]

            indices.append(idx)

        tree_node_embedding = self.src_node_embedding[torch.tensor(indices,
                                                                   dtype=torch.long,
                                                                   device=self.device)]

        return tree_node_embedding
