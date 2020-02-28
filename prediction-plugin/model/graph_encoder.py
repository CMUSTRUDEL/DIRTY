from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import nn as nn

from model.embedding import NodeTypeEmbedder, SubTokenEmbedder
from model.encoder import Encoder
from model.gnn import GatedGraphNeuralNetwork, AdjacencyList
from model.sequential_encoder import SequentialEncoder
from utils import util
from utils.ast import AbstractSyntaxTree
from utils.grammar import Grammar
from utils.graph import PackedGraph
from utils.vocab import Vocab


class GraphASTEncoder(Encoder):
    """An encoder based on gated recurrent graph neural networks"""
    def __init__(self,
                 gnn: GatedGraphNeuralNetwork,
                 connections: List[str],
                 node_syntax_type_embedding_size: int,
                 decoder_hidden_size: int,
                 node_type_embedder: NodeTypeEmbedder,
                 node_content_embedder: SubTokenEmbedder,
                 vocab: Vocab,
                 config):
        super(GraphASTEncoder, self).__init__()

        self.connections = connections
        self.gnn = gnn

        self.vocab = vocab
        self.grammar = grammar = vocab.grammar

        self.node_syntax_type_embedding = nn.Embedding(
            len(grammar.syntax_types) + 2,
            node_syntax_type_embedding_size
        )
        self.variable_master_node_type_idx = len(grammar.syntax_types)
        self.master_node_type_idx = self.variable_master_node_type_idx + 1

        self.var_node_name_embedding = \
            nn.Embedding(len(vocab.source), gnn.hidden_size, padding_idx=0)

        self.node_type_embedder = node_type_embedder
        self.node_content_embedder = node_content_embedder

        self.type_and_content_hybrid = nn.Linear(
            node_syntax_type_embedding_size
            + node_type_embedder.embeddings.embedding_dim
            + node_content_embedder.embeddings.embedding_dim,
            gnn.hidden_size,
            bias=False
        )

        self.decoder_cell_init = \
            nn.Linear(gnn.hidden_size, decoder_hidden_size)

        self.init_with_seq_encoding = config['init_with_seq_encoding']
        if self.init_with_seq_encoding:
            self.seq_encoder = SequentialEncoder.build(config['seq_encoder'])
            if config['seq_encoder']['source_encoding_size'] \
               != gnn.hidden_size:
                self.seq_variable_encoding_to_graph_linear = \
                    nn.Linear(
                        config['seq_encoder']['source_encoding_size'],
                        gnn.hidden_size
                    )
            else:
                self.seq_variable_encoding_to_graph_linear = lambda x: x

        self.config: Dict = config

    @property
    def device(self):
        return self.node_syntax_type_embedding.weight.device

    @classmethod
    def default_params(cls):
        return {
            'gnn': {
                'hidden_size': 128,
                'layer_timesteps': [8],
                'residual_connections': {'0': [0]}
            },
            'connections': {
                'top_down',
                'bottom_up',
                'variable_master_nodes',
                'terminals',
                'master_node',
                'func_root_to_arg'
            },
            'vocab_file': None,
            'bpe_model_path': None,
            'node_syntax_type_embedding_size': 64,
            'node_type_embedding_size': 64,
            'node_content_embedding_size': 128,
            'init_with_seq_encoding': False
        }

    @classmethod
    def build(cls, config):
        params = util.update(GraphASTEncoder.default_params(), config)

        if False:
            print(params)

        connections = params['connections']
        connection2edge_type = {
            'top_down': 1,
            'bottom_up': 1,
            'variable_master_nodes': 2,
            'terminals': 2,
            'master_node': 2,
            'var_usage': 2,
            'func_root_to_arg': 1
        }
        num_edge_types = sum(connection2edge_type[key] for key in connections)
        gnn = GatedGraphNeuralNetwork(
            hidden_size=params['gnn']['hidden_size'],
            layer_timesteps=params['gnn']['layer_timesteps'],
            residual_connections=params['gnn']['residual_connections'],
            num_edge_types=num_edge_types
        )

        vocab = Vocab.load(params['vocab_file'])
        node_type_embedder = NodeTypeEmbedder(
            len(vocab.grammar.variable_types),
            params['node_type_embedding_size']
        )
        node_content_embedder = SubTokenEmbedder(
            vocab.obj_name.subtoken_model_path,
            params['node_content_embedding_size']
        )

        model = cls(gnn,
                    params['connections'],
                    params['node_syntax_type_embedding_size'],
                    params['decoder_hidden_size'],
                    node_type_embedder,
                    node_content_embedder,
                    vocab,
                    config=params)

        return model

    def forward(self, tensor_dict: Dict[str, torch.Tensor]):
        tree_node_init_encoding = \
            self.get_batched_tree_init_encoding(tensor_dict)

        if self.init_with_seq_encoding:
            # scatter sequential encoding results to variable positions
            seq_encoding = self.seq_encoder(tensor_dict['seq_encoder_input'])
            seq_var_encoding = seq_encoding['variable_encoding']
            tree_node_init_encoding[
                tensor_dict['variable_master_node_ids']
            ] = self.seq_variable_encoding_to_graph_linear(
                seq_var_encoding.view(
                    -1,
                    seq_var_encoding.size(-1)
                )[tensor_dict['var_repr_flattened_positions']]
            )

        # (batch_size, node_encoding_size)
        tree_node_encoding = self.gnn.compute_node_representations(
            initial_node_representation=tree_node_init_encoding,
            adjacency_lists=tensor_dict['adj_lists']
        )

        connections = self.config['connections']

        if 'variable_master_nodes' in connections:
            variable_master_node_ids = tensor_dict['variable_master_node_ids']
            variable_encoding = tree_node_encoding[variable_master_node_ids]
        else:
            var_nodes_encoding = \
                tree_node_encoding[tensor_dict['variable_node_ids']]
            variable_num = tensor_dict['variable_mention_nums'].size(0)
            variable_encoding = torch.zeros(
                variable_num,
                var_nodes_encoding.size(-1),
                device=self.device
            )
            var_nodes_encoding_sum = variable_encoding.scatter_add_(
                0,
                tensor_dict[
                    'variable_node_variable_ids'
                ].unsqueeze(-1).expand_as(var_nodes_encoding),
                var_nodes_encoding
            )
            variable_encoding = \
                var_nodes_encoding_sum \
                / tensor_dict['variable_mention_nums'].unsqueeze(-1)

        # restore variable encoding
        variable_encoding = \
            variable_encoding[
                tensor_dict['variable_encoding_restoration_indices']
            ] * tensor_dict[
                'variable_encoding_restoration_indices_mask'
            ].unsqueeze(-1)

        context_encoding = dict(
            tree_node_init_encoding=tree_node_init_encoding,
            packed_tree_node_encoding=tree_node_encoding,
            variable_encoding=variable_encoding,
        )

        context_encoding.update(tensor_dict)

        return context_encoding

    # noinspection PyUnboundLocalVariable
    @classmethod
    def to_packed_graph(cls,
                        asts: List[AbstractSyntaxTree],
                        connections: List,
                        init_with_seq_encoding: bool = False):
        # type: (...) -> Tuple[PackedGraph, Dict]
        packed_graph = PackedGraph(asts)
        max_variable_num = max(len(ast.variables) for ast in asts)

        node_adj_list = []
        terminal_nodes_adj_list = []
        master_node_adj_list = []
        var_master_nodes_adj_list = []
        var_usage_adj_list = []
        func_root_to_arg_adj_list = []
        # list of node ids of variable mentions
        var_node_ids = []
        # list of positions of variable encodings in the flattened version of
        # restored `variable_encoding`
        var_repr_flattened_positions = []
        var_node_variable_ids = []
        # list of number of mentions for each variable
        var_mention_nums = []

        use_variable_master_node = 'variable_master_nodes' in connections
        variable_repr_restoration_indices = \
            torch.zeros(len(asts), max_variable_num, dtype=torch.long)
        variable_repr_restoration_indices_mask = \
            torch.zeros(len(asts), max_variable_num)
        variable_cum_count = 0

        for ast_id, ast in enumerate(asts):
            for prev_node, succ_node in ast.adjacency_list:
                prev_node_packed_id = \
                    packed_graph.get_packed_node_id(ast_id, prev_node)
                succ_node_packed_id = \
                    packed_graph.get_packed_node_id(ast_id, succ_node)

                node_adj_list.append((
                    prev_node_packed_id,
                    succ_node_packed_id
                ))

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
                master_node_id = packed_graph.register_node(
                    ast_id,
                    'master_node',
                    group='master_nodes'
                )

                master_node_adj_list.extend([
                    (
                        master_node_id,
                        packed_graph.get_packed_node_id(ast_id, node)
                    )
                    for node in ast
                ])

            for i, (var_name, var_nodes) in enumerate(ast.variables.items()):
                var_node_num = len(var_nodes)
                # add data flow links
                if 'var_usage' in connections:
                    for _idx in range(len(var_nodes) - 1):
                        _var_node, _next_var_node = \
                            var_nodes[_idx], var_nodes[_idx + 1]
                        var_usage_adj_list.append((
                            packed_graph.get_packed_node_id(ast_id,
                                                            _var_node),
                            packed_graph.get_packed_node_id(ast_id,
                                                            _next_var_node)
                        ))

                if 'func_root_to_arg' in connections and var_nodes[0].is_arg:
                    func_root_node_id = \
                        packed_graph.get_packed_node_id(ast_id, ast.root)
                    for var_node in var_nodes:
                        func_root_to_arg_adj_list.append((
                            func_root_node_id,
                            packed_graph.get_packed_node_id(ast_id, var_node)
                        ))

                if use_variable_master_node:
                    var_master_node_id, node_id_in_group = \
                        packed_graph.register_node(
                            ast_id,
                            var_name,
                            group='variable_master_nodes',
                            return_node_index_in_group=True
                        )

                    var_master_nodes_adj_list.extend([(
                        var_master_node_id,
                        packed_graph.get_packed_node_id(ast_id, node))
                        for node in var_nodes
                    ])
                else:
                    var_node_packed_ids = [
                        packed_graph.get_packed_node_id(ast_id, node)
                        for node in var_nodes
                    ]
                    var_node_ids.extend(var_node_packed_ids)
                    var_node_variable_ids.extend(
                        [variable_cum_count] * var_node_num
                    )
                    var_mention_nums.append(var_node_num)

                variable_repr_restoration_indices[ast_id, i] = \
                    variable_cum_count
                var_repr_flattened_positions.append(
                    i + max_variable_num * ast_id
                )
                variable_cum_count += 1
            variable_repr_restoration_indices_mask[
                ast_id, :len(ast.variables)
            ] = 1.

        adj_lists = []
        if 'top_down' in connections:
            adj_lists.append(node_adj_list)
        if 'bottom_up' in connections:
            reversed_node_adj_list = [(n2, n1) for n1, n2 in node_adj_list]
            adj_lists.append(reversed_node_adj_list)
        if 'variable_master_nodes' in connections:
            reversed_var_master_nodes_adj_list = \
                [(n2, n1) for n1, n2 in var_master_nodes_adj_list]
            adj_lists.append(master_node_adj_list)
            adj_lists.append(reversed_var_master_nodes_adj_list)
        if 'var_usage' in connections:
            reversed_var_usage_adj_list = \
                [(n2, n1) for n1, n2 in var_usage_adj_list]
            adj_lists.append(var_usage_adj_list)
            adj_lists.append(reversed_var_usage_adj_list)
        if 'func_root_to_arg' in connections:
            adj_lists.append(func_root_to_arg_adj_list)
        if 'terminals' in connections:
            reversed_terminal_nodes_adj_list = \
                [(n2, n1) for n1, n2 in terminal_nodes_adj_list]
            adj_lists.append(terminal_nodes_adj_list)
            adj_lists.append(reversed_terminal_nodes_adj_list)
        if 'master_node' in connections:
            reversed_master_node_adj_list = \
                [(n2, n1) for n1, n2 in master_node_adj_list]
            adj_lists.append(master_node_adj_list)
            adj_lists.append(reversed_master_node_adj_list)

        adj_lists = [
            AdjacencyList(adj_list=adj_list, node_num=packed_graph.size)
            for adj_list in adj_lists
        ]

        max_tree_node_num = max(tree.size for tree in packed_graph.trees)
        tree_restoration_indices = torch.zeros(
            packed_graph.tree_num,
            max_tree_node_num,
            dtype=torch.long
        )
        tree_restoration_indices_mask = torch.zeros(
            packed_graph.tree_num,
            max_tree_node_num,
            dtype=torch.float
        )

        max_terminal_node_num = max(len(tree.terminal_nodes)
                                    for tree in packed_graph.trees)
        terminal_node_restoration_indices = torch.zeros(
            packed_graph.tree_num,
            max_terminal_node_num,
            dtype=torch.long
        )
        terminal_node_restoration_indices_mask = torch.zeros(
            packed_graph.tree_num,
            max_terminal_node_num,
            dtype=torch.float
        )

        for ast_id in range(packed_graph.tree_num):
            packed_node_ids = \
                list(packed_graph.node_groups[ast_id]['ast_nodes'].values())
            tree_restoration_indices[ast_id, :len(packed_node_ids)] = \
                torch.tensor(packed_node_ids)
            tree_restoration_indices_mask[ast_id, :len(packed_node_ids)] = 1.

            tree = packed_graph.trees[ast_id]
            terminal_node_ids = [
                packed_graph.get_packed_node_id(ast_id, node)
                for node in tree.terminal_nodes
            ]
            terminal_node_restoration_indices[
                ast_id, :len(terminal_node_ids)
            ] = torch.tensor(terminal_node_ids)
            terminal_node_restoration_indices_mask[
                ast_id, :len(terminal_node_ids)
            ] = 1.

        tensor_dict = {
            'adj_lists': adj_lists,
            'variable_encoding_restoration_indices':
                variable_repr_restoration_indices,
            'variable_encoding_restoration_indices_mask':
                variable_repr_restoration_indices_mask,
            'tree_restoration_indices': tree_restoration_indices,
            'tree_restoration_indices_mask': tree_restoration_indices_mask,
            'terminal_node_restoration_indices':
                terminal_node_restoration_indices,
            'terminal_node_restoration_indices_mask':
                terminal_node_restoration_indices_mask,
            'packed_graph_size': packed_graph.size
        }

        if init_with_seq_encoding:
            tensor_dict['var_repr_flattened_positions'] = \
                torch.tensor(var_repr_flattened_positions, dtype=torch.long)

        if use_variable_master_node:
            tensor_dict['variable_master_node_ids'] = \
                [_id for node, _id in
                 packed_graph.get_nodes_by_group('variable_master_nodes')]
        else:
            tensor_dict['variable_node_ids'] = torch.tensor(var_node_ids)
            tensor_dict['variable_node_variable_ids'] = \
                torch.tensor(var_node_variable_ids)
            tensor_dict['variable_mention_nums'] = \
                torch.tensor(var_mention_nums, dtype=torch.float)

        return packed_graph, tensor_dict

    @classmethod
    def to_tensor_dict(cls, packed_graph: PackedGraph,
                       grammar: Grammar,
                       vocab: Vocab) -> Dict[str, torch.Tensor]:
        obj_name_bpe_model = vocab.obj_name.subtoken_model
        obj_name_bpe_pad_idx = vocab.obj_name.subtoken_model.pad_id()

        # predefined index
        variable_master_node_type_idx = len(grammar.syntax_types)
        master_node_type_idx = variable_master_node_type_idx + 1

        node_syntax_type_indices = \
            torch.zeros(packed_graph.size, dtype=torch.long)
        var_node_name_indices = \
            torch.zeros(packed_graph.size, dtype=torch.long)
        tree_node_to_tree_id_map = \
            torch.zeros(packed_graph.size, dtype=torch.long)

        sub_tokens_list = []
        node_with_subtokens_indices = []
        node_type_tokens_list = [0 for _ in range(packed_graph.size)]

        for i, (ast_id, group, node_key) in \
                enumerate(packed_graph.nodes.values()):
            ast = packed_graph.trees[ast_id]

            node_type_tokens = []
            if group == 'ast_nodes':
                node = ast.id_to_node[node_key]
                type_idx = grammar.syntax_type_to_id[node.node_type]
                if node.is_variable_node:
                    var_node_name_indices[i] = vocab.source[node.old_name]

                # function root with type `block` also has an name entry
                # storing the name of the function
                if node.node_type == 'obj' \
                   or node.node_type == 'block' \
                   and hasattr(node, 'name'):
                    # compute variable embedding
                    node_sub_tokens = \
                        obj_name_bpe_model.encode_as_ids(node.name)
                    sub_tokens_list.append(node_sub_tokens)
                    node_with_subtokens_indices.append(i)

                if hasattr(node, 'type_tokens'):
                    node_type_tokens = [
                        vocab.grammar.variable_type_to_id(t)
                        for t in node.type_tokens
                    ]

            elif group == 'variable_master_nodes':
                type_idx = variable_master_node_type_idx
                var_node_name_indices[i] = vocab.source[node_key]
            elif group == 'master_nodes':
                type_idx = master_node_type_idx
            else:
                raise ValueError()

            tree_node_to_tree_id_map[i] = ast_id
            node_syntax_type_indices[i] = type_idx
            node_type_tokens_list[i] = node_type_tokens

        sub_tokens_indices = None
        if node_with_subtokens_indices:
            max_subtoken_num = max(len(x) for x in sub_tokens_list)
            sub_tokens_indices = np.zeros(
                (len(sub_tokens_list), max_subtoken_num),
                dtype=np.int64
            )
            sub_tokens_indices.fill(obj_name_bpe_pad_idx)
            for i, token_ids in enumerate(sub_tokens_list):
                sub_tokens_indices[i, :len(token_ids)] = token_ids

            sub_tokens_indices = torch.from_numpy(sub_tokens_indices)

        max_typetoken_num = max(len(x) for x in node_type_tokens_list)
        node_type_indices = np.zeros(
            (len(node_type_tokens_list), max_typetoken_num),
            dtype=np.int64
        )
        for i, type_ids in enumerate(node_type_tokens_list):
            node_type_indices[i, :len(type_ids)] = type_ids
        node_type_indices = torch.from_numpy(node_type_indices)

        return dict(
            batch_size=packed_graph.tree_num,
            tree_num=packed_graph.tree_num,
            node_syntax_type_indices=node_syntax_type_indices,
            node_type_indices=node_type_indices,
            var_node_name_indices=var_node_name_indices,
            node_with_subtokens_indices=torch.tensor(
                node_with_subtokens_indices,
                dtype=torch.long
            ),
            sub_tokens_indices=sub_tokens_indices,
            tree_node_to_tree_id_map=tree_node_to_tree_id_map
        )

    def get_batched_tree_init_encoding(self,
                                       tensor_dict: Dict[str, torch.Tensor]):
        # type: (...) -> torch.Tensor
        node_syntax_type_embedding = self.node_syntax_type_embedding(
            tensor_dict['node_syntax_type_indices']
        )
        node_type_embedding = \
            self.node_type_embedder(tensor_dict['node_type_indices'])
        node_content_embedding = self.var_node_name_embedding(
            tensor_dict['var_node_name_indices']) * \
            torch.ne(
                tensor_dict['var_node_name_indices'], 0.
            ).float().unsqueeze(-1)

        if tensor_dict['node_with_subtokens_indices'].size(0) > 0:
            obj_node_content_embedding = \
                self.node_content_embedder(tensor_dict['sub_tokens_indices'])
            node_content_embedding = node_content_embedding.scatter(
                0,
                tensor_dict[
                    'node_with_subtokens_indices'
                ].unsqueeze(-1).expand_as(obj_node_content_embedding),
                obj_node_content_embedding
            )

        tree_node_embedding = self.type_and_content_hybrid(
            torch.cat(
                [
                    node_syntax_type_embedding,
                    node_type_embedding,
                    node_content_embedding
                ],
                dim=-1
            )
        )

        return tree_node_embedding

    def unpack_encoding(self,
                        flattened_node_encodings: torch.Tensor,
                        batch_syntax_trees: List[AbstractSyntaxTree],
                        example_node2batch_node_map: Dict):
        batch_size = len(batch_syntax_trees)
        max_node_num = max(tree.size for tree in batch_syntax_trees)

        index = np.zeros((batch_size, max_node_num), dtype=np.int64)
        batch_tree_node_masks = \
            torch.zeros(batch_size, max_node_num, device=self.device)
        for e_id, syntax_tree in enumerate(batch_syntax_trees):
            example_nodes_with_batch_id = [
                (example_node_id, batch_node_id)
                for (_e_id, example_node_id), batch_node_id
                in example_node2batch_node_map.items()
                if _e_id == e_id
            ]
            sorted_example_nodes_with_batch_id = \
                sorted(example_nodes_with_batch_id, key=lambda t: t[0])
            example_nodes_batch_id = \
                [t[1] for t in sorted_example_nodes_with_batch_id]

            index[e_id, :len(example_nodes_batch_id)] = example_nodes_batch_id
            batch_tree_node_masks[e_id, :len(example_nodes_batch_id)] = 1.

        batch_node_encoding = flattened_node_encodings[
            torch.from_numpy(index).to(flattened_node_encodings.device)
        ]
        batch_node_encoding.data.masked_fill_(
            (1. - batch_tree_node_masks).byte().unsqueeze(-1), 0.
        )

        return dict(batch_tree_node_encoding=batch_node_encoding,
                    batch_tree_node_masks=batch_tree_node_masks)

    def get_decoder_init_state(self, context_encoding, config=None):
        # compute initial decoder's state via average pooling

        # (packed_graph_size, encoding_size)
        packed_tree_node_encoding = \
            context_encoding['packed_tree_node_encoding']

        tree_num = context_encoding['tree_num']
        total_node_num = context_encoding['tree_node_to_tree_id_map'].size(0)
        encoding_size = packed_tree_node_encoding.size(-1)
        zero_encoding = \
            packed_tree_node_encoding.new_zeros(tree_num, encoding_size)

        node_encoding_sum = zero_encoding.scatter_add_(
            0,
            context_encoding['tree_node_to_tree_id_map'].unsqueeze(-1).expand(
                -1,
                encoding_size
            ),
            packed_tree_node_encoding)
        tree_node_num = \
            packed_tree_node_encoding.new_zeros(tree_num).scatter_add_(
                0,
                context_encoding['tree_node_to_tree_id_map'],
                packed_tree_node_encoding.new_zeros(total_node_num).fill_(1.)
            )
        avg_node_encoding = node_encoding_sum / tree_node_num.unsqueeze(-1)

        c_0 = self.decoder_cell_init(avg_node_encoding)
        h_0 = torch.tanh(c_0)

        return h_0, c_0

    def get_attention_memory(self,
                             context_encoding,
                             att_target='terminal_nodes'):
        packed_tree_node_enc = \
            context_encoding['packed_tree_node_encoding']
        if att_target == 'ast_nodes':
            memory = packed_tree_node_enc[
                context_encoding['tree_restoration_indices']
            ]
            mask = context_encoding['tree_restoration_indices_mask']
        elif att_target == 'terminal_nodes':
            memory = packed_tree_node_enc[
                context_encoding['terminal_node_restoration_indices']
            ]
            mask = context_encoding['terminal_node_restoration_indices_mask']
        else:
            raise ValueError('unknown attention target')

        return memory, mask
