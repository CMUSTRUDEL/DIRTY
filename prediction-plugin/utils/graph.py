from collections import defaultdict, OrderedDict
from typing import List, Dict

from utils.ast import AbstractSyntaxTree, SyntaxNode


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

    def register_node(self,
                      tree_id,
                      node,
                      group='ast_nodes',
                      return_node_index_in_group=False):
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
            node_group = self.node_groups[i]
            for node, packed_node_id in node_group[group].items():
                yield node, packed_node_id

    @property
    def tree_num(self):
        return len(self.trees)
