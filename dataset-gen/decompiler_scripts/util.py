from collections import defaultdict
import ida_hexrays
import ida_lines
import ida_pro
import json

UNDEF_ADDR = 0xFFFFFFFFFFFFFFFF


def get_expr_name(expr):
    name = expr.print1(None)
    name = ida_lines.tag_remove(name)
    name = ida_pro.str2user(name)
    return name

class CFuncTree:
    def __init__(self):
        self.next_node_num = 0
        # list of citem_t
        self.items = []
        # (citem_t, node#) tuples
        self.reverse = []
        # dict of sets of next nodes
        self.succs = defaultdict(set)
        # previous node number (or None)
        self.pred = defaultdict(lambda: None)

    def size(self):
        return len(self.items)

    def add_node(self, i):
        cur_node_num = self.next_node_num
        self.next_node_num += 1
        self.items.append(i)
        self.reverse.append((i, cur_node_num))
        return cur_node_num

    def add_edge(self, src, dest):
        if self.pred[dest]:
            raise ValueError(
                f"Cannot add edge ({src} -> {dest}), "
                f"({self.pred[dest]} -> {dest}) exists"
            )
        self.pred[dest] = src
        self.succs[src].add(dest)

    def get_pred_ea(self, n):
        pred = self.pred[n]
        if pred:
            pred_item = self.items[pred]
            if pred_item.ea == UNDEF_ADDR:
                return self.get_pred_ea(pred)
            return pred_item.ea
        return UNDEF_ADDR

    def get_node_label(self, n):
        item = self.items[n]
        op = item.op
        insn = item.cinsn
        expr = item.cexpr
        parts = [ida_hexrays.get_ctype_name(op)]
        if op == ida_hexrays.cot_ptr:
            parts.append(f".{expr.ptrsize}")
        elif op == ida_hexrays.cot_memptr:
            parts.append(f".{expr.ptrsize} (m={expr.m})")
        elif op == ida_hexrays.cot_memref:
            parts.append(f" (m={expr.m})")
        elif op in [ida_hexrays.cot_obj, ida_hexrays.cot_var]:
            parts.append(f".{expr.refwidth} {get_expr_name(expr)}")
        elif op in [ida_hexrays.cot_num,
                    ida_hexrays.cot_helper,
                    ida_hexrays.cot_str]:
            parts.append(f" {get_expr_name(expr)}")
        elif op == ida_hexrays.cit_goto:
            parts.append(f" LABEL_{insn.cgoto.label_num}")
        elif op == ida_hexrays.cit_asm:
            parts.append("<asm statements; unsupported>")
        parts.append(f", ea: {item.ea:08x}")
        if item.is_expr() and expr is not None and not expr.type.empty():
            parts.append(", ")
            tstr = expr.type._print()
            parts.append(tstr if tstr else "?")
        return "".join(parts)


    # Takes a cexpr.type and returns info about it
    def serialize_type(self, t):
        info = { "name": t._print(),
                 "score": t.calc_score(),
        }
        if not t.is_decl_void():
            info["size"] = t.get_size()
            if t.get_size() != t.get_unpadded_size():
                info["unpadded_size"] = t.get_unpadded_size()
        if t.get_udt_nmembers() >= 0:
            info["udt_nmembers"] = t.get_udt_nmembers()
        return info

    def json_tree(self, n):
        """Puts the tree in a format suitable for JSON"""
        # Each node has a unique ID
        node_info = {"node_id": n}
        item = self.items[n]
        # This is the type of ctree node
        node_info["node_type"] = ida_hexrays.get_ctype_name(item.op)
        # This is the type of the data (in C-land)
        if item.is_expr() and not item.cexpr.type.empty():
            node_info["type"] = self.serialize_type(item.cexpr.type)
        node_info["address"] = f"{item.ea:08x}"
        if item.ea == UNDEF_ADDR:
            node_info["parent_address"] = f"{self.get_pred_ea(n):08x}"
        # Specific info for different node types
        if item.op == ida_hexrays.cot_ptr:
            node_info["pointer_size"] = item.cexpr.ptrsize
        elif item.op == ida_hexrays.cot_memptr:
            node_info.update({
                "pointer_size": item.cexpr.ptrsize,
                "m": item.cexpr.m
                })
        elif item.op == ida_hexrays.cot_memref:
            node_info["m"] = item.cexpr.m
        elif item.op == ida_hexrays.cot_obj:
            node_info.update({
                "name": get_expr_name(item.cexpr),
                "ref_width": item.cexpr.refwidth
            })
        elif item.op == ida_hexrays.cot_var:
            _, var_id, old_name, new_name = \
                get_expr_name(item.cexpr).split("@@")
            node_info.update({
                "var_id": var_id,
                "old_name": old_name,
                "new_name": new_name,
                "ref_width": item.cexpr.refwidth
            })
        elif item.op in [ida_hexrays.cot_num,
                         ida_hexrays.cot_str,
                         ida_hexrays.cot_helper]:
            node_info["name"] = get_expr_name(item.cexpr)
        # Get info for children of this node
        successors = set(self.succs[n])
        successor_trees = []
        if item.is_expr():
            to_remove = set()
            for s in successors:
                if item.x == self.items[s]:
                    to_remove.add(s)
                    node_info["x"] = self.json_tree(s)
                if item.y == self.items[s]:
                    to_remove.add(s)
                    node_info["y"] = self.json_tree(s)
                if item.z == self.items[s]:
                    to_remove.add(s)
                    node_info["z"] = self.json_tree(s)
            successors.difference_update(to_remove)
        if successors:
            for succ in successors:
                successor_trees.append(self.json_tree(succ))
        if successor_trees != []:
            node_info["children"] = successor_trees
        return node_info

    def print_tree(self):
        tree = json.dumps(self.json_tree(0))
        print(tree)

    def dump(self):
        print(f"{self.size()} items:")
        for n in range(self.size()):
            print(f"\t{n}: {ida_hexrays.get_ctype_name(self.items[n].op)}")

        print("pred:")
        for child in range (self.size()):
            print(f"\t{child}: {self.pred[child]}")

        print("succs:")
        for parent in range(self.size()):
            print(f"\t{parent}: {self.succs[parent]}")



class CFuncTreeBuilder(ida_hexrays.ctree_parentee_t):
    def __init__(self, tree):
        ida_hexrays.ctree_parentee_t.__init__(self)
        self.tree = tree

    def process(self, item):
        new_node_id = self.tree.add_node(item)
        if len(self.parents) > 1:
            parent_id = None
            parent_obj_id = self.parents.back().obj_id
            for item, node_id in self.tree.reverse:
                if item.obj_id == parent_obj_id:
                    parent_id = node_id
                    break
            if parent_id is not None:
                self.tree.add_edge(parent_id, new_node_id)
        return 0

    def visit_insn(self, i):
        return self.process(i)

    def visit_expr(self, e):
        return self.process(e)
