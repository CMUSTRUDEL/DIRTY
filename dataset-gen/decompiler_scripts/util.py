import ida_hexrays
import ida_lines
import ida_pro
import json
import re

UNDEF_ADDR = 0xFFFFFFFFFFFFFFFF

hexrays_vars = re.compile("^(v|a)[0-9]+$")


def get_expr_name(expr):
    name = expr.print1(None)
    name = ida_lines.tag_remove(name)
    name = ida_pro.str2user(name)
    return name


class CFuncGraph:
    def __init__(self, highlight):
        # list of citem_t
        self.items = []
        # (citem_t, node#) tuples
        self.reverse = []
        # list of lists of next nodes
        self.succs = []
        # list of lists of previous nodes
        self.preds = []
        self.highlight = highlight

    def nsucc(self, n):
        return len(self.succs[n]) if self.size() else 0

    def npred(self, n):
        return len(self.preds[n]) if self.size() else 0

    def succ(self, n, i):
        return self.succs[n][i]

    def pred(self, n, i):
        return self.preds[n][i]

    def size(self):
        return len(self.preds)

    def add_node(self):
        n = self.size()

        def resize(array, new_size):
            if new_size > len(array):
                while len(array) < new_size:
                    array.append([])
            else:
                array = array[:new_size]
            return array

        self.preds = resize(self.preds, n+1)
        self.succs = resize(self.succs, n+1)
        return n

    def add_edge(self, x, y):
        self.preds[y].append(x)
        self.succs[x].append(y)

    def get_pred_ea(self, n):
        if self.npred(n) == 1:
            pred = self.pred(n, 0)
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

    def json_tree(self, n):
        """Puts the tree in a format suitable for JSON"""
        # Each node has a unique ID
        node_info = {"node_id": n}
        item = self.items[n]
        # This is the type of ctree node
        node_info["node_type"] = ida_hexrays.get_ctype_name(item.op)
        # This is the type of the data (in C-land)
        if item.is_expr() and not item.cexpr.type.empty():
            node_info["type"] = item.cexpr.type._print()
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
        successors = []
        x_successor = None
        y_successor = None
        z_successor = None
        for i in range(self.nsucc(n)):
            successors.append(self.succ(n, i))
        successor_trees = []
        if item.is_expr():
            if item.x:
                for s in successors:
                    if item.x == self.items[s]:
                        successors.remove(s)
                        x_successor = self.json_tree(s)
                        break
            if item.y:
                for s in successors:
                    if item.y == self.items[s]:
                        successors.remove(s)
                        y_successor = self.json_tree(s)
                        break
            if item.z:
                for s in successors:
                    if item.z == self.items[s]:
                        successors.remove(s)
                        z_successor = self.json_tree(s)
                        break
        if successors:
            for succ in successors:
                successor_trees.append(self.json_tree(succ))
        if successor_trees != []:
            node_info["children"] = successor_trees
        if x_successor:
            node_info["x"] = x_successor
        if y_successor:
            node_info["y"] = y_successor
        if z_successor:
            node_info["z"] = z_successor
        return node_info

    def print_tree(self):
        tree = json.dumps(self.json_tree(0))
        print(tree)

    def dump(self):
        print(f"{len(self.items)} items:")
        for idx, item in enumerate(self.items):
            print(f"\t{idx}: {ida_hexrays.get_ctype_name(item.op)}")

        print("succs:")
        for parent, s in enumerate(self.succs):
            print(f"\t{parent}: {s}")

        print("preds:")
        for child, p in enumerate(self.preds):
            print(f"\t{child}: {p}")


class GraphBuilder(ida_hexrays.ctree_parentee_t):
    def __init__(self, cg):
        ida_hexrays.ctree_parentee_t.__init__(self)
        self.cg = cg

    def add_node(self, i):
        n = self.cg.add_node()
        if n <= len(self.cg.items):
            self.cg.items.append(i)
        self.cg.items[n] = i
        self.cg.reverse.append((i, n))
        return n

    def process(self, i):
        n = self.add_node(i)
        if n < 0:
            return n
        if len(self.parents) > 1:
            lp = self.parents.back().obj_id
            for k, v in self.cg.reverse:
                if k.obj_id == lp:
                    p = v
                    break
            self.cg.add_edge(p, n)
        return 0

    def visit_insn(self, i):
        return self.process(i)

    def visit_expr(self, e):
        return self.process(e)
