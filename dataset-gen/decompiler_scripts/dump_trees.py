from collections import defaultdict
from util import UNDEF_ADDR, CFuncTree, CFuncTreeBuilder, get_expr_name
import idaapi
import idautils
import ida_auto
import ida_funcs
import ida_hexrays
import ida_kernwin
import ida_pro
import jsonlines
import pickle
import os
import re

fun_locals = dict()
varmap = dict()
# Dictionary mapping variable ids to (orig, renamed) pairs
varnames = dict()
var_id = 0
sentinel_vars = re.compile('@@VAR_[0-9]+')


class RenamedTreeBuilder(CFuncTreeBuilder):
    def __init__(self, tree, func, addresses):
        self.func = func
        self.addresses = addresses
        super(RenamedTreeBuilder, self).__init__(tree)

    def visit_expr(self, e):
        global var_id
        if e.op is ida_hexrays.cot_var:
            # Save original name of variable
            original_name = get_expr_name(e)
            if not sentinel_vars.match(original_name):
                # Get new name of variable
                addresses = frozenset(self.addresses[original_name])
                if addresses in varmap and varmap[addresses] != '::NONE::':
                    new_name = varmap[addresses]
                else:
                    new_name = original_name
                # Save names
                varnames[var_id] = (original_name, new_name)
                score = e.type.calc_score()
                # Rename variables to @@VAR_[id]@@[orig name]@@[new name]
                self.func.get_lvars()[e.v.idx].name = \
                    f"@@VAR_{var_id}@@{original_name}@@{new_name}:new_{score}"
                var_id += 1
        return self.process(e)


class AddressCollector:
    def __init__(self, ct):
        self.ct = ct
        self.addresses = defaultdict(set)

    def collect(self):
        for item in self.ct.items:
            if item.op is ida_hexrays.cot_var:
                name = get_expr_name(item)
                if item.ea != UNDEF_ADDR:
                    self.addresses[name].add(item.ea)
                else:
                    item_id = [item_id for (i, item_id) in self.ct.reverse
                               if i == item][0]
                    # item_id = self.ct.reverse[item]
                    ea = self.ct.get_pred_ea(item_id)
                    if ea != UNDEF_ADDR:
                        self.addresses[name].add(ea)


def func(ea):
    """Process a single function given its EA"""
    f = idaapi.get_func(ea)
    function_name = ida_funcs.get_func_name(ea)
    if f is None:
        print("Please position the cursor within a function")

    cfunc = None
    try:
        cfunc = idaapi.decompile(f)
    except ida_hexrays.DecompilationFailure as e:
        print(f"Failed to decompile {ea:x}: {function_name}!")
        raise e

    # Rename decompilation tree
    ct = CFuncTree()
    tb = CFuncTreeBuilder(ct)
    tb.apply_to(cfunc.body, None)
    ac = AddressCollector(ct)
    ac.collect()
    rt = RenamedTreeBuilder(ct, cfunc, ac.addresses)
    rt.apply_to(cfunc.body, None)

    # Create tree from collected names
    function_info = dict()
    cfunc.build_c_tree()
    function_info["user_vars"] = fun_locals[ea]
    function_info["lvars"] = [v.name for v in cfunc.get_lvars() if v.name != '']
    new_tree = CFuncTree()
    new_builder = CFuncTreeBuilder(new_tree)
    new_builder.apply_to(cfunc.body, None)
    function_info["function"] = function_name
    function_info["ast"] = new_tree.json_tree(0)

    raw_code = ""
    for line in cfunc.get_pseudocode():
        raw_code += f'{idaapi.tag_remove(line.line)}\n'
    function_info["raw_code"] = raw_code
    return function_info


class custom_action_handler(ida_kernwin.action_handler_t):
    def __init__(self):
        ida_kernwin.action_handler_t.__init__(self)


class collect_vars(custom_action_handler):
    def activate(self, ctx):
        print('Collecting vars.')
        file_name = os.path.join(os.environ['OUTPUT_DIR'],
                                 os.environ['PREFIX'])
        jsonl_file_name = f"{file_name}.jsonl"
        with open(jsonl_file_name, 'w+') as jsonl_file:
            with jsonlines.Writer(jsonl_file) as writer:
                for ea in fun_locals:
                    try:
                        writer.write(func(ea))
                    except ida_hexrays.DecompilationFailure:
                        print("Decompilation failure")
                        continue
                    except ValueError as e:
                        print(e)
                        continue
        print('Vars collected.')
        return 1


def main():
    global renamed_prefix
    global varmap
    global fun_locals
    global varnames
    renamed_prefix = os.path.join(os.environ['OUTPUT_DIR'], 'functions',
                                  os.environ['PREFIX'])
    # Load collected variables and function locals
    with open(os.environ['COLLECTED_VARS'], 'rb') as vars_fh, \
         open(os.environ['FUN_LOCALS'], 'rb') as locals_fh:
        varmap = pickle.load(vars_fh)
        fun_locals = pickle.load(locals_fh)

    # Collect decompilation info
    cv = collect_vars()
    cv.activate(None)


ida_auto.auto_wait()
if not idaapi.init_hexrays_plugin():
    idaapi.load_plugin('hexrays')
    idaapi.load_plugin('hexx64')
    if not idaapi.init_hexrays_plugin():
        print("Unable to load Hex-rays")
    else:
        print(f"Hex-rays version {idaapi.get_hexrays_version()} detected")
main()
ida_pro.qexit(0)
