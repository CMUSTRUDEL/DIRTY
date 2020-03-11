# Usage: IDALOG=/dev/stdout ./idat64 -B -S/path/to/collect.py /path/to/binary

from collections import defaultdict
from util import UNDEF_ADDR, CFuncTree, TreeBuilder, \
    hexrays_vars, get_expr_name
import idaapi
from idautils import Functions
import ida_auto
import ida_hexrays
import ida_kernwin
import ida_pro
import pickle
import os

# frozenset of addrs -> varname
varmap = dict()


class CollectTree(CFuncTree):
    """Collects a map of a set of addresses to a variable name.
    For each variable, this collects the addresses corresponding to its uses.
    """
    def __init__(self, fun_locals):
        self.fun_locals = fun_locals
        super().__init__()

    def collect_vars(self, ea):
        # Don't care about this function if there aren't user-defined locals
        if ea not in self.fun_locals:
            return
        user_locals = self.fun_locals[ea]
        rev_dict = defaultdict(set)
        for n in range(len(self.items)):
            item = self.items[n]
            if item.op is ida_hexrays.cot_var:
                name = get_expr_name(item.cexpr)
                score = item.cexpr.type.calc_score()
                if name in user_locals:
                # if not hexrays_vars.match(name):
                    if item.ea != UNDEF_ADDR:
                        rev_dict[(name, score)].add(item.ea)
                    else:
                        ea = self.get_pred_ea(n)
                        if ea != UNDEF_ADDR:
                            rev_dict[(name, score)].add(ea)
        # ::NONE:: is a sentinel value used to indicate that two different
        # variables map to the same set of addresses. This happens in small
        # functions that use all of their arguments to call another function.
        for (name, score), addrs in rev_dict.items():
            addrs = frozenset(addrs)
            if (addrs in varmap):
                print("collision")
                print(f"current: {varmap[addrs]}")
                print(f"new: {name}, score: {score}")
                varmap[addrs] = '::NONE::'
            else:
                varmap[addrs] = name

class Collector(ida_kernwin.action_handler_t):
    def __init__(self):
        # eas -> list of user defined locals
        self.fun_locals = dict()
        ida_kernwin.action_handler_t.__init__(self)

    def dump_info(self):
        print(self.fun_locals)
        with open(os.environ['COLLECTED_VARS'], 'wb') as vars_fh, \
             open(os.environ['FUN_LOCALS'], 'wb') as locals_fh:
            pickle.dump(varmap, vars_fh)
            pickle.dump(self.fun_locals, locals_fh)
            vars_fh.flush()
            locals_fh.flush()

    def activate(self, ctx):
        def func(ea):
            f = idaapi.get_func(ea)
            if f is None:
                print("Please position the cursor within a function")
                return True
            cfunc = None
            try:
                cfunc = idaapi.decompile(f)
            except ida_hexrays.DecompilationFailure:
                pass

            if cfunc is None:
                # print(f"Failed to decompile {ea:x}!")
                return True
            cur_locals = [v.name for v in cfunc.get_lvars() if v.has_user_name]
            if cur_locals == [] or cur_locals == ['']:
                return True
            self.fun_locals[ea] = cur_locals
            # Build decompilation tree
            ct = CollectTree(self.fun_locals)
            tb = TreeBuilder(ct)
            tb.apply_to(cfunc.body, None)
            ct.collect_vars(ea)
        print("Collecting vars.")
        for ea in Functions():
            func(ea)
        print(f"{len(varmap)} vars collected in {len(self.fun_locals)}/{len(list(Functions()))} functions.")
        print(f"{set(varmap.values())}")
        self.dump_info()
        return 1

ida_auto.auto_wait()
if not idaapi.init_hexrays_plugin():
    idaapi.load_plugin('hexrays')
    idaapi.load_plugin('hexx64')
    if not idaapi.init_hexrays_plugin():
        print("Unable to load Hex-rays")
    else:
        print(f"Hex-rays version {idaapi.get_hexrays_version()}")


def main():
    cv = Collector()
    cv.activate(None)

main()
ida_pro.qexit(0)
