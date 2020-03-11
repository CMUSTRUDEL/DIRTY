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


class CollectTree(CFuncTree):
    """Collects a map of a set of addresses to a variable name.
    For each variable, this collects the addresses corresponding to its uses.
    """
    def __init__(self, user_locals, varmap):
        # List of user-defined locals in this function
        self.user_locals = user_locals
        self.varmap = varmap
        super().__init__()

    def collect_vars(self):
        rev_dict = defaultdict(set)
        for n in range(len(self.items)):
            item = self.items[n]
            if item.op is ida_hexrays.cot_var:
                name = get_expr_name(item.cexpr)
                score = item.cexpr.type.calc_score()
                if name in self.user_locals:
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
            if (addrs in self.varmap):
                print("collision")
                print(f"current: {self.varmap[addrs]}")
                print(f"new: {name}, score: {score}")
                self.varmap[addrs] = '::NONE::'
            else:
                self.varmap[addrs] = name


class Collector(ida_kernwin.action_handler_t):
    def __init__(self):
        # eas -> list of user defined locals
        self.fun_locals = defaultdict(list)
        # frozenset of addrs -> varname
        self.varmap = dict()
        ida_kernwin.action_handler_t.__init__(self)

    def dump_info(self):
        print(self.fun_locals)
        with open(os.environ['COLLECTED_VARS'], 'wb') as vars_fh, \
             open(os.environ['FUN_LOCALS'], 'wb') as locals_fh:
            pickle.dump(self.varmap, vars_fh)
            pickle.dump(self.fun_locals, locals_fh)
            vars_fh.flush()
            locals_fh.flush()

    def activate(self, ctx):
        print("Collecting vars.")
        for ea in Functions():
            f = idaapi.get_func(ea)
            cfunc = None
            try:
                cfunc = idaapi.decompile(f)
            except ida_hexrays.DecompilationFailure:
                continue
            if cfunc is None:
                continue
            cur_locals = [v.name for v in cfunc.get_lvars() \
                          if v.has_user_name and v.name != '']
            if cur_locals == []:
                continue
            self.fun_locals[ea] = cur_locals
            # Build decompilation tree
            ct = CollectTree(self.fun_locals[ea], self.varmap)
            tb = TreeBuilder(ct)
            tb.apply_to(cfunc.body, None)
            ct.collect_vars()
        print(f"{len(self.varmap)} vars collected in "
              f"{len(self.fun_locals)}/{len(list(Functions()))} functions.")
        if len(set(self.varmap.values())) > 0:
            print(f"{set(self.varmap.values())}")
        self.dump_info()
        return 1


ida_auto.auto_wait()
if not idaapi.init_hexrays_plugin():
    idaapi.load_plugin('hexrays')
    idaapi.load_plugin('hexx64')
    if not idaapi.init_hexrays_plugin():
        print("Unable to load Hex-rays")
        ida_pro.qexit(1)
    else:
        print(f"Hex-rays version {idaapi.get_hexrays_version()}")

cv = Collector()
cv.activate(None)
ida_pro.qexit(0)
