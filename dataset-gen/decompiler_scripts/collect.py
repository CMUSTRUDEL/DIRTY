# Usage: IDALOG=/dev/stdout ./idat64 -B -S/path/to/collect.py /path/to/binary

from collections import defaultdict
from util import UNDEF_ADDR, CFuncGraph, GraphBuilder, hexrays_vars, get_expr_name
import idaapi
import ida_hexrays
import ida_kernwin
import ida_pro
import ida_gdl
import pickle
import os

varmap = dict()                 # frozenset of addrs -> varname

# Collect a map of a set of addresses to a variable name.
# For each variable, this collects the addresses corresponding to its uses.
class CollectGraph(CFuncGraph):
    def collect_vars(self):
        rev_dict = defaultdict(set)
        for n in xrange(len(self.items)):
            item = self.items[n]
            if item.op is ida_hexrays.cot_var:
                name = get_expr_name(item.cexpr)
                if not hexrays_vars.match(name):
                    if item.ea != UNDEF_ADDR:
                        rev_dict[name].add(item.ea)
                    else:
                        ea = self.get_pred_ea(n)
                        if ea != UNDEF_ADDR:
                            rev_dict[name].add(ea)
        # ::NONE:: is a sentinel value used to indicate that two different
        # variables map to the same set of addresses. This happens in small
        # functions that use all of their arguments to call another function.
        for name, addrs in rev_dict.iteritems():
            addrs = frozenset(addrs)
            if (addrs in varmap):
                varmap[addrs] = '::NONE::'
            else:
                varmap[addrs] = name

def func(ea):
    f = idaapi.get_func(ea)
    if f is None:
        print('Please position the cursor within a function')
        return True
    cfunc = None
    try:
        cfunc = idaapi.decompile(f)
    except ida_hexrays.DecompilationFailure:
        pass

    if cfunc is None:
        print('Failed to decompile %x!' % ea)
        return True

    # Build decompilation graph
    cg = CollectGraph(None)
    gb = GraphBuilder(cg)
    gb.apply_to(cfunc.body, None)
    cg.collect_vars()

class custom_action_handler(ida_kernwin.action_handler_t):
    def __init__(self):
        ida_kernwin.action_handler_t.__init__(self)

class collect_vars(custom_action_handler):
    def activate(self, ctx):
        print('Collecting vars.')
        for ea in Functions():
            func(ea)
        print('Vars collected.')
        return 1

class dump_info(custom_action_handler):
    def activate(self, ctx):
        with open(os.environ['COLLECTED_VARS'], 'w') as vars_fh:
            pickle.dump(varmap, vars_fh)
            vars_fh.flush()
        return 1

idaapi.autoWait()
if not idaapi.init_hexrays_plugin():
    idaapi.load_plugin('hexrays')
    idaapi.load_plugin('hexx64')
    if not idaapi.init_hexrays_plugin():
        print('Unable to load Hex-rays')
    else:
        print('Hex-rays version %s has been detecetd' % idaapi.get_hexrays_version())

def main():
    cv = collect_vars()
    cv.activate(None)
    dv = dump_info()
    dv.activate(None)

main()
ida_pro.qexit(0)
