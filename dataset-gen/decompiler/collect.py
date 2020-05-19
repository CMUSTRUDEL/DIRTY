# Usage: IDALOG=/dev/stdout ./idat64 -B -S/path/to/collect.py /path/to/binary

from collections import defaultdict
from util import UNDEF_ADDR, CFuncTree, CFuncTreeBuilder, get_expr_name
import idaapi
from idautils import Functions
import ida_auto
import ida_hexrays
import ida_kernwin
import ida_pro
import pickle
import os
import yaml


class CollectTree(CFuncTree):
    """Collects a map of a set of addresses to a variable name.
    For each variable, this collects the addresses corresponding to its uses.

    Attributes
    user_locals: List of names of user-defined locals in this function
    varmap: Dictionary mapping frozensets of addresses to variable names
    """

    def __init__(self, user_locals, varmap):
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
            if addrs in self.varmap:
                print("collision")
                print(f"current: {self.varmap[addrs]}")
                print(f"new: {name}, score: {score}")
                self.varmap[addrs] = "::NONE::"
            else:
                self.varmap[addrs] = name


class Collector(ida_kernwin.action_handler_t):
    def __init__(self):
        # eas -> list of user defined locals
        self.fun_locals = defaultdict(list)
        # frozenset of addrs -> varname
        self.varmap = dict()
        try:
            with open(os.environ["TYPE_DBASE"], "rb") as type_dbase:
                self.type_dbase = pickle.load(type_dbase)
        except Exception as e:
            print(e)
            self.type_dbase = defaultdict(set)
        ida_kernwin.action_handler_t.__init__(self)

    def dump_info(self):
        """Dumps the collected variables and function locals to the files
        specified by the environment variables `COLLECTED_VARS` and
        `FUN_LOCALS` respectively.
        """
        print(self.fun_locals)
        with open(os.environ["COLLECTED_VARS"], "wb") as vars_fh, open(
            os.environ["FUN_LOCALS"], "wb"
        ) as locals_fh, open(os.environ["TYPE_DBASE"], "wb") as type_dbase, open(
            "types.yaml", "w"
        ) as type_yaml:
            pickle.dump(self.varmap, vars_fh)
            pickle.dump(self.fun_locals, locals_fh)
            pickle.dump(self.type_dbase, type_dbase)
            yaml.dump(
                self.type_dbase, type_yaml, default_flow_style=False, allow_unicode=True
            )
            vars_fh.flush()
            locals_fh.flush()
            type_dbase.flush()
            type_yaml.flush()
            for size in sorted([s for s in self.type_dbase]):
                print(f"{size}: {self.type_dbase[size]}")

    def activate(self, ctx):
        """Runs the collector"""
        print("Collecting vars and types.")
        # `ea` is the start address of a single function
        for ea in Functions():
            f = idaapi.get_func(ea)
            cfunc = None
            try:
                cfunc = idaapi.decompile(f)
            except ida_hexrays.DecompilationFailure:
                # print(f"Could not decompile at address {ea:08x}")
                continue
            if cfunc is None:
                continue
            # For each variable, store its base type
            print(cfunc)
            print(
                f"frame size: {f.frsize:#04x}, stackoff delta: {cfunc.get_stkoff_delta():#04x}"
            )
            # Collect all the locations
            widths = dict()
            for v in cfunc.get_lvars():
                if v.is_stk_var():
                    corrected = v.get_stkoff() - cfunc.get_stkoff_delta()
                    bp_offset = f.frsize - corrected
                    # Store the location and width
                    widths[corrected] = v.width
                    print(
                        f"width: {v.width:#04x} offset: {v.get_stkoff():#04x} corrected: {corrected:#04x} bp_offset: {bp_offset:#04x} var: {v.type().dstr()} {v.name}"
                    )
                if v.type() and not v.type().is_funcptr():
                    cur_type = v.type().copy()
                    while cur_type.is_ptr_or_array() and not cur_type.is_pvoid():
                        cur_type.remove_ptr_or_array()
                    if cur_type.is_const():
                        cur_type.clr_const()
                    self.type_dbase[cur_type.get_size()].add(cur_type.dstr())
            # Divide up the locations into chunks
            locations = sorted(widths.keys())
            chunks = dict()
            previous_loc = None
            for loc in locations:
                if (
                    previous_loc is not None
                    and previous_loc + chunks[previous_loc] == loc
                ):
                    chunks[previous_loc] += widths[loc]
                else:
                    chunks[loc] = widths[loc]
                    previous_loc = loc

            for loc in sorted(chunks.keys()):
                print(f"loc: {loc:#04x}, size: {chunks[loc]:#04x}")
            cur_locals = [
                v.name for v in cfunc.get_lvars() if v.has_user_name and v.name != ""
            ]
            if cur_locals == []:
                continue
            self.fun_locals[ea] = cur_locals
            # Build decompilation tree
            ct = CollectTree(self.fun_locals[ea], self.varmap)
            tb = CFuncTreeBuilder(ct)
            tb.apply_to(cfunc.body, None)
            ct.collect_vars()
        # print(f"{len(self.varmap)} vars collected in "
        #       f"{len(self.fun_locals)}/{len(list(Functions()))} functions.")
        # if len(set(self.varmap.values())) > 0:
        #     print(f"{set(self.varmap.values())}")
        self.dump_info()
        return 1


ida_auto.auto_wait()
if not idaapi.init_hexrays_plugin():
    idaapi.load_plugin("hexrays")
    idaapi.load_plugin("hexx64")
    if not idaapi.init_hexrays_plugin():
        print("Unable to load Hex-rays")
        ida_pro.qexit(1)
    else:
        print(f"Hex-rays version {idaapi.get_hexrays_version()}")

cv = Collector()
cv.activate(None)
ida_pro.qexit(0)
