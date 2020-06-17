from collections import defaultdict
from typing import Dict, List

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

from functioninfo import FunctionInfo
from typeinfo import TypeLib, TypeLibCodec
from util import (
    UNDEF_ADDR,
    CFuncTree,
    CFuncTreeBuilder,
    get_expr_name,
    get_var_id,
    get_old_name,
    get_new_name,
)

class Collector(ida_kernwin.action_handler_t):
    def __init__(self):
        # Load function info
        with open(os.environ["FUNCTIONS"], "rb") as function_fh:
            self.functions: Dict[int, FunctionInfo] = pickle.load(function_fh)
        ida_kernwin.action_handler_t.__init__(self)

    def activate(self, ctx) -> int:
        """Renames variables"""
        print("Renaming variables.")

        # Decompile
        for ea, func_info in self.functions.items():
            # Decompile
            f = idaapi.get_func(ea)
            cfunc = None
            try:
                cfunc = idaapi.decompile(f)
            except ida_hexrays.DecompilationFailure:
                continue
            if cfunc is None:
                continue

            # Rename variables to keep track of:
            # - Argument or Local (A or L)
            # - Location
            # - Old name
            new_names: Dict[int, str]
            for idx, v in enumerate(cfunc.get_lvars()):
                tag = "A" if v.is_arg_var else "L"
                location: str
                if v.is_stk_var():
                    corrected = v.get_stkoff() - cfunc.get_stkoff_delta()
                    offset = f.frsize - corrected
                    location = f"S{offset}"
                if v.is_reg_var():
                    location = f"R{v.get_reg1()}"
                new_names[idx] = f"@@{tag}@@{location}@@{v.name}"
            for idx in to_rename.keys():
                cfunc.get_lvars()[idx].name = new_names[idx]


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
    rt = RenamedTreeBuilder(ct, cfunc)
    rt.apply_to(cfunc.body, None)

    # Create tree from collected names
    function_info = dict()
    # print("Vars (name, on stack?, offset, typed?, user name?, noptr?):")
    # print([(v.name, v.is_stk_var(), v.get_stkoff(), v.has_user_name, v.is_noptr_var) for v in cfunc.get_lvars()])
    cfunc.build_c_tree()
    function_info["user_vars"] = fun_locals[ea]
    function_info["lvars"] = {
        get_var_id(v.name): {
            "old": get_old_name(v.name),
            "new": get_new_name(v.name),
            "type": TypeLib.parse_ida_type(v.type()),
        }
        for v in cfunc.get_lvars()
        if v.name != ""
    }
    new_tree = CFuncTree()
    new_builder = CFuncTreeBuilder(new_tree)
    new_builder.apply_to(cfunc.body, None)
    function_info["function"] = function_name
    function_info["ast"] = new_tree.json_tree(0)
    raw_code = ""
    for line in cfunc.get_pseudocode():
        raw_code += f"{idaapi.tag_remove(line.line)}\n"
    function_info["raw_code"] = raw_code
    return function_info


ida_auto.auto_wait()
if not idaapi.init_hexrays_plugin():
    idaapi.load_plugin("hexrays")
    idaapi.load_plugin("hexx64")
    if not idaapi.init_hexrays_plugin():
        print("Unable to load Hex-rays")
    else:
        print(f"Hex-rays version {idaapi.get_hexrays_version()} detected")
cv = Collector()
cv.activate(None)
ida_pro.qexit(0)
