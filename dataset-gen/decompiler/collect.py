# Usage: IDALOG=/dev/stdout ./idat64 -B -S/path/to/collect.py /path/to/binary

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Optional, Set

import idaapi
from idautils import Functions
import ida_auto
import ida_funcs
import ida_hexrays
import ida_kernwin
import ida_pro
import ida_struct
import json
import pickle
import os
import yaml

import typeinfo as ti

from functioninfo import FunctionInfo
from varinfo import Location, Stack, Register, Variable


class Collector(ida_kernwin.action_handler_t):
    def __init__(self):
        self.functions: Dict[int, FunctionInfo] = dict()

        # Load the type library
        try:
            with open(os.environ["TYPE_LIB"], "rb") as type_lib_file:
                self.type_lib = ti.TypeLibCodec.decode(type_lib_file.read())
        except Exception as e:
            print("Could not find type library, creating a new one")
            self.type_lib = ti.TypeLib()
        ida_kernwin.action_handler_t.__init__(self)

    def dump_info(self) -> None:
        """Dumps the collected variables and function locals to the files
        specified by the environment variables `COLLECTED_VARS` and
        `FUN_LOCALS` respectively.
        """
        with open(os.environ["FUNCTIONS"], "wb") as functions_fh, open(
            os.environ["TYPE_LIB"], "w"
        ) as type_lib_fh:
            for ea, func in self.functions.items():
                print(f"{ea:x}: {func}")
            pickle.dump(self.functions, functions_fh)
            type_lib_fh.write(ti.TypeLibCodec.encode(self.type_lib))
            functions_fh.flush()
            type_lib_fh.flush()

    def activate(self, ctx) -> int:
        """Collects types, user-defined variables, and their locations"""
        print("Collecting vars and types.")
        # `ea` is the start address of a single function
        for ea in Functions():
            # Decompile
            f = idaapi.get_func(ea)
            cfunc = None
            try:
                cfunc = idaapi.decompile(f)
            except ida_hexrays.DecompilationFailure:
                continue
            if cfunc is None:
                continue

            # Function info
            name: str = ida_funcs.get_func_name(ea)

            self.type_lib.add_ida_type(cfunc.type.get_rettype())
            return_type = ti.TypeLib.parse_ida_type(cfunc.type.get_rettype())

            def collect_variables(
                variables: Iterable[ida_typeinf.tinfo_t],
            ) -> DefaultDict[Location, Set[Variable]]:
                """Collects Variables from a list of tinfo_ts"""
                collected_vars: DefaultDict[Location, Set[Variable]] = defaultdict(set)
                for v in variables:
                    if v.name == "" or not v.type():
                        continue
                    # Add all types to the typelib
                    self.type_lib.add_ida_type(v.type())
                    typ: ti.TypeInfo = ti.TypeLib.parse_ida_type(v.type())

                    loc: Location
                    if v.is_stk_var():
                        corrected = v.get_stkoff() - cfunc.get_stkoff_delta()
                        offset = f.frsize - corrected
                        loc = Stack(offset)
                    if v.is_reg_var():
                        loc = Register(v.get_reg1())
                    collected_vars[loc].add(
                        Variable(typ=typ, name=v.name, user=v.has_user_name)
                    )
                return collected_vars

            arguments = collect_variables(cfunc.arguments)
            local_vars = collect_variables(
                [v for v in cfunc.get_lvars() if not v.is_arg_var]
            )
            self.functions[ea] =  FunctionInfo(
                name=name,
                return_type=return_type,
                arguments=arguments,
                local_vars=local_vars,
            )
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
