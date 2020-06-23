# Usage: IDALOG=/dev/stdout ./idat64 -B -S/path/to/collect.py /path/to/binary

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Optional, Set

import idaapi
from idautils import Functions
import ida_funcs
import ida_hexrays
import ida_kernwin
import pickle
import os

import decompiler.typeinfo as ti

from .function import Function
from .variable import Location, Stack, Register, Variable


class Collector(ida_kernwin.action_handler_t):
    """Generic class to collect information from a binary"""

    def __init__(self):
        # Load the type library
        try:
            with open(os.environ["TYPE_LIB"], "rb") as type_lib_file:
                self.type_lib = ti.TypeLibCodec.decode(type_lib_file.read())
        except Exception as e:
            print("Could not find type library, creating a new one")
            self.type_lib = ti.TypeLib()
        super().__init__(self)

    def write_type_lib(self) -> None:
        """Dumps the type library to the file specified by the environment variable
        `TYPE_LIB`.
        """
        with open(os.environ["TYPE_LIB"], "w") as type_lib_fh:
            type_lib_fh.write(ti.TypeLibCodec.encode(self.type_lib))
            type_lib_fh.flush()

    def collect_variables(
        self, variables: Iterable[ida_typeinf.tinfo_t],
    ) -> DefaultDict[Location, Set[Variable]]:
        """Collects Variables from a list of tinfo_ts and adds their types to the type
        library."""
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

    def activate(self, ctx) -> int:
        """Runs the collector"""
        raise NotImplementedError


class CollectDebug(Collector):
    """Class for collecting debug information"""

    def __init__(self):
        self.functions: Dict[int, Function] = dict()
        super().__init__(self)

    def write_functions(self) -> None:
        """Dumps the collected functions to the file specified by the environment
        variable `FUNCTIONS`.
        """
        with open(os.environ["FUNCTIONS"], "wb") as functions_fh:
            pickle.dump(self.functions, functions_fh)
            functions_fh.flush()

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

            arguments = self.collect_variables(cfunc.arguments)
            local_vars = self.collect_variables(
                [v for v in cfunc.get_lvars() if not v.is_arg_var]
            )
            self.functions[ea] = Function(
                name=name,
                return_type=return_type,
                arguments=arguments,
                local_vars=local_vars,
            )
        self.write_type_lib()
        self.write_functions()
        return 1


class CollectDecompiler(Collector):
    """Class for collecting decompiler-specific information"""

    def __init__(self):
        # Load the functions collected by CollectDebug
        with open(os.environ["FUNCTIONS"], "rb") as functions_fh:
            self.debug_functions: Dict[int, Function] = pickle.load(functions_fh)
        self.decompiler_functions: Dict[int, Function] = dict()
        super().__init__(self)

    def activate(self, ctx) -> int:
        """Collects types, user-defined variables, their locations in addition to the
        AST and raw code.
        """
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

            arguments = self.collect_variables(cfunc.arguments)
            local_vars = self.collect_variables(
                [v for v in cfunc.get_lvars() if not v.is_arg_var]
            )
            self.functions[ea] = Function(
                name=name,
                return_type=return_type,
                arguments=arguments,
                local_vars=local_vars,
            )
        self.dump_info()
        return 1
