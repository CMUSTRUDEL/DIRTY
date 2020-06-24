from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Optional, Set

import idaapi as ida
from idautils import Functions
import pickle
import os

from decompiler.function import Function
from decompiler.typeinfo import TypeInfo, TypeLib, TypeLibCodec
from decompiler.variable import Location, Stack, Register, Variable


class Collector(ida.action_handler_t):
    """Generic class to collect information from a binary"""

    def __init__(self):
        # Load the type library
        try:
            with open(os.environ["TYPE_LIB"], "rb") as type_lib_file:
                self.type_lib = TypeLibCodec.decode(type_lib_file.read())
        except Exception as e:
            print("Could not find type library, creating a new one")
            self.type_lib = TypeLib()
        super().__init__(self)

    def write_type_lib(self) -> None:
        """Dumps the type library to the file specified by the environment variable
        `TYPE_LIB`.
        """
        with open(os.environ["TYPE_LIB"], "w") as type_lib_fh:
            type_lib_fh.write(TypeLibCodec.encode(self.type_lib))
            type_lib_fh.flush()

    def collect_variables(
        self, frsize: int, stkoff_delta: int, variables: Iterable[ida.lvar_t],
    ) -> DefaultDict[Location, Set[Variable]]:
        """Collects Variables from a list of tinfo_ts and adds their types to the type
        library."""
        collected_vars: DefaultDict[Location, Set[Variable]] = defaultdict(set)
        for v in variables:
            if v.name == "" or not v.type():
                continue
            # Add all types to the typelib
            self.type_lib.add_ida_type(v.type())
            typ: TypeInfo = TypeLib.parse_ida_type(v.type())

            loc: Location
            if v.is_stk_var():
                corrected = v.get_stkoff() - stkoff_delta
                offset = frsize - corrected
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
