import gzip
import os
from collections import defaultdict
from typing import DefaultDict, Iterable, Optional, Set

from csvnpm.binary.dire_types import TypeInfo, TypeLib, TypeLibCodec
from csvnpm.binary.variable import Location, Register, Stack, Variable
from csvnpm.ida import idaapi as ida


class Collector(ida.action_handler_t):
    """Generic class to collect information from a binary"""

    def __init__(self):
        # Load the type library
        self.type_lib_file_name = os.path.join(
            os.environ["OUTPUT_DIR"],
            "types",
            os.environ["PREFIX"] + ".json.gz",
        )
        try:
            with gzip.open(self.type_lib_file_name, "rt") as type_lib_file:
                self.type_lib = TypeLibCodec.decode(type_lib_file.read())
        except Exception as e:
            print(e)
            print("Could not find type library, creating a new one")
            self.type_lib = TypeLib()
        super().__init__()

    def write_type_lib(self) -> None:
        """Dumps the type library to the file specified by the environment variable
        `TYPE_LIB`.
        """
        with gzip.open(self.type_lib_file_name, "wt") as type_lib_file:
            encoded = TypeLibCodec.encode(self.type_lib)
            type_lib_file.write(encoded)
            type_lib_file.flush()

    def collect_variables(
        self,
        frsize: int,
        stkoff_delta: int,
        variables: Iterable[ida.lvar_t],
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

            loc: Optional[Location] = None
            if v.is_stk_var():
                corrected = v.get_stkoff() - stkoff_delta
                offset = frsize - corrected
                loc = Stack(offset)
            if v.is_reg_var():
                loc = Register(v.get_reg1())
            if loc is not None:
                collected_vars[loc].add(
                    Variable(typ=typ, name=v.name, user=v.has_user_info)
                )
        return collected_vars

    def activate(self, ctx) -> int:
        """Runs the collector"""
        raise NotImplementedError
