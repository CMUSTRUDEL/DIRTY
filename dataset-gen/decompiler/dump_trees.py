import idaapi as ida

import os
import pickle

import idaapi as ida
from idautils import Functions

from collect import Collector
from function import Function
from typeinfo import TypeLib


class CollectDecompiler(Collector):
    """Class for collecting decompiler-specific information"""

    def __init__(self):
        # Load the functions collected by CollectDebug
        with open(os.environ["FUNCTIONS"], "rb") as functions_fh:
            self.debug_functions: Dict[int, Function] = pickle.load(functions_fh)
        self.decompiler_functions: Dict[int, Function] = dict()
        super().__init__()

    # FIXME
    def write_info(self) -> None:
        pass

    def activate(self, ctx) -> int:
        """Collects types, user-defined variables, their locations in addition to the
        AST and raw code.
        """
        print("Collecting vars and types.")
        # `ea` is the start address of a single function
        for ea in Functions():
            # Decompile
            f = ida.get_func(ea)
            cfunc = None
            try:
                cfunc = ida.decompile(f)
            except ida.DecompilationFailure:
                continue
            if cfunc is None:
                continue

            # Function info
            name: str = ida.get_func_name(ea)

            self.type_lib.add_ida_type(cfunc.type.get_rettype())
            return_type = TypeLib.parse_ida_type(cfunc.type.get_rettype())

            arguments = self.collect_variables(
                f.frsize, cfunc.get_stkoff_delta(), cfunc.arguments
            )
            local_vars = self.collect_variables(
                f.frsize,
                cfunc.get_stkoff_delta(),
                [v for v in cfunc.get_lvars() if not v.is_arg_var],
            )
            self.decompiler_functions[ea] = Function(
                name=name,
                return_type=return_type,
                arguments=arguments,
                local_vars=local_vars,
            )
        self.write_info()
        return 1


ida.auto_wait()
if not ida.init_hexrays_plugin():
    ida.load_plugin("hexrays")
    ida.load_plugin("hexx64")
    if not ida.init_hexrays_plugin():
        print("Unable to load Hex-rays")
        ida.qexit(1)
    else:
        print(f"Hex-rays version {ida.get_hexrays_version()}")

decompiler = CollectDecompiler()
decompiler.activate(None)
ida.qexit(0)
