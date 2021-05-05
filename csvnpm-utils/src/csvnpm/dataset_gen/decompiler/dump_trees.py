import gzip
import os
import pickle
from typing import Dict, List

import jsonlines  # type: ignore
from csvnpm.binary.dire_types import TypeLib
from csvnpm.binary.function import CollectedFunction, Function
from csvnpm.binary.ida_ast import AST
from csvnpm.ida import ida_lines
from csvnpm.ida import idaapi as ida
from csvnpm.ida import idautils

from .collect import Collector


class CollectDecompiler(Collector):
    """Class for collecting decompiler-specific information"""

    def __init__(self):
        print("Initializing collect decompiler")
        super().__init__()
        print("Loading functions")
        # Load the functions collected by CollectDebug
        with open(os.environ["FUNCTIONS"], "rb") as functions_fh:
            self.debug_functions: Dict[int, Function] = pickle.load(functions_fh)
        print("Done")
        self.functions: List[CollectedFunction] = list()
        self.output_file_name = os.path.join(
            os.environ["OUTPUT_DIR"],
            "bins",
            os.environ["PREFIX"] + ".jsonl.gz",
        )

    def write_info(self) -> None:
        with gzip.open(self.output_file_name, "wt") as output_file:
            with jsonlines.Writer(output_file, compact=True) as writer:
                for cf in self.functions:
                    writer.write(cf.to_json())

    def activate(self, ctx) -> int:
        """Collects types, user-defined variables, their locations in addition to the
        AST and raw code.
        """
        print("Collecting vars and types.")
        for ea in (ea for ea in idautils.Functions() if ea in self.debug_functions):
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
            raw_code = ""
            for line in cfunc.get_pseudocode():
                raw_code += f"{' '.join(ida_lines.tag_remove(line.line).split())}\n"
            ast = AST(function=cfunc)
            decompiler = Function(
                ast=ast,
                name=name,
                return_type=return_type,
                arguments=arguments,
                local_vars=local_vars,
                raw_code=raw_code,
            )
            self.functions.append(
                CollectedFunction(
                    ea=ea,
                    debug=self.debug_functions[ea],
                    decompiler=decompiler,
                )
            )
        self.write_info()
        return 1


def main():
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
    print("Done with activate")
    ida.qexit(0)


if __name__ == "__main__":
    main()
