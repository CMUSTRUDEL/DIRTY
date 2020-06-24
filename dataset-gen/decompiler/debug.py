# Usage: IDALOG=/dev/stdout ./idat64 -B -S/path/to/collect.py /path/to/binary

import idaapi as ida

from .collect import CollectDebug

ida.auto_wait()
if not ida.init_hexrays_plugin():
    ida.load_plugin("hexrays")
    ida.load_plugin("hexx64")
    if not ida.init_hexrays_plugin():
        print("Unable to load Hex-rays")
        ida.qexit(1)
    else:
        print(f"Hex-rays version {ida.get_hexrays_version()}")

debug = CollectDebug()
debug.activate(None)
ida.qexit(0)
