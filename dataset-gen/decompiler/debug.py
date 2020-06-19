# Usage: IDALOG=/dev/stdout ./idat64 -B -S/path/to/collect.py /path/to/binary

from collect import CollectDebug

ida_auto.auto_wait()
if not idaapi.init_hexrays_plugin():
    idaapi.load_plugin("hexrays")
    idaapi.load_plugin("hexx64")
    if not idaapi.init_hexrays_plugin():
        print("Unable to load Hex-rays")
        ida_pro.qexit(1)
    else:
        print(f"Hex-rays version {idaapi.get_hexrays_version()}")

debug = CollectDebug()
debug.activate(None)
ida_pro.qexit(0)
