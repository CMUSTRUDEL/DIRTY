import logging

try:
    import idaapi
except ModuleNotFoundError:
    logging.warning("unable to load [idaapi], stub loaded instead")
    from ._ida_stubs import idaapi  # type: ignore[no-redef] # noqa: F401

try:
    import ida_auto
except ModuleNotFoundError:
    logging.warning("unable to load [ida_auto], stub loaded instead")
    from ._ida_stubs import ida_auto  # type: ignore[no-redef] # noqa: F401

try:
    import ida_funcs
except ModuleNotFoundError:
    logging.warning("unable to load [ida_funcs], stub loaded instead")
    from ._ida_stubs import ida_funcs  # type: ignore[no-redef] # noqa: F401

try:
    import ida_hexrays
except ModuleNotFoundError:
    logging.warning("unable to load [ida_hexrays], stub loaded instead")
    from ._ida_stubs import ida_hexrays  # type: ignore[no-redef] # noqa: F401

try:
    import ida_kernwin
except ModuleNotFoundError:
    logging.warning("unable to load [ida_kernwin], stub loaded instead")
    from ._ida_stubs import ida_kernwin  # type: ignore[no-redef] # noqa: F401

try:
    import ida_lines
except ModuleNotFoundError:
    logging.warning("unable to load [ida_lines], stub loaded instead")
    from ._ida_stubs import ida_lines  # type: ignore[no-redef] # noqa: F401

try:
    import ida_pro
except ModuleNotFoundError:
    logging.warning("unable to load [ida_pro], stub loaded instead")
    from ._ida_stubs import ida_pro  # type: ignore[no-redef] # noqa: F401

try:
    import ida_typeinf
except ModuleNotFoundError:
    logging.warning("unable to load [ida_typeinf], stub loaded instead")
    from ._ida_stubs import ida_typeinf  # type: ignore[no-redef] # noqa: F401

try:
    import idautils
except ModuleNotFoundError:
    logging.warning("unable to load [idautils], stub loaded instead")
    from ._ida_stubs import idautils  # type: ignore[no-redef] # noqa: F401
