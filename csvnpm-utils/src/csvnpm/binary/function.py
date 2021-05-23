from collections import defaultdict
from json import dumps
from typing import DefaultDict, Mapping, Optional, Set, Tuple

from csvnpm.binary.dire_types import TypeInfo, TypeLibCodec
from csvnpm.binary.ida_ast import AST
from csvnpm.binary.variable import Location, Stack, Variable, location_from_json_key


class Function:
    """Holds information about a C function

    name: name of the function
    return_type: return type of the function
    arguments: list of arguments to the function
    local_vars: list of local variables
    """

    def __init__(
        self,
        *,
        ast: Optional[AST] = None,
        name: str,
        return_type: TypeInfo,
        arguments: Mapping[Location, Set[Variable]],
        local_vars: Mapping[Location, Set[Variable]],
        raw_code: Optional[str] = None,
    ):
        self._ast = ast
        self._name = name
        self._return_type = return_type
        self._arguments = defaultdict(set, arguments)
        self._local_vars = defaultdict(set, local_vars)
        self._raw_code = raw_code

    def to_json(self):
        ast = self.ast.to_json() if self.ast else None
        arguments = dict()
        for key, args in self.arguments.items():
            arguments[key.json_key()] = [arg.to_json() for arg in args]
        local_vars = dict()
        for key, locs in self.local_vars.items():
            local_vars[key.json_key()] = [loc.to_json() for loc in locs]
        return {
            "t": ast,
            "n": self.name,
            "r": self.return_type._to_json(),
            "a": arguments,
            "l": local_vars,
            "c": self.raw_code,
        }

    @classmethod
    def from_json(cls, d):
        ast = AST.from_json(d["t"]) if d["t"] else None
        return_type = TypeLibCodec.decode(dumps(d["r"]))
        arguments = dict()
        for key, args in d["a"].items():
            arguments[location_from_json_key(key)] = {
                Variable.from_json(arg) for arg in args
            }
        local_vars = dict()
        for key, locs in d["l"].items():
            local_vars[location_from_json_key(key)] = {
                Variable.from_json(loc) for loc in locs
            }
        return cls(
            ast=ast,
            name=d["n"],
            return_type=return_type,
            arguments=arguments,
            local_vars=local_vars,
            raw_code=d["c"],
        )

    @property
    def ast(self) -> Optional[AST]:
        return self._ast

    @property
    def arguments(self) -> DefaultDict[Location, Set[Variable]]:
        return self._arguments

    @property
    def has_user_names(self) -> bool:
        arg_vars = (v for vs in self.arguments.values() for v in vs)
        local_vars = (v for vs in self.local_vars.values() for v in vs)
        return any(v.user for v in arg_vars) or any(v.user for v in local_vars)

    @property
    def local_vars(self) -> DefaultDict[Location, Set[Variable]]:
        return self._local_vars

    @property
    def locations(self) -> Set[Location]:
        return set(self.arguments.keys()).union(set(self.local_vars.keys()))

    @property
    def name(self) -> str:
        return self._name

    @property
    def return_type(self) -> TypeInfo:
        return self._return_type

    @property
    def raw_code(self) -> Optional[str]:
        return self._raw_code

    @staticmethod
    def stack_layout(vars) -> Tuple[Tuple[int, ...], Tuple[int, ...], bool]:
        """Returns the layout of the stack as a pair of tuples. The first
        tuple is the accessible offsets on the stack, while the second
        are the start offsets of data in the stack.

        This is useful for using with a TypeLib to get the next replacement
        types. For example, if you have a TypeLib in the variable `lib` and
        you want to get the list of next possible retypings for a function in
        the variable `f`, you would use:

        accessible, starts = f.stack_layout()
        lib.get_next_replacements(accessible, starts)
        """
        # List of tuples of (offset, TypeInfo)
        accessible = []
        starts = []
        for loc in vars:
            if isinstance(loc, Stack):
                for v in vars[loc]:
                    accessible += [
                        loc.offset - v.typ.size + acc
                        for acc in v.typ.accessible_offsets()
                    ]
                    starts += [
                        loc.offset - v.typ.size + start
                        for start in v.typ.start_offsets()
                    ]
        return (
            tuple(sorted(set(accessible))),
            tuple(sorted(set(starts))),
            len(accessible) != len(set(accessible)) or len(starts) != len(set(starts)),
        )

    def __repr__(self) -> str:
        ret = (
            f"{self.return_type} {self.name}\n"
            f"\tArguments: {dict(self.arguments)}\n"
            f"\tLocal vars: {dict(self.local_vars)}"
        )
        if self.ast:
            ret += f"\n\tAST: {self.ast}"
        if self.raw_code:
            ret += f"\n\tRaw code: {self.raw_code}"
        return ret


class CollectedFunction:
    """Collected information about a single function. Has both debug and
    decompiler-generated data.
    """

    def __init__(self, *, ea: int, debug: Function, decompiler: Function):
        self.name: str = debug.name
        self.ea = ea
        self.debug = debug
        self.decompiler = decompiler

    def to_json(self):
        return {
            "e": self.ea,
            "b": self.debug.to_json(),
            "c": self.decompiler.to_json(),
        }

    @classmethod
    def from_json(cls, d):
        debug = Function.from_json(d["b"])
        decompiler = Function.from_json(d["c"])
        return cls(ea=d["e"], debug=debug, decompiler=decompiler)

    def __repr__(self):
        return (
            f"{self.ea} {self.name}\n"
            f"Debug: {self.debug}\n"
            f"Decompiler: {self.decompiler}\n"
        )
