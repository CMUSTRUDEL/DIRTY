from collections import defaultdict

from typing import DefaultDict, Mapping, Optional, Set

from typeinfo import TypeInfo
from varinfo import Location, Variable


class FunctionInfo:
    """Holds information about a C function

    name: name of the function
    return_type: return type of the function
    arguments: list of arguments to the function
    local_vars: list of local variables
    """

    def __init__(
        self,
        *,
        name: str,
        return_type: TypeInfo,
        arguments: Mapping[Location, Set[Variable]],
        local_vars: Mapping[Location, Set[Variable]],
    ):
        self._name = name
        self._return_type = return_type
        self._arguments = defaultdict(set, arguments)
        self._local_vars = defaultdict(set, local_vars)

    @property
    def arguments(self) -> DefaultDict[Location, Set[Variable]]:
        return self._arguments

    @property
    def has_user_names(self) -> bool:
        arg_vars = (v for vs in self.arguments.values() for v in vs)
        local_vars = (v for vs in self.local_vars.values() for v in vs)
        return any(v.user_name for v in arg_vars) or any(v.user_name for v in local_vars)

    @property
    def local_vars(self) -> DefaultDict[Location, Set[Variable]]:
        return self._local_vars

    @property
    def locations(self) -> Set[Location]:
        return set(self.arguments.keys()) + set(self.local_vars.keys())

    @property
    def name(self) -> str:
        return self._name

    @property
    def return_type(self) -> TypeInfo:
        return self._return_type

    def __repr__(self) -> str:
        return (
            f"{self.return_type} {self.name}\n"
            f"    Arguments:  {dict(self.arguments)}\n"
            f"    Local vars: {dict(self.local_vars)}"
        )
