"""Information about variables in a function"""

from typing import Any, Optional

from .typeinfo import TypeInfo


class Location:
    """A variable location"""

    pass


class Register(Location):
    """A register

    name: the name of the register
    """

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Register) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"Reg {self.name}"


class Stack(Location):
    """A location on the stack

    offset: the offset from the base pointer
    """

    def __init__(self, offset: int):
        self.offset = offset

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Stack) and self.offset == other.offset

    def __hash__(self) -> int:
        return hash(self.offset)

    def __repr__(self) -> str:
        return f"Stk 0x{self.offset:x}"


class Variable:
    """A variable

    typ: the type of the variable
    name: an optional user-defined name for the variable
    user: true if the name is user-defined
    """


    def __init__(self, typ: TypeInfo, name: str, user: bool):
        self.typ = typ
        self.name = name
        self.user = user

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Variable)
            and self.name == other.name
            and self.typ == other.typ
        )

    def __hash__(self):
        return hash((self.name, self.typ))

    def __repr__(self) -> str:
        name_source = "U" if self.user else "A"
        return f"{str(self.typ)} {self.name} ({name_source})"
