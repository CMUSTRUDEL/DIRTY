"""Information about variables in a function"""

from typing import Any, Optional

from typeinfo import TypeInfo


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
    user_name: an optional user-defined name for the variable
    """

    class Name:
        """A name of a varaible."""

        def __init__(self, name):
            self._name = name

        @property
        def name(self) -> str:
            return self._name

    class Auto(Name):
        """An auto-generated name."""

        def __eq__(self, other: Any) -> bool:
            return isinstance(other, Variable.Auto) and self.name == other.name

        def __hash__(self) -> int:
            return hash(self.name)

        def __repr__(self):
            return f"{self.name} (A)"

    class User(Name):
        """A user-generated name."""

        def __eq__(self, other: Any) -> bool:
            return isinstance(other, Variable.User) and self.name == other.name

        def __hash__(self) -> int:
            return hash(self.name)

        def __repr__(self):
            return f"{self.name} (U)"

    def __init__(self, typ: TypeInfo, name: str, user: bool):
        self.typ = typ
        if user:
            self.name = Variable.User(name)
        else:
            self.name = Variable.Auto(name)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Variable)
            and self.name == other.name
            and self.typ == other.typ
        )

    def __hash__(self):
        return hash((self.name, self.typ))

    def __repr__(self) -> str:
        return f"{str(self.typ)} {self.name}"
