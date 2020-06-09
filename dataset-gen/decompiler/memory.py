"""C Memory Information

Encodes information about C memory.
"""
import typeinfo as ti
import typing as t


class Memory:
    """Represents memory"""

    def __init__(self, size: int):
        self.size = size


class Undefined(Memory):
    """Represents undefined memory"""

    pass


class Stack(Memory):
    """Represents a location on the stack"""

    def __init__(self, *, offset: int, size: int):
        self.offset = offset
        self.size = size


class Register(Memory):
    """Represents a register"""

    def __init__(self, *, name: t.Any, size: int):
        self.name = str(name)
        self.size = size
