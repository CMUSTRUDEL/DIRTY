from typing import Iterable, Optional

from decompiler.typeinfo import Typeinfo, TypeinfoDecoder


class InvalidEdgeException(Exception):
    def __init__(self, l1: int, l2: int, msg: Optional[str] = None):
        self.l1 = l1
        self.l2 = l2
        if msg is None:
            self.msg = "Invalid edge"
        else:
            self.msg = msg

    def __str__(self):
        return f"{self.msg}: {self.l1:#04x} -> {self.l2:#04x}"


class Edge:
    """Represents an edge between two locations in memory"""

    def __init__(self, l1: int, l2: int):
        if l1 == l2:
            raise InvalidEdgeException(l1, l2)
        self.start = min(l1, l2)
        self.end = max(l1, l2)
        self.size = self.end - self.start
        self._addresses = None
        self._internal_addresses = None

    @property
    def addresses(self):
        """The set of addresses contained in this edge"""
        if self._addresses is None:
            self._addresses = set(range(self.start, self.end))
            self._addresses.add(self.end)
        return self._addresses

    @property
    def internal_addresses(self):
        """The set of addresses contained in this edge, excluding the start and
        end addresses"""
        if self._internal_addresses is None:
            self._internal_addresses = self.addresses.difference((self.start, self.end))
        return self._internal_addresses

    def __str__(self):
        return f"{self.start:#04x} -> {self.end:#04x}"


class Def(Edge):
    """An edge for a defined section of memory"""

    def __init__(self, l1: int, l2: int, typ: Typeinfo, varname: str):
        super().__init__(l1, l2)
        if self.size != typ.size:
            raise InvalidEdgeException(l1, l2, f"Size mismatch")
        self.typ = typ
        self.varname = varname

    def __eq__(self, other):
        if isinstance(other, Def):
            return (
                self.start == other.start
                and self.end == other.end
                and self.typ == other.typ
                and self.varname == other.varname
            )
        return False

    def __str__(self):
        return f"{super().__str__()} {self.typ} {self.varname}"


class Undef(Edge):
    """An edge for an undefined section of memory"""

    def __init__(self, l1: int, l2: int):
        super().__init__(l1, l2)

    def __eq__(self, other):
        if isinstance(other, Undef):
            return self.start == other.start and self.end == other.end
        return False


class InvalidMemoryException(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return f"{self.msg}"


class Memory:
    """Memory is defined as a set of edges between locations"""

    def __init__(self, edges: Iterable[Edge]):
        self.edges = edges
        self.check_edges()

    def check_edges(self):
        """Checks that the edges do not overlap"""
        nodes = set()
        for e in self.edges:
            if e.internal_addresses.intersection(nodes):
                raise InvalidMemoryException("Overlapping edges")
            nodes.update(e.addresses)

    def __str__(self):
        string = ""
        for edge in self.edges:
            string += f"{edge}\n"
        return string
