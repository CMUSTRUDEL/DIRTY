"""C Type Information

Encodes information about C types, and provides functions to serialize types.
"""
from json import JSONEncoder
import typing as t


class Typeinfo:
    """Stores information about a type"""

    def __init__(self, *, name: t.Optional[str] = None, size: int):
        self.name = name
        self.size = size

    def _to_json(self):
        return {
            "__t": [],
            "name": self.name,
            "size": self.size
        }

    def __eq__(self, other):
        if isinstance(other, Typeinfo):
            return self.name == other.name and self.size == other.size
        return False

    def __str__(self):
        return f"{self.name}"


class Pointer(Typeinfo):
    """Stores information about a pointer"""

    WIDTH = 8

    def __init__(self, referenced_type: Typeinfo):
        self.referenced_type = referenced_type
        self.size = Pointer.WIDTH

    def _to_json(self):
        return {
            "__p": [],
            "referenced_type": self.referenced_type._to_json()
        }

    def __eq__(self, other):
        if isinstance(other, Pointer):
            return self.referenced_type == other.referenced_type
        return False

    def __str__(self):
        return f"{str(self.referenced_type)} *"


class Array(Typeinfo):
    """Stores information about an array"""

    def __init__(self, *, base_type: Typeinfo, nelements: int):
        self.base_type = base_type
        self.nelements = nelements
        self.size = base_type.size * nelements

    def _to_json(self):
        return {
            "__a": [],
            "base_type": self.base_type._to_json(),
            "nelements": self.nelements,
        }

    def __eq__(self, other):
        if isinstance(other, Array):
            return (
                self.nelements == other.nelements and self.base_type == other.base_type
            )
        return False

    def __str__(self):
        return f"{self.base_type}[{self.nelements}]"


class UDT(Typeinfo):
    """An object representing struct or union types"""

    def __init__(self):
        raise NotImplementedError

    class Field:
        """Information about a field in a struct or union"""

        def __init__(self, name: str, typ: Typeinfo):
            self.name = name
            self.typ = typ
            self.size = self.typ.size

        def _to_json(self):
            return {"__f": [], "name": self.name, "typ": self.typ._to_json()}

        def __str__(self):
            return f"{str(self.typ)} {self.name}"

    class Padding:
        """Padding bits in a struct or union"""

        def __init__(self, size: int):
            self.size = size

        def _to_json(self):
            return {"__p": [], "size": self.size}

        def __str__(self):
            return f"PADDING ({self.size})"


class Struct(UDT):
    """Stores information about a struct"""

    def __init__(
        self,
        *,
        name: t.Optional[str] = None,
        layout: t.Iterable[t.Union[UDT.Field, UDT.Padding, "Struct", "Union"]],
    ):
        self.name = name
        self.layout = layout
        self.size = 0
        for l in layout:
            self.size += l.size

    def _to_json(self):
        return {
            "__s": [],
            "name": self.name,
            "layout": [l._to_json() for l in self.layout],
        }

    def __str__(self):
        ret = f"struct {self.name} {{"
        for l in self.layout:
            ret += f"\n\t{str(l)}"
        ret += "\n}"
        return ret

    def __eq__(self, other):
        if isinstance(other, Struct):
            return self.name == other.name and self.layout == other.layout
        return False


class Union(UDT):
    """Stores information about a union"""

    def __init__(
        self,
        *,
        name: t.Optional[str] = None,
        members: t.Iterable[t.Union[UDT.Field, "Struct", "Union"]],
        padding: t.Optional[UDT.Padding] = None,
    ):
        self.name = name
        self.members = members
        self.padding = padding
        self.size = max(m.size for m in members)
        if self.padding is not None:
            self.size += self.padding.size

    def _to_json(self):
        return {
            "__u": [],
            "name": self.name,
            "members": [m._to_json() for m in self.members],
            "padding": self.padding,
        }

    def __str__(self):
        ret = f"union {self.name} {{"
        for m in self.members:
            ret += f"\n\t{str(m)}"
        if self.padding is not None:
            ret += f"\n\t{str(self.padding)}"
        ret += "\n}"
        return ret

    def __eq__(self, other):
        if isinstance(other, Union):
            return (
                self.name == other.name
                and self.members == other.members
                and self.padding == other.padding
            )
        return False


class TypeEncoder(JSONEncoder):
    def default(self, t):
        if hasattr(t, "_to_json"):
            return t._to_json()
        return super().default(t)
