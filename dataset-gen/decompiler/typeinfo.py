import typing as t


class Typeinfo:
    """Stores information about a type"""

    def __init__(self, *, name: t.Optional[str], size: int):
        self.name = name
        self.size = size

    def __eq__(self, other):
        if isinstance(other, Typeinfo):
            return self.name == other.name and self.size == other.size
        return False

    def __str__(self):
        return f"({self.size}) {self.name}"


class Field:
    """Information about a field in a struct or union"""

    def __init__(self, name: str, typ: Typeinfo):
        self.name = name
        self.typ = typ
        self.size = self.typ.size

    def __str__(self):
        return f"{self.typ.__str__()} {self.name}"


class Padding:
    """Padding bits in a struct or union"""

    def __init__(self, size: int):
        self.size = size

    def __str__(self):
        return f"({self.size}) PADDING"


class Struct(Typeinfo):
    """Stores information about a struct"""

    def __init__(
        self,
        *,
        name: t.Optional[str] = None,
        layout: t.Iterable[t.Union[Field, Padding, 'Struct', 'Union']],
    ):
        self.name = name
        self.layout = layout
        self.size = 0
        for l in layout:
            self.size += l.size

    def __str__(self):
        ret = f"({self.size}) struct {self.name} {{"
        for l in self.layout:
            ret += f"\n\t{l.__str__()}"
        ret += "\n}"
        return ret


class Union(Typeinfo):
    """Stores information about a union"""

    def __init__(
        self,
        *,
        name: t.Optional[str] = None,
        members: t.Iterable[t.Union[Field, Struct, 'Union']],
        padding: t.Optional[Padding] = None,
    ):
        self.name = name
        self.members = members
        self.padding = padding
        self.size = max(m.size for m in members)
        if self.padding is not None:
            self.size += self.padding.size

    def __str__(self):
        ret = f"({self.size}) union {self.name} {{"
        for m in self.members:
            ret += f"\n\t{m.__str__()}"
        if self.padding is not None:
            ret += f"\n\t{self.padding.__str__()}"
        ret += "\n}"
        return ret
