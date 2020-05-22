"""C Type Information

Encodes information about C types, and provides functions to serialize types.

The JSON that is output prioritizes compactness over readability.
Note that all sizes are in 8-bit bytes
"""
from json import JSONEncoder, dumps, loads
import typing as t


class Typeinfo:
    """Stores information about a type"""

    def __init__(self, *, name: t.Optional[str] = None, size: int):
        self.name = name
        self.size = size

    @classmethod
    def _from_json(cls, d):
        """Decodes from a dictionary"""
        return cls(name=d["n"], size=d["s"])

    def _to_json(self):
        """Encodes as JSON

        The 'T' field encodes which Typeinfo class is represented by this JSON:
            0: Typeinfo
            1: Array
            2: Pointer
            3: UDT.Field
            4: UDT.Padding
            5: Struct
            6: Union
            7: Void
        """
        return {"T": 0, "n": self.name, "s": self.size}

    def __eq__(self, other):
        if isinstance(other, Typeinfo):
            return self.name == other.name and self.size == other.size
        return False

    def __hash__(self):
        return hash((self.name, self.size))

    def __str__(self):
        return f"{self.name}"


class Array(Typeinfo):
    """Stores information about an array"""

    def __init__(self, *, base_type: Typeinfo, nelements: int):
        self.base_type = base_type
        self.nelements = nelements
        self.size = base_type.size * nelements

    @classmethod
    def _from_json(cls, d):
        return cls(base_type=d["b"], nelements=d["n"])

    def _to_json(self):
        return {
            "T": 1,
            "b": self.base_type._to_json(),
            "n": self.nelements,
        }

    def __eq__(self, other):
        if isinstance(other, Array):
            return (
                self.nelements == other.nelements and self.base_type == other.base_type
            )
        return False

    def __hash__(self):
        return hash((self.nelements, self.base_type))

    def __str__(self):
        return f"{self.base_type}[{self.nelements}]"


class Pointer(Typeinfo):
    """Stores information about a pointer.

    Note that the referenced type is by name because recursive data structures
    would recurse indefinitely.
    """

    WIDTH = 8

    def __init__(self, referenced_type_name: str):
        self.referenced_type_name = referenced_type_name
        self.size = Pointer.WIDTH

    @classmethod
    def _from_json(cls, d):
        return cls(referenced_type=d["t"])

    def _to_json(self):
        return {"T": 2, "t": self.referenced_type_name}

    def __eq__(self, other):
        if isinstance(other, Pointer):
            return self.referenced_type_name == other.referenced_type_name
        return False

    def __hash__(self):
        return hash(self.referenced_type_name)

    def __str__(self):
        return f"{str(self.referenced_type_name)} *"


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

        @classmethod
        def _from_json(cls, d):
            return cls(name=d["n"], typ=d["t"])

        def _to_json(self):
            return {"T": 3, "n": self.name, "t": self.typ._to_json()}

        def __eq__(self, other):
            if isinstance(other, UDT.Field):
                return self.name == other.name and self.typ == other.typ
            return False

        def __hash__(self):
            return hash((self.name, self.typ))

        def __str__(self):
            return f"{str(self.typ)} {self.name}"

    class Padding:
        """Padding bytes in a struct or union"""

        def __init__(self, size: int):
            self.size = size

        @classmethod
        def _from_json(cls, d):
            return cls(size=d["s"])

        def _to_json(self):
            return {"T": 4, "s": self.size}

        def __eq__(self, other):
            if isinstance(other, UDT.Padding):
                return self.size == other.size
            return False

        def __hash__(self):
            return self.size

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
        self.layout = tuple(layout)
        self.size = 0
        for l in layout:
            self.size += l.size

    @classmethod
    def _from_json(cls, d):
        return cls(name=d["n"], layout=d["l"])

    def _to_json(self):
        return {
            "T": 5,
            "n": self.name,
            "l": [l._to_json() for l in self.layout],
        }

    def __eq__(self, other):
        if isinstance(other, Struct):
            return self.name == other.name and self.layout == other.layout
        return False

    def __hash__(self):
        return hash((self.name, self.layout))

    def __str__(self):
        if self.name is None:
            ret = f"struct {{ "
        else:
            ret = f"struct {self.name} {{ "
        for l in self.layout:
            ret += f"{str(l)}; "
        ret += "}"
        return ret


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
        self.members = tuple(members)
        self.padding = padding
        # Set size to 0 if there are no members
        try:
            self.size = max(m.size for m in members)
        except ValueError:
            self.size = 0
        if self.padding is not None:
            self.size += self.padding.size

    @classmethod
    def _from_json(cls, d):
        return cls(name=d["n"], members=d["m"], padding=d["p"])

    def _to_json(self):
        return {
            "T": 6,
            "n": self.name,
            "m": [m._to_json() for m in self.members],
            "p": self.padding,
        }

    def __eq__(self, other):
        if isinstance(other, Union):
            return (
                self.name == other.name
                and self.members == other.members
                and self.padding == other.padding
            )
        return False

    def __hash__(self):
        return hash((self.name, self.members, self.padding))

    def __str__(self):
        if self.name is None:
            ret = f"union {{ "
        else:
            ret = f"union {self.name} {{ "
        for m in self.members:
            ret += f"{str(m)}; "
        if self.padding is not None:
            ret += f"{str(self.padding)}; "
        ret += "}"
        return ret


class Void(Typeinfo):
    size = 0

    def __init__(self):
        pass

    @classmethod
    def _from_json(cls, d):
        return cls()

    def _to_json(self):
        return {"T": 7}

    def __eq__(self, other):
        return isinstance(other, Void)

    def __hash__(self):
        return 0

    def __str__(self):
        return "void"


class TypeinfoCodec:
    """Encoder/decoder functions for Typeinfo"""

    @staticmethod
    def decode(encoded: str):
        """Decodes a JSON string"""

        def as_typeinfo(d):
            return {
                0: Typeinfo,
                1: Array,
                2: Pointer,
                3: UDT.Field,
                4: UDT.Padding,
                5: Struct,
                6: Union,
                7: Void,
            }[d["T"]]._from_json(d)

        return loads(encoded, object_hook=as_typeinfo)

    class _TypeEncoder(JSONEncoder):
        def default(self, t):
            if hasattr(t, "_to_json"):
                return t._to_json()
            return super().default(t)

    @staticmethod
    def encode(typeinfo: Typeinfo):
        """Encodes a Typeinfo as JSON"""
        # 'separators' removes spaces after , and : for efficiency
        return dumps(typeinfo, cls=TypeinfoCodec._TypeEncoder, separators=(",", ":"))
