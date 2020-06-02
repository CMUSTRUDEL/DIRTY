"""C Type Information

Encodes information about C types, and provides functions to serialize types.

The JSON that is output prioritizes compactness over readability.
Note that all sizes are in 8-bit bytes
"""
from collections import defaultdict
from json import JSONEncoder, dumps, loads
import typing as t

try:
    import ida_typeinf
except ImportError:
    print("Warning: could not import ida_typeinf. TypeLib.add() will not work")


class TypeLib:
    """A library of types.

    Allows access to types by size.

    The usual dictionary magic methods are implemented, allowing for
    dictionary-like access to TypeInfo.
    """

    def __init__(self, data: t.Optional[t.DefaultDict[str, t.Set["TypeInfo"]]] = None):
        if data is None:
            self._data: t.DefaultDict[str, t.Set["TypeInfo"]] = defaultdict(set)
        else:
            self._data = data

    def add_ida_type(
        self, typ: "ida_typeinf.tinfo_t", worklist: t.Optional[t.Set[str]] = None
    ) -> None:
        """Adds an element to the TypeLib by parsing an IDA tinfo_t object"""

        if worklist is None:
            worklist = set()
        if typ.dstr() in worklist or typ.is_void():
            return
        worklist.add(typ.dstr())
        if typ.is_funcptr() or "(" in typ.dstr():
            new_type = FunctionPointer(name=typ.dstr())
        elif typ.is_decl_ptr():
            new_type = Pointer(typ.get_pointed_object().dstr())
            self.add_ida_type(typ.get_pointed_object(), worklist)
        elif typ.is_array():
            self.add_ida_type(typ.get_array_element(), worklist)
            # To get array type info, first create an
            # array_type_data_t then call get_array_details to
            # populate it. Unions and structs follow a similar
            # pattern.
            array_info = ida_typeinf.array_type_data_t()
            typ.get_array_details(array_info)
            nelements = array_info.nelems
            element_size = array_info.elem_type.get_size()
            element_type = array_info.elem_type.dstr()
            new_type = Array(
                nelements=nelements,
                element_size=element_size,
                element_type=element_type,
            )
        elif typ.is_udt():
            udt_info = ida_typeinf.udt_type_data_t()
            typ.get_udt_details(udt_info)
            name = typ.dstr()
            size = udt_info.total_size
            nmembers = typ.get_udt_nmembers()
            if typ.is_union():
                members = []
                largest_size = 0
                for n in range(nmembers):
                    member = ida_typeinf.udt_member_t()
                    # To get the nth member set OFFSET to n and tell find_udt_member
                    # to search by index.
                    member.offset = n
                    typ.find_udt_member(member, ida_typeinf.STRMEM_INDEX)
                    largest_size = max(largest_size, member.size)
                    type_name = member.type.dstr()
                    self.add_ida_type(member.type, worklist)
                    members.append(
                        UDT.Field(
                            name=member.name, size=member.size, type_name=type_name
                        )
                    )
                end_padding = size - (largest_size // 8)
                if end_padding > 0:
                    new_type = Union(
                        name=name, members=members, padding=UDT.Padding(end_padding),
                    )
                else:
                    new_type = Union(name=name, members=members)
            elif typ.is_struct():
                layout: t.List[t.Union[UDT.Member, "Struct", "Union"]] = []
                next_offset = 0
                for n in range(nmembers):
                    member = ida_typeinf.udt_member_t()
                    member.offset = n
                    typ.find_udt_member(member, ida_typeinf.STRMEM_INDEX)
                    # Check for padding. Careful, because offset and
                    # size are in bits, not bytes.
                    if member.offset != next_offset:
                        layout.append(UDT.Padding((member.offset - next_offset) // 8))
                    next_offset = member.offset + member.size
                    type_name = member.type.dstr()
                    self.add_ida_type(member.type, worklist)
                    layout.append(
                        UDT.Field(
                            name=member.name, size=member.size, type_name=type_name
                        )
                    )
                # Check for padding at the end
                end_padding = size - next_offset // 8
                if end_padding > 0:
                    layout.append(UDT.Padding(end_padding))
                new_type = Struct(name=name, layout=layout)
        else:
            new_type = TypeInfo(name=typ.dstr(), size=typ.get_size())
        self._data[new_type.size].add(new_type)

    @classmethod
    def _from_json(cls, d):
        data = defaultdict(set)
        for (size, types) in d.items():
            decoded_types = set()
            for t in types:
                decoded_types.add(TypeInfoCodec.decode(v))
            data[k] = decoded_types
        return cls(data)

    def _to_json(self):
        return dict(self._data)

    def __contains__(self, key: int) -> bool:
        return key in self._data

    def __getitem__(self, key: int) -> t.Set["TypeInfo"]:
        return self._data[key]

    def __setitem__(self, key: int, item: t.Set["TypeInfo"]) -> None:
        self._data[key] = item

    def __str__(self) -> str:
        ret = ""
        for n in sorted(self._data.keys()):
            ret += f"{n}: ["
            for t in self._data[n]:
                ret += f"{t}, "
            ret += f"]\n"
        return ret


class TypeInfo:
    """Stores information about a type"""

    def __init__(self, *, name: t.Optional[str], size: int):
        self.name = name
        self.size = size

    @classmethod
    def _from_json(cls, d):
        """Decodes from a dictionary"""
        return cls(name=d["n"], size=d["s"])

    def _to_json(self):
        """Encodes as JSON

        The 'T' field encodes which TypeInfo class is represented by this JSON:
            0: TypeInfo
            1: Array
            2: Pointer
            3: UDT.Field
            4: UDT.Padding
            5: Struct
            6: Union
            7: Void
            8: Function Pointer
        """
        return {"T": 0, "n": self.name, "s": self.size}

    def __eq__(self, other):
        if isinstance(other, TypeInfo):
            return self.name == other.name and self.size == other.size
        return False

    def __hash__(self):
        return hash((self.name, self.size))

    def __str__(self):
        return f"{self.name}"


class Array(TypeInfo):
    """Stores information about an array"""

    def __init__(self, *, nelements: int, element_size: int, element_type: str):
        self.element_type = element_type
        self.element_size = element_size
        self.nelements = nelements
        self.size = element_size * nelements

    @classmethod
    def _from_json(cls, d):
        return cls(nelements=d["n"], element_size=d["s"], element_type=d["t"])

    def _to_json(self):
        return {
            "T": 1,
            "n": self.nelements,
            "s": self.element_size,
            "t": self.element_type,
        }

    def __eq__(self, other):
        if isinstance(other, Array):
            return (
                self.nelements == other.nelements
                and self.element_size == other.element_size
                and self.element_type == other.element_type
            )
        return False

    def __hash__(self):
        return hash((self.nelements, self.element_size, self.element_type))

    def __str__(self):
        if self.nelements == 0:
            return f"{self.element_type}[]"
        return f"{self.element_type}[{self.nelements}]"


class Pointer(TypeInfo):
    """Stores information about a pointer.

    Note that the referenced type is by name because recursive data structures
    would recurse indefinitely.
    """

    size = 8

    def __init__(self, target_type_name: str):
        self.target_type_name = target_type_name

    @classmethod
    def _from_json(cls, d):
        return cls(d["t"])

    def _to_json(self):
        return {"T": 2, "t": self.target_type_name}

    def __eq__(self, other):
        if isinstance(other, Pointer):
            return self.target_type_name == other.target_type_name
        return False

    def __hash__(self):
        return hash(self.target_type_name)

    def __str__(self):
        return f"{self.target_type_name} *"


class UDT(TypeInfo):
    """An object representing struct or union types"""

    def __init__(self):
        raise NotImplementedError

    class Member:
        """A member of a UDT. Can be a Field or Padding"""

        size: int = 0

    class Field(Member):
        """Information about a field in a struct or union"""

        def __init__(self, *, name: str, size: int, type_name: str):
            self.name = name
            self.type_name = type_name
            self.size = size

        @classmethod
        def _from_json(cls, d):
            return cls(name=d["n"], type_name=d["t"], size=d["s"])

        def _to_json(self):
            return {"T": 3, "n": self.name, "t": self.type_name, "s": self.size}

        def __eq__(self, other):
            if isinstance(other, UDT.Field):
                return self.name == other.name and self.type_name == other.type_name
            return False

        def __hash__(self):
            return hash((self.name, self.type_name))

        def __str__(self):
            return f"{self.type_name} {self.name}"

    class Padding(Member):
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
        layout: t.Iterable[t.Union[UDT.Member, "Struct", "Union"]],
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


class Void(TypeInfo):
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


class FunctionPointer(TypeInfo):
    """Stores information about a function pointer.

    Currently only one function pointer is supported.
    """

    size = Pointer.size

    def __init__(self, name: str):
        self.name = name

    @classmethod
    def _from_json(cls, d):
        return cls(d["n"])

    def _to_json(self):
        return {"T": 8, "n": self.name}

    def __eq__(self, other):
        if isinstance(other, FunctionPointer):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return f"{self.name}"


class _Codec:
    """Encoder/Decoder functions"""

    @staticmethod
    def decode(encoded: str):
        raise NotImplemented

    class _Encoder(JSONEncoder):
        def default(self, obj):
            if hasattr(obj, "_to_json"):
                return obj._to_json()
            if isinstance(obj, set):
                return list(obj)
            return super().default(obj)

    @staticmethod
    def encode(o: t.Union[TypeLib, TypeInfo]):
        """Encodes a TypeLib or TypeInfo as JSON"""
        # 'separators' removes spaces after , and : for efficiency
        return dumps(o, cls=_Codec._Encoder, separators=(",", ":"))


class TypeLibCodec(_Codec):
    """Encoder/decoder for TypeLib"""

    @staticmethod
    def decode(encoded: str):
        """Decodes a JSON string"""
        return loads(encoded, object_hook=TypeLib._from_json)


class TypeInfoCodec(_Codec):
    """Encoder/decoder for TypeInfo"""

    @staticmethod
    def decode(encoded: str):
        """Decodes a JSON string"""

        def as_typeinfo(d):
            return {
                0: TypeInfo,
                1: Array,
                2: Pointer,
                3: UDT.Field,
                4: UDT.Padding,
                5: Struct,
                6: Union,
                7: Void,
                8: FunctionPointer,
            }[d["T"]]._from_json(d)

        return loads(encoded, object_hook=as_typeinfo)
