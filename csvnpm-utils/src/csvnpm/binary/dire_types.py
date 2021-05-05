"""C Type Information

Encodes information about C types, and provides functions to serialize types.

The JSON that is output prioritizes compactness over readability.
Note that all sizes are in 8-bit bytes
"""
import gzip
import os
from collections import defaultdict
from json import JSONEncoder, dumps, loads
from typing import (
    Any,
    DefaultDict,
    Dict,
    ItemsView,
    Iterable,
    KeysView,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
)
from typing import Union as tUnion
from typing import ValuesView

from csvnpm.ida import ida_typeinf


class TypeLib:
    """A library of types.

    Allows access to types by size.

    The usual dictionary magic methods are implemented, allowing for
    dictionary-like access to TypeInfo.
    """

    class Entry(NamedTuple):
        """A single entry in the TypeLib"""

        frequency: int
        typeinfo: "TypeInfo"

        def __eq__(self, other) -> bool:
            if isinstance(other, TypeLib.Entry):
                return other.typeinfo == self.typeinfo
            return False

        def __repr__(self) -> str:
            return f"({self.frequency}, {str(self.typeinfo)})"

    class EntryList(list):
        """A list of entries in the TypeLib. Each is list of Entries sorted by
        frequency.
        """

        def __init__(self, data: List["TypeLib.Entry"] = []) -> None:
            self._data = data
            self._typeinfo_to_idx: Dict[TypeInfo, int] = {
                entry.typeinfo: idx for idx, entry in enumerate(self._data)
            }

        @property
        def frequency(self) -> int:
            """The total frequency for this entry list"""
            return sum(c.frequency for c in self._data)

        def add_n(self, item: "TypeInfo", n: int) -> bool:
            """Add n items, increasing frequency if it already exists.
            Returns True if the item already existed.
            """
            update_idx: Optional[int] = None
            frequency: int
            typeinfo: "TypeInfo"
            if item in self._typeinfo_to_idx:
                update_idx = self._typeinfo_to_idx[item]
            else:
                update_idx = None
            # for idx, entry in enumerate(self._data):
            #     if entry.typeinfo == item:
            #         update_idx = idx
            #         break
            if update_idx is not None:
                old_entry = self._data[update_idx]
                self._data[update_idx] = TypeLib.Entry(
                    frequency=old_entry.frequency + n,
                    typeinfo=old_entry.typeinfo,
                )
                # self._sort()
                return True
            else:
                # Don't need to sort if we're just appending with freq 1
                self._typeinfo_to_idx[item] = len(self._data)
                self._data.append(TypeLib.Entry(frequency=n, typeinfo=item))
                return False

        def add(self, item: "TypeInfo") -> bool:
            """Add an item, increasing frequency if it already exists.
            Returns True if the item already existed.
            """
            return self.add_n(item, 1)

        def add_entry(self, entry: "TypeLib.Entry") -> bool:
            """Add an Entry, returns True if the entry already existed"""
            return self.add_n(entry.typeinfo, entry.frequency)

        def add_all(self, other: "TypeLib.EntryList") -> None:
            """Add all entries in other"""
            for entry in other:
                self.add_entry(entry)

        def get_freq(self, item: "TypeInfo") -> Optional[int]:
            """Get the frequency of an item, None if it does not exist"""
            for entry in self:
                if entry.typeinfo == item:
                    return entry.frequency
            return None

        def _sort(self):
            """Sorts the internal list by frequency"""
            self._data.sort(reverse=True, key=lambda entry: entry.frequency)
            self._typeinfo_to_idx = {
                entry.typeinfo: idx for idx, entry in enumerate(self._data)
            }

        def sort(self):
            return self._sort()

        def _to_json(self) -> List["TypeLib.Entry"]:
            return self._data

        def __iter__(self):
            yield from self._data

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, i: Any) -> Any:
            return self._data[i]

        def __setitem__(self, i: Any, v: Any) -> None:
            self._data[i] = v

        def __repr__(self) -> str:
            return f"{[(entry) for entry in self._data]}"

        def prune(self, freq) -> None:
            self._data = [entry for entry in self._data if entry[0] >= freq]

    def __init__(self, data: Optional[DefaultDict[int, "TypeLib.EntryList"]] = None):
        if data is None:
            self._data: DefaultDict[int, "TypeLib.EntryList"] = defaultdict(
                TypeLib.EntryList
            )
        else:
            self._data = data

    @staticmethod
    def parse_ida_type(typ: "ida_typeinf.tinfo_t") -> "TypeInfo":
        """Parses an IDA tinfo_t object"""
        if typ.is_void():
            return Void()
        if typ.is_funcptr() or "(" in typ.dstr():
            return FunctionPointer(name=typ.dstr())
        if typ.is_decl_ptr():
            return Pointer(typ.get_pointed_object().dstr())
        if typ.is_array():
            # To get array type info, first create an
            # array_type_data_t then call get_array_details to
            # populate it. Unions and structs follow a similar
            # pattern.
            array_info = ida_typeinf.array_type_data_t()
            typ.get_array_details(array_info)
            nelements = array_info.nelems
            element_size = array_info.elem_type.get_size()
            element_type = array_info.elem_type.dstr()
            return Array(
                nelements=nelements,
                element_size=element_size,
                element_type=element_type,
            )
        if typ.is_udt():
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
                    members.append(
                        UDT.Field(
                            name=member.name,
                            size=member.size,
                            type_name=type_name,
                        )
                    )
                end_padding = size - (largest_size // 8)
                if end_padding == 0:
                    return Union(name=name, members=members)
                return Union(
                    name=name,
                    members=members,
                    padding=UDT.Padding(end_padding),
                )
            else:
                # UDT is a struct
                layout: List[tUnion[UDT.Member, "Struct", "Union"]] = []
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
                    layout.append(
                        UDT.Field(
                            name=member.name,
                            size=member.size,
                            type_name=type_name,
                        )
                    )
                # Check for padding at the end
                end_padding = size - next_offset // 8
                if end_padding > 0:
                    layout.append(UDT.Padding(end_padding))
                return Struct(name=name, layout=layout)
        return TypeInfo(name=typ.dstr(), size=typ.get_size())

    def add_ida_type(
        self,
        typ: "ida_typeinf.tinfo_t",
        worklist: Optional[Set[str]] = None,
    ) -> None:
        """Adds an element to the TypeLib by parsing an IDA tinfo_t object"""
        if worklist is None:
            worklist = set()
        if typ.dstr() in worklist or typ.is_void():
            return
        worklist.add(typ.dstr())
        new_type: TypeInfo = self.parse_ida_type(typ)
        # If this type isn't a duplicate, break down the subtypes
        if not self._data[new_type.size].add(new_type):
            if typ.is_decl_ptr() and not (typ.is_funcptr() or "(" in typ.dstr()):
                self.add_ida_type(typ.get_pointed_object(), worklist)
            elif typ.is_array():
                self.add_ida_type(typ.get_array_element(), worklist)
            elif typ.is_udt():
                udt_info = ida_typeinf.udt_type_data_t()
                typ.get_udt_details(udt_info)
                name = typ.dstr()  # noqa: F841
                size = udt_info.total_size  # noqa: F841
                nmembers = typ.get_udt_nmembers()
                for n in range(nmembers):
                    member = ida_typeinf.udt_member_t()
                    # To get the nth member set OFFSET to n and tell find_udt_member
                    # to search by index.
                    member.offset = n
                    typ.find_udt_member(member, ida_typeinf.STRMEM_INDEX)
                    self.add_ida_type(member.type, worklist)

    def add_entry_list(self, size: int, entries: "TypeLib.EntryList") -> None:
        """Add an entry list of items of size 'size'"""
        if size in self:
            self[size].add_all(entries)
        else:
            self[size] = entries

    def add(self, typ):
        entry = TypeLib.EntryList()
        entry.add(typ)
        self.add_entry_list(typ.size, entry)

    def add_json_file(self, json_file: str, *, threads: int = 1) -> None:
        """Adds the info in a serialized (gzipped) JSON file to this TypeLib"""
        other: Optional[Any] = None
        with open(json_file, "r") as other_file:
            other = TypeLibCodec.decode(other_file.read())
        if other is not None and isinstance(other, TypeLib):
            for size, entries in other.items():
                self.add_entry_list(size, entries)

    def sort(self) -> None:
        for size, entries in self.items():
            entries.sort()

    @staticmethod
    def fix_bit(typ):
        succeed = True
        for m in typ.layout:
            if isinstance(m, UDT.Field):
                succeed &= m.size % 8 == 0
                if not succeed:
                    break
                m.size //= 8
            elif isinstance(m, Struct):
                succeed &= TypeLib.fix_bit(m)
        typ.size = sum(m.size for m in typ.layout)
        return succeed

    def fix(self):
        """HACK: workaround to deal with the struct bit/bytes problem in the data."""
        new_lib = TypeLib()
        for size in self.keys():
            for entry in self[size]:
                succeed = True
                if isinstance(entry.typeinfo, Struct):
                    succeed &= TypeLib.fix_bit(entry.typeinfo)
                    nsize = entry.typeinfo.size
                else:
                    nsize = size
                if succeed:
                    if nsize not in new_lib:
                        new_lib.add_entry_list(nsize, TypeLib.EntryList())
                    new_lib[nsize].add_entry(entry)
        return new_lib

    def make_cached_replacement_dict(self):
        self.cached_replacement_dict = defaultdict(set)
        for size in self.keys():
            if size > 1024:
                continue
            for entry in self[size]:
                self.cached_replacement_dict[
                    entry.typeinfo.accessible_offsets(),
                    entry.typeinfo.start_offsets(),
                ].add(entry.typeinfo)

    def valid_layout_for_types(self, rest_a, rest_s, typs):
        for typ in typs:
            if len(rest_a) == 0:
                return False
            ret = self.get_next_replacements(rest_a, rest_s)
            find = False
            for typ_set, a, s in ret:
                if typ in typ_set:
                    rest_a, rest_s = a, s
                    find = True
                    break
            if not find:
                return False
        return True

    def get_replacements(
        self, types: Tuple["TypeInfo", ...]
    ) -> Iterable[Tuple["TypeInfo", ...]]:
        """Given a list of types, get all possible lists of replacements"""
        raise NotImplementedError

    def get_next_replacements(
        self, accessible: Tuple[int, ...], start_offsets: Tuple[int, ...]
    ) -> List[Tuple[Set["TypeInfo"], Tuple[int, ...], Tuple[int, ...]]]:
        """Given a memory layout, get a list of next possible legal types.

        accessible: accessible (i.e., non-padding) addresses in memory
        start_offsets: legal offsets for the start of a type

        Returns an iterable of (type, accessible, start) tuples, where
        "type" is a legal type, "accessible" is a tuple of remaining
        accessible locations if this type is used, and "start" is the
        remaining start locations if this type is used.

        Notes:
        - The first start offset and accessible offset should be the same
        - The list returned is sorted by decreasing frequency in the library
        """
        if not hasattr(self, "cached_replacement_dict"):
            self.make_cached_replacement_dict()
        start = accessible[0]
        if start != start_offsets[0]:
            # print("No replacements, start != first accessible")
            return []

        # The last element of accessible is the last address in memory.
        length = (accessible[-1] - start) + 1
        replacements = []
        # Filter out types that are too long or of size zero
        for size in filter(lambda s: s <= length and s != 0, self.keys()):
            # Compute the memory layout of the remainder
            rest_accessible: Tuple[int, ...] = tuple(
                s for s in accessible if s >= (size + start)
            )
            rest_start: Tuple[int, ...] = tuple(
                s for s in start_offsets if s >= (size + start)
            )
            # If the remainder of the start offsets is not either an empty tuple
            # or if the first element of the new start offsets is not the same
            # as the first member of the new accessible, this is not a legal
            # size.
            if len(rest_start) != 0 and (
                len(rest_accessible) == 0 or rest_start[0] != rest_accessible[0]
            ):
                continue
            # If there are no more start offsets, but there are still accessible
            # offsets, this is not a legal replacement.
            if len(rest_start) == 0 and len(rest_accessible) != 0:
                continue
            shifted_cur_accessible = tuple(
                s - start for s in accessible if s < (size + start)
            )
            shifted_cur_start = tuple(
                s - start for s in start_offsets if s < (size + start)
            )
            typs: Set["TypeInfo"] = self.cached_replacement_dict[
                shifted_cur_accessible, shifted_cur_start
            ]
            replacements.append((typs, rest_accessible, rest_start))
        return replacements
        {typ.typeinfo: (a, s) for (typ, a, s) in replacements}

    @staticmethod
    def accessible_of_types(types: Iterable["TypeInfo"]) -> List[int]:
        """Given a list of types, get the list of accessible offsets.
        This is suitable for use with get_next_replacement.
        """
        offset = 0
        accessible = []
        for t in types:
            accessible += [offset + a for a in t.accessible_offsets()]
            offset += t.size
        return accessible

    @staticmethod
    def start_offsets_of_types(types: Iterable["TypeInfo"]) -> List[int]:
        """Given a list of types, get the list of accessible offsets.
        This is suitable for use with get_next_replacement.
        """
        offset = 0
        start_offsets = []
        for t in types:
            start_offsets += [offset + s for s in t.start_offsets()]
            offset += t.size
        return start_offsets

    def items(self) -> ItemsView[int, "TypeLib.EntryList"]:
        return self._data.items()

    def keys(self) -> KeysView[int]:
        return self._data.keys()

    def values(self) -> ValuesView["TypeLib.EntryList"]:
        return self._data.values()

    @classmethod
    def load_dir(cls, path: str, *, threads: int = 1):
        """Loads all the serialized (gzipped) JSON files in a directory"""
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        with gzip.open(files[0], "rt") as first_serialized:
            new_lib = TypeLibCodec.decode(first_serialized.read())

        if isinstance(new_lib, TypeLib):  # None is not a TypeLib
            for f in files[1:]:
                new_lib.add_json_file(f, threads=threads)
        return new_lib

    @classmethod
    def _from_json(cls, d: Dict[str, Any]) -> "TypeLib":
        data: DefaultDict[int, "TypeLib.EntryList"] = defaultdict(TypeLib.EntryList)
        # Convert lists of types into sets
        for key, lib_entry in d.items():
            if key == "T":
                continue
            entry_list = [
                TypeLib.Entry(frequency=f, typeinfo=ti) for (f, ti) in lib_entry
            ]
            data[int(key)] = TypeLib.EntryList(entry_list)
        return cls(data)

    def _to_json(self) -> Dict[Any, Any]:
        """Encodes as JSON

        The 'T' field encodes which TypeInfo class is represented by this JSON:
            E: TypeLib.EntryList
            0: TypeLib
            1: TypeInfo
            2: Array
            3: Pointer
            4: UDT.Field
            5: UDT.Padding
            6: Struct
            7: Union
            8: Void
            9: FunctionPointer

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
        encoded: Dict[Any, Any] = {
            str(key): val._to_json() for key, val in self._data.items()
        }
        encoded["T"] = 0
        return encoded

    def __contains__(self, key: int) -> bool:
        return key in self._data

    def __iter__(self) -> Iterable[int]:
        for k in self._data.keys():
            yield k

    def __getitem__(self, key: int) -> "TypeLib.EntryList":
        return self._data[key]

    def __setitem__(self, key: int, item: "TypeLib.EntryList") -> None:
        self._data[key] = item

    def __str__(self) -> str:
        ret = ""
        for n in sorted(self._data.keys()):
            ret += f"{n}: {self._data[n]}\n"
        return ret

    def prune(self, freq) -> None:
        for key in self._data:
            self._data[key].prune(freq)

        for key, entry in self._data.items():
            if len(entry) == 0:
                self._data.pop(key)


class TypeInfo:
    """Stores information about a type"""

    def __init__(self, *, name: Optional[str], size: int):
        self.name = name
        self.size = size

    def accessible_offsets(self) -> Tuple[int, ...]:
        """Offsets accessible in this type"""
        return tuple(range(self.size))

    def inaccessible_offsets(self) -> Tuple[int, ...]:
        """Inaccessible offsets in this type (e.g., padding in a Struct)"""
        return tuple()

    def start_offsets(self) -> Tuple[int, ...]:
        """Start offsets of elements in this type"""
        return (0,)

    def replacable_with(self, others: Tuple["TypeInfo", ...]) -> bool:
        """Check if this type can be replaced with others"""
        if self.size != sum(other.size for other in others):
            return False
        cur_offset = 0
        other_start: Tuple[int, ...] = tuple()
        other_accessible: Tuple[int, ...] = tuple()
        other_inaccessible: Tuple[int, ...] = tuple()
        for other in others:

            def displace(offsets: Tuple[int, ...]) -> Tuple[int, ...]:
                return tuple(off + cur_offset for off in offsets)

            other_start += displace(other.start_offsets())
            other_accessible += displace(other.accessible_offsets())
            other_inaccessible += displace(other.inaccessible_offsets())
        return (
            set(self.start_offsets()).issubset(other_start)
            and self.accessible_offsets() == other_accessible
            and self.inaccessible_offsets() == other_inaccessible
        )

    @classmethod
    def _from_json(cls, d: Dict[str, Any]) -> "TypeInfo":
        """Decodes from a dictionary"""
        return cls(name=d["n"], size=d["s"])

    def _to_json(self) -> Dict[str, Any]:
        return {"T": 1, "n": self.name, "s": self.size}

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, TypeInfo):
            return self.name == other.name and self.size == other.size
        return False

    def __hash__(self) -> int:
        return hash((self.name, self.size))

    def __str__(self) -> str:
        return f"{self.name}"

    def tokenize(self) -> List[str]:
        return [str(self), "<eot>"]

    @classmethod
    def detokenize(cls, subtypes: List[str]) -> List[str]:
        """A list of concatenated subtypes separated by <eot>"""
        ret: List[str] = []
        current: List[str] = []
        for subtype in subtypes:
            if subtype == "<eot>":
                ret.append(cls.parse_subtype(current))
                current = []
            else:
                current.append(subtype)
        return ret

    @classmethod
    def parse_subtype(cls, subtypes: List[str]) -> str:
        if len(subtypes) == 0:
            return ""
        if subtypes[0] == "<struct>":
            ret = "struct"
            if len(subtypes) == 1:
                return ret
            ret += f" {subtypes[1]} {{ "
            for subtype in subtypes[2:]:
                ret += f"{subtype}; "
            return ret + "}"
        elif subtypes[0] == "<ptr>":
            if len(subtypes) == 1:
                return " *"
            else:
                return f"{subtypes[1]} *"
        elif subtypes[0] == "<array>":
            if len(subtypes) < 3:
                return "[]"
            else:
                return f"{subtypes[1]}{subtypes[2]}"
        else:
            return subtypes[0]


class Array(TypeInfo):
    """Stores information about an array"""

    def __init__(self, *, nelements: int, element_size: int, element_type: str):
        self.element_type = element_type
        self.element_size = element_size
        self.nelements = nelements
        self.size = element_size * nelements

    def start_offsets(self) -> Tuple[int, ...]:
        """Returns the start offsets elements in this array

        For example, the type int[4] has start offsets [0, 4, 8, 12] (for 4-byte ints).
        """
        return tuple(range(self.size)[:: self.element_size])

    @classmethod
    def _from_json(cls, d: Dict[str, Any]) -> "Array":
        return cls(nelements=d["n"], element_size=d["s"], element_type=d["t"])

    def _to_json(self) -> Dict[str, Any]:
        return {
            "T": 2,
            "n": self.nelements,
            "s": self.element_size,
            "t": self.element_type,
        }

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Array):
            return (
                self.nelements == other.nelements
                and self.element_size == other.element_size
                and self.element_type == other.element_type
            )
        return False

    def __hash__(self) -> int:
        return hash((self.nelements, self.element_size, self.element_type))

    def __str__(self) -> str:
        if self.nelements == 0:
            return f"{self.element_type}[]"
        return f"{self.element_type}[{self.nelements}]"

    def tokenize(self) -> List[str]:
        return [
            "<array>",
            f"{self.element_type}",
            f"[{self.nelements}]",
            "<eot>",
        ]


class Pointer(TypeInfo):
    """Stores information about a pointer.

    Note that the referenced type is by name because recursive data structures
    would recurse indefinitely.
    """

    size = 8

    def __init__(self, target_type_name: str):
        self.target_type_name = target_type_name

    @classmethod
    def _from_json(cls, d: Dict[str, Any]) -> "Pointer":
        return cls(d["t"])

    def _to_json(self) -> Dict[str, Any]:
        return {"T": 3, "t": self.target_type_name}

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Pointer):
            return self.target_type_name == other.target_type_name
        return False

    def __hash__(self) -> int:
        return hash(self.target_type_name)

    def __str__(self) -> str:
        return f"{self.target_type_name} *"

    def tokenize(self) -> List[str]:
        return ["<ptr>", self.target_type_name, "<eot>"]


class UDT(TypeInfo):
    """An object representing struct or union types"""

    def __init__(self) -> None:
        raise NotImplementedError

    class Member:
        """A member of a UDT. Can be a Field or Padding"""

        size: int = 0

        def __init__(self) -> None:
            raise NotImplementedError

        @classmethod
        def _from_json(cls, d: Dict[str, Any]) -> "UDT.Member":
            raise NotImplementedError

        def _to_json(self) -> Dict[str, Any]:
            raise NotImplementedError

    class Field(Member):
        """Information about a field in a struct or union"""

        def __init__(self, *, name: str, size: int, type_name: str):
            self.name = name
            self.type_name = type_name
            self.size = size

        @classmethod
        def _from_json(cls, d: Dict[str, Any]) -> "UDT.Field":
            return cls(name=d["n"], type_name=d["t"], size=d["s"])

        def _to_json(self) -> Dict[str, Any]:
            return {
                "T": 4,
                "n": self.name,
                "t": self.type_name,
                "s": self.size,
            }

        def __eq__(self, other: Any) -> bool:
            if isinstance(other, UDT.Field):
                return self.name == other.name and self.type_name == other.type_name
            return False

        def __hash__(self) -> int:
            return hash((self.name, self.type_name))

        def __str__(self) -> str:
            return f"{self.type_name} {self.name}"

    class Padding(Member):
        """Padding bytes in a struct or union"""

        def __init__(self, size: int):
            self.size = size

        @classmethod
        def _from_json(cls, d: Dict[str, int]) -> "UDT.Padding":
            return cls(size=d["s"])

        def _to_json(self) -> Dict[str, int]:
            return {"T": 5, "s": self.size}

        def __eq__(self, other: Any) -> bool:
            if isinstance(other, UDT.Padding):
                return self.size == other.size
            return False

        def __hash__(self) -> int:
            return self.size

        def __str__(self) -> str:
            return f"PADDING ({self.size})"


class Struct(UDT):
    """Stores information about a struct"""

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        layout: Iterable[tUnion[UDT.Member, "Struct", "Union"]],
    ):
        self.name = name
        self.layout = tuple(layout)
        self.size = sum(lay.size for lay in layout)

    def has_padding(self) -> bool:
        """True if the Struct has padding"""
        return any((isinstance(m, UDT.Padding) for m in self.layout))

    def accessible_offsets(self) -> Tuple[int, ...]:
        """Offsets accessible in this struct"""
        accessible: Tuple[int, ...] = tuple()
        current_offset = 0
        for m in self.layout:
            next_offset = current_offset + m.size
            if isinstance(m, UDT.Field):
                for offset in range(current_offset, next_offset):
                    accessible += (offset,)
            current_offset = next_offset
        return accessible

    def inaccessible_offsets(self) -> Tuple[int, ...]:
        """Offsets inaccessible in this struct"""
        if not self.has_padding():
            return tuple()
        inaccessible: Tuple[int, ...] = tuple()
        current_offset = 0
        for m in self.layout:
            next_offset = current_offset + m.size
            if isinstance(m, UDT.Padding):
                for offset in range(current_offset, next_offset):
                    inaccessible += (offset,)
            current_offset = next_offset
        return inaccessible

    def start_offsets(self) -> Tuple[int, ...]:
        """Returns the start offsets of fields in this struct

        For example, if int is 4-bytes, char is 1-byte, and long is 8-bytes,
        a struct with the layout:
        [int, char, padding(3), long, long]
        has offsets [0, 4, 8, 16].
        """
        starts: Tuple[int, ...] = tuple()
        current_offset = 0
        for m in self.layout:
            if isinstance(m, UDT.Field):
                starts += (current_offset,)
            current_offset += m.size
        return starts

    @classmethod
    def _from_json(cls, d: Dict[str, Any]) -> "Struct":
        return cls(name=d["n"], layout=d["l"])

    def _to_json(self) -> Dict[str, Any]:
        return {
            "T": 6,
            "n": self.name,
            "l": [lay._to_json() for lay in self.layout],
        }

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Struct):
            return self.name == other.name and self.layout == other.layout
        return False

    def __hash__(self) -> int:
        return hash((self.name, self.layout))

    def __str__(self) -> str:
        if self.name is None:
            ret = "struct {{ "
        else:
            ret = f"struct {self.name} {{ "
        for lay in self.layout:
            ret += f"{str(lay)}; "
        ret += "}"
        return ret

    def tokenize(self) -> List[str]:
        return (
            ["<struct>", self.name if self.name is not None else ""]
            + [str(lay) for lay in self.layout]
            + ["<eot>"]
        )


class Union(UDT):
    """Stores information about a union"""

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        members: Iterable[tUnion[UDT.Field, "Struct", "Union"]],
        padding: Optional[UDT.Padding] = None,
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

    def has_padding(self) -> bool:
        """Returns True if this Union has padding"""
        return self.padding is not None

    def accessible_offsets(self) -> Tuple[int, ...]:
        """Offsets accessible in this Union"""
        return tuple(range(max(m.size for m in self.members)))

    def inaccessible_offsets(self) -> Tuple[int, ...]:
        """Offsets inaccessible in this Union"""
        if not self.has_padding():
            return tuple()
        return tuple(range(max(m.size for m in self.members), self.size))

    def start_offsets(self) -> Tuple[int, ...]:
        """Returns the start offsets elements in this Union"""
        return (0,)

    @classmethod
    def _from_json(cls, d: Dict[str, Any]) -> "Union":
        return cls(name=d["n"], members=d["m"], padding=d["p"])

    def _to_json(self) -> Dict[str, Any]:
        return {
            "T": 8,
            "n": self.name,
            "m": [m._to_json() for m in self.members],
            "p": self.padding,
        }

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Union):
            return (
                self.name == other.name
                and self.members == other.members
                and self.padding == other.padding
            )
        return False

    def __hash__(self) -> int:
        return hash((self.name, self.members, self.padding))

    def __str__(self) -> str:
        if self.name is None:
            ret = "union {{ "
        else:
            ret = f"union {self.name} {{ "
        for m in self.members:
            ret += f"{str(m)}; "
        if self.padding is not None:
            ret += f"{str(self.padding)}; "
        ret += "}"
        return ret

    def tokenize(self) -> List[str]:
        raise NotImplementedError


class Void(TypeInfo):
    size = 0

    def __init__(self) -> None:
        pass

    @classmethod
    def _from_json(cls, d: Dict[str, Any]) -> "Void":
        return cls()

    def _to_json(self) -> Dict[str, int]:
        return {"T": 8}

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Void)

    def __hash__(self) -> int:
        return 0

    def __str__(self) -> str:
        return "void"


class Disappear(TypeInfo):
    """Target type for variables that don't appear in the ground truth function"""

    size = 0

    def __init__(self) -> None:
        pass

    @classmethod
    def _from_json(cls, d: Dict[str, Any]) -> "Disappear":
        return cls()

    def _to_json(self) -> Dict[str, int]:
        return {"T": 10}

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Disappear)

    def __hash__(self) -> int:
        return 0

    def __str__(self) -> str:
        return "disappear"


class FunctionPointer(TypeInfo):
    """Stores information about a function pointer."""

    size = Pointer.size

    def __init__(self, name: str):
        self.name = name

    def replacable_with(self, other: Tuple["TypeInfo", ...]) -> bool:
        # No function pointers are replacable for now
        return False

    @classmethod
    def _from_json(cls, d: Dict[str, tUnion[str, int, None]]) -> "FunctionPointer":
        return cls(d["n"])  # type: ignore

    def _to_json(self) -> Dict[str, Any]:
        return {"T": 9, "n": self.name}

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, FunctionPointer):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return f"{self.name}"


class TypeLibCodec:
    """Encoder/Decoder functions"""

    CodecTypes = tUnion["TypeLib", "TypeLib.EntryList", "TypeInfo", "UDT.Member"]

    @staticmethod
    def decode(encoded: str) -> CodecTypes:
        """Decodes a JSON string"""

        return loads(encoded, object_hook=TypeLibCodec.read_metadata)

    @staticmethod
    def read_metadata(d: Dict[str, Any]) -> "TypeLibCodec.CodecTypes":
        classes: Dict[
            tUnion[int, str],
            tUnion[
                Type["TypeLib"],
                Type["TypeLib.EntryList"],
                Type["TypeInfo"],
                Type["UDT.Member"],
            ],
        ] = {
            "E": TypeLib.EntryList,
            0: TypeLib,
            1: TypeInfo,
            2: Array,
            3: Pointer,
            4: UDT.Field,
            5: UDT.Padding,
            6: Struct,
            7: Union,
            8: Void,
            9: FunctionPointer,
            10: Disappear,
        }
        return classes[d["T"]]._from_json(d)  # type: ignore

    class _Encoder(JSONEncoder):
        def default(self, obj: Any) -> Any:
            if hasattr(obj, "_to_json"):
                return obj._to_json()
            if isinstance(obj, set):
                return list(obj)
            return super().default(obj)

    @staticmethod
    def encode(o: CodecTypes) -> str:
        """Encodes a TypeLib or TypeInfo as JSON"""
        # 'separators' removes spaces after , and : for efficiency
        return dumps(o, cls=TypeLibCodec._Encoder, separators=(",", ":"))
