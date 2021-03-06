STRMEM_INDEX = 1


class array_type_data_t:
    elem_type: "tinfo_t"
    base: int
    nelems: int


class udt_type_data_t:
    total_size: int
    unpadded_size: int
    is_union: bool


class udt_member_t:
    offset: int
    size: int
    name: str
    cmt: str
    type: "tinfo_t"
    effalign: int
    tafld_bits: int
    fda: str


class tinfo_t:
    def dstr(self) -> str:
        ...

    def find_udt_member(self, udm: "udt_member_t", strmem_flags: int) -> int:
        ...

    def get_array_details(self, ai: "array_type_data_t") -> bool:
        ...

    def get_array_element(self) -> "tinfo_t":
        ...

    def get_pointed_object(self) -> "tinfo_t":
        ...

    def get_size(self) -> int:
        ...

    def get_udt_details(self, udt: "udt_type_data_t") -> bool:
        ...

    def get_udt_nmembers(self) -> int:
        ...

    def get_rettype(self, *args) -> "tinfo_t":
        ...

    def is_array(self) -> bool:
        ...

    def is_decl_ptr(self) -> bool:
        ...

    def is_funcptr(self) -> bool:
        ...

    def is_udt(self) -> bool:
        ...

    def is_union(self) -> bool:
        ...

    def is_void(self) -> bool:
        ...
