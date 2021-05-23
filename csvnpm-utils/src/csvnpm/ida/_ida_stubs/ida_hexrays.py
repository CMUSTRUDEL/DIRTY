from enum import IntEnum
from typing import Any, List

from .ida_typeinf import tinfo_t


class DecompilationFailure(Exception):
    pass


# This is technically cfuncptr_t, but that is just a reference-counted
# version of cfunc_t
def decompile(*args) -> "cfunc_t":
    ...


def get_hexrays_version() -> str:
    ...


def init_hexrays_plugin() -> bool:
    ...


def load_plugin(name: str) -> None:
    ...


class ctype_t(IntEnum):
    cot_empty = 0
    cot_comma = 1
    cot_asg = 2
    cot_asgbor = 3
    cot_asgxor = 4
    cot_asgband = 5
    cot_asgadd = 6
    cot_asgsub = 7
    cot_asgmul = 8
    cot_asgsshr = 9
    cot_asgushr = 10
    cot_asgshl = 11
    cot_asgsdiv = 12
    cot_asgudiv = 13
    cot_asgsmod = 14
    cot_asgumod = 15
    cot_tern = 16
    cot_lor = 17
    cot_land = 18
    cot_bor = 19
    cot_xor = 20
    cot_band = 21
    cot_eq = 22
    cot_ne = 23
    cot_sge = 24
    cot_uge = 25
    cot_sle = 26
    cot_ule = 27
    cot_sgt = 28
    cot_ugt = 29
    cot_slt = 30
    cot_ult = 31
    cot_sshr = 32
    cot_ushr = 33
    cot_shl = 34
    cot_add = 35
    cot_sub = 36
    cot_mul = 37
    cot_sdiv = 38
    cot_udiv = 39
    cot_smod = 40
    cot_umod = 41
    cot_fadd = 42
    cot_fsub = 43
    cot_fmul = 44
    cot_fdiv = 45
    cot_fneg = 46
    cot_neg = 47
    cot_cast = 48
    cot_lnot = 49
    cot_bnot = 50
    cot_ptr = 51
    cot_ref = 52
    cot_postinc = 53
    cot_postdec = 54
    cot_preinc = 55
    cot_predec = 56
    cot_call = 57
    cot_idx = 58
    cot_memref = 59
    cot_memptr = 60
    cot_num = 61
    cot_fnum = 62
    cot_str = 63
    cot_obj = 64
    cot_var = 65
    cot_insn = 66
    cot_sizeof = 67
    cot_helper = 68
    cot_type = 69
    cot_last = cot_type
    cit_empty = 70
    cit_block = 71
    cit_expr = 72
    cit_if = 73
    cit_for = 74
    cit_while = 75
    cit_do = 76
    cit_switch = 77
    cit_break = 78
    cit_continue = 79
    cit_return = 80
    cit_goto = 81
    cit_asm = 82


cot_empty = ctype_t.cot_empty
cot_comma = ctype_t.cot_comma
cot_asg = ctype_t.cot_asg
cot_asgbor = ctype_t.cot_asgbor
cot_asgxor = ctype_t.cot_asgxor
cot_asgband = ctype_t.cot_asgband
cot_asgadd = ctype_t.cot_asgadd
cot_asgsub = ctype_t.cot_asgsub
cot_asgmul = ctype_t.cot_asgmul
cot_asgsshr = ctype_t.cot_asgsshr
cot_asgushr = ctype_t.cot_asgushr
cot_asgshl = ctype_t.cot_asgshl
cot_asgsdiv = ctype_t.cot_asgsdiv
cot_asgudiv = ctype_t.cot_asgudiv
cot_asgsmod = ctype_t.cot_asgsmod
cot_asgumod = ctype_t.cot_asgumod
cot_tern = ctype_t.cot_tern
cot_lor = ctype_t.cot_lor
cot_land = ctype_t.cot_land
cot_bor = ctype_t.cot_bor
cot_xor = ctype_t.cot_xor
cot_band = ctype_t.cot_band
cot_eq = ctype_t.cot_eq
cot_ne = ctype_t.cot_ne
cot_sge = ctype_t.cot_sge
cot_uge = ctype_t.cot_uge
cot_sle = ctype_t.cot_sle
cot_ule = ctype_t.cot_ule
cot_sgt = ctype_t.cot_sgt
cot_ugt = ctype_t.cot_ugt
cot_slt = ctype_t.cot_slt
cot_ult = ctype_t.cot_ult
cot_sshr = ctype_t.cot_sshr
cot_ushr = ctype_t.cot_ushr
cot_shl = ctype_t.cot_shl
cot_add = ctype_t.cot_add
cot_sub = ctype_t.cot_sub
cot_mul = ctype_t.cot_mul
cot_sdiv = ctype_t.cot_sdiv
cot_udiv = ctype_t.cot_udiv
cot_smod = ctype_t.cot_smod
cot_umod = ctype_t.cot_umod
cot_fadd = ctype_t.cot_fadd
cot_fsub = ctype_t.cot_fsub
cot_fmul = ctype_t.cot_fmul
cot_fdiv = ctype_t.cot_fdiv
cot_fneg = ctype_t.cot_fneg
cot_neg = ctype_t.cot_neg
cot_cast = ctype_t.cot_cast
cot_lnot = ctype_t.cot_lnot
cot_bnot = ctype_t.cot_bnot
cot_ptr = ctype_t.cot_ptr
cot_ref = ctype_t.cot_ref
cot_postinc = ctype_t.cot_postinc
cot_postdec = ctype_t.cot_postdec
cot_preinc = ctype_t.cot_preinc
cot_predec = ctype_t.cot_predec
cot_call = ctype_t.cot_call
cot_idx = ctype_t.cot_idx
cot_memref = ctype_t.cot_memref
cot_memptr = ctype_t.cot_memptr
cot_num = ctype_t.cot_num
cot_fnum = ctype_t.cot_fnum
cot_str = ctype_t.cot_str
cot_obj = ctype_t.cot_obj
cot_var = ctype_t.cot_var
cot_insn = ctype_t.cot_insn
cot_sizeof = ctype_t.cot_sizeof
cot_helper = ctype_t.cot_helper
cot_type = ctype_t.cot_type
cot_last = ctype_t.cot_last
cit_empty = ctype_t.cit_empty
cit_block = ctype_t.cit_block
cit_expr = ctype_t.cit_expr
cit_if = ctype_t.cit_if
cit_for = ctype_t.cit_for
cit_while = ctype_t.cit_while
cit_do = ctype_t.cit_do
cit_switch = ctype_t.cit_switch
cit_break = ctype_t.cit_break
cit_continue = ctype_t.cit_continue
cit_return = ctype_t.cit_return
cit_goto = ctype_t.cit_goto
cit_asm = ctype_t.cit_asm


class citem_t:
    ea: int
    obj_id: int
    op: "ctype_t"


class cexpr_t(citem_t):
    a: List[Any]
    fpc: "fnumber_t"
    insn: "cinsn_t"
    m: int
    n: int
    obj_ea: int
    ptrsize: int
    string: str
    type: tinfo_t
    v: "var_ref_t"
    x: "cexpr_t"
    y: "cexpr_t"
    z: "cexpr_t"


class cinsn_t(citem_t):
    cblock: "cblock_t"
    cexpr: "cexpr_t"
    cif: "cif_t"
    cfor: "cfor_t"
    cwhile: "cwhile_t"
    cdo: "cdo_t"
    cswitch: "cswitch_t"
    creturn: "creturn_t"
    cgoto: "cgoto_t"
    casm: "casm_t"


class ceinsn_t:
    obj_id: int
    expr: "cexpr_t"


class var_ref_t:
    mba: Any
    idx: int


class fnumber_t:
    fnum: int
    nbytes: int


class qlist_cinsn_t(list):
    # list is technically a lie
    pass


class cblock_t(qlist_cinsn_t):
    obj_id: int


class cif_t(ceinsn_t):
    ithen: "cinsn_t"
    ielse: "cinsn_t"


class cloop_t(ceinsn_t):
    body: "cinsn_t"


class cdo_t(cloop_t):
    pass


class cfor_t(cloop_t):
    init: "cexpr_t"
    step: "cexpr_t"


class cwhile_t(cloop_t):
    pass


class cnumber_t:
    def value(self, typ: tinfo_t) -> int:
        ...


class ccase_t(cinsn_t):
    values: List[int]

    def size(self) -> int:
        ...

    def value(self, i: int) -> int:
        ...


class cswitch_t(ceinsn_t):
    mvnf: "cnumber_t"
    cases: List["ccase_t"]


class creturn_t(ceinsn_t):
    pass


class cgoto_t:
    obj_id: int
    label_num: int


class casm_t:
    obj_id: int


class carg_t(cexpr_t):
    is_vararg: bool
    formal_type: tinfo_t


class lvar_locator_t:
    def get_stkoff(self) -> int:
        ...

    def get_reg1(self) -> int:
        ...

    def get_reg2(self) -> int:
        ...

    def is_reg1(self) -> bool:
        ...

    def is_reg2(self) -> bool:
        ...

    def is_reg_var(self) -> bool:
        ...

    def is_stk_var(self) -> bool:
        ...


class lvar_t(lvar_locator_t):
    name: str

    @property
    def is_arg_var(self) -> bool:
        ...

    @property
    def has_user_name(self) -> bool:
        ...

    def type(self) -> "tinfo_t":
        ...


class cfunc_t:
    arguments: List["lvar_t"]
    lvars: List["lvar_t"]
    body: "cblock_t"
    type: tinfo_t

    def get_lvars(self) -> List["lvar_t"]:
        ...

    def get_stkoff_delta(self) -> int:
        ...


class ctree_visitor_t:
    def leave_insn(self, *args) -> int:
        ...

    def leave_expr(self, *args) -> int:
        ...

    def parent_insn(self, *args) -> "cinsn_t":
        ...

    def parent_expr(self, *args) -> "cexpr_t":
        ...


class ctree_parentee_t(ctree_visitor_t):
    pass
