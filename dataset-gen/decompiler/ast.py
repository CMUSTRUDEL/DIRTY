import typing as t

from collections import defaultdict

import ida_hexrays as hr

from .typeinfo import TypeLib, TypeInfo
from .variable import Variable


class Statement:
    """A statement. Corresponds to the Hex-Rays ctype_t types"""

    def __init__(self, node_id: int):
        self.node_id = node_id

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Statement":
        return cls(node_id=item.obj_id)


class Expression(Statement):
    """An expression, considered a special case of a statement. Corresponds to the
    Hex-Rays cot_ types"""

    @classmethod
    def from_item(cls, item: hr.cexpr_t) -> "Expression":
        return cls(node_id=item.obj_id)


class UnaryExpression(Expression):
    """A unary expression. Has one operand stored in x"""

    def __init__(self, node_id: int, x: Expression):
        self.node_id = node_id
        self.x = x

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "UnaryExpression":
        node_id = item.obj_id
        x = parse_hexrays_expression(item.x)
        return cls(node_id=node_id, x=x)


class BinaryExpression(Expression):
    """A binary expression. Has two operands stored in x and y"""

    def __init__(self, node_id: int, x: Expression, y: Expression):
        self.node_id = node_id
        self.x = x
        self.y = y

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "BinaryExpression":
        node_id = item.obj_id
        x = parse_hexrays_expression(item.x)
        y = parse_hexrays_expression(item.y)
        return cls(node_id=node_id, x=x, y=y)


class Comma(BinaryExpression):
    """cot_comma (x, y)"""

    pass


class Asg(BinaryExpression):
    """cot_asg (x = y)"""

    pass


class Asgbor(BinaryExpression):
    """cot_asgbor (x |= y)"""

    pass


class Asgxor(BinaryExpression):
    """cot_asgxor (x ^= y)"""

    pass


class Asgband(BinaryExpression):
    """cot_asgband (x &= y)"""

    pass


class Asgadd(BinaryExpression):
    """cot_asgadd (x += y)"""

    pass


class Asgsub(BinaryExpression):
    """cot_asgsub (x -= y)"""

    pass


class Asgmul(BinaryExpression):
    """cot_asgmul (x *= y)"""

    pass


class Asgsshr(BinaryExpression):
    """cot_asgsshr (x >>= y signed)"""

    pass


class Asgushr(BinaryExpression):
    """cot_asgushr (x >>= y unsigned)"""

    pass


class Asgshl(BinaryExpression):
    """cot_asgshl (x <<= y)"""

    pass


class Asgsdiv(BinaryExpression):
    """cot_asgsdiv (x /= y signed)"""

    pass


class Asgudiv(BinaryExpression):
    """cot_asgudiv (x /= y unsigned)"""

    pass


class Asgsmod(BinaryExpression):
    """cot_asgsmod (x %= y signed)"""

    pass


class Asgumod(BinaryExpression):
    """cot_asgumod (x %= y unsigned)"""

    pass


class Tern(Expression):
    """cot_tern (x ? y : z)"""

    def __init__(self, node_id: int, x: Expression, y: Expression, z: Expression):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Tern":
        node_id = item.obj_id
        x = parse_hexrays_expression(item.x)
        y = parse_hexrays_expression(item.y)
        z = parse_hexrays_expression(item.z)
        return cls(node_id=node_id, x=x, y=y, z=z)


class Lor(BinaryExpression):
    """cot_lor (x || y)"""

    pass


class Land(BinaryExpression):
    """cot_land (x && y)"""

    pass


class Bor(BinaryExpression):
    """cot_bor (x | y)"""

    pass


class Xor(BinaryExpression):
    """cot_xor (x ^ y)"""

    pass


class Band(BinaryExpression):
    """cot_band (x & y)"""

    pass


class Eq(BinaryExpression):
    """cot_eq (x == y int or fpu)"""

    pass


class Ne(BinaryExpression):
    """cot_ne (x != y int or fpu)"""

    pass


class Sge(BinaryExpression):
    """cot_sge (x >= y signed or fpu)"""

    pass


class Uge(BinaryExpression):
    """cot_uge (x >= y unsigned)"""

    pass


class Sle(BinaryExpression):
    """cot_sle (x <= y signed or fpu)"""

    pass


class Ule(BinaryExpression):
    """cot_ule (x <= y unsigned)"""

    pass


class Sgt(BinaryExpression):
    """cot_sgt (x > y signed or fpu)"""

    pass


class Ugt(BinaryExpression):
    """cot_ugt (x > y unsigned)"""

    pass


class Slt(BinaryExpression):
    """cot_slt (x < y signed or fpu)"""

    pass


class Ult(BinaryExpression):
    """cot_ult (x < y unsigned)"""

    pass


class Sshr(BinaryExpression):
    """cot_sshr (x >> y signed)"""

    pass


class Ushr(BinaryExpression):
    """cot_ushr (x >> y unsigned)"""

    pass


class Shl(BinaryExpression):
    """cot_shl (x << y)"""

    pass


class Add(BinaryExpression):
    """cot_add (x + y)"""

    pass


class Sub(BinaryExpression):
    """cot_sub (x - y)"""

    pass


class Mul(BinaryExpression):
    """cot_mul (x * y)"""

    pass


class Sdiv(BinaryExpression):
    """cot_sdiv (x / y signed)"""

    pass


class Udiv(BinaryExpression):
    """cot_udiv (x / y unsigned)"""

    pass


class Smod(BinaryExpression):
    """cot_smod (x % y signed)"""

    pass


class Umod(BinaryExpression):
    """cot_umod (x % y unsigned)"""

    pass


class Fadd(BinaryExpression):
    """cot_fadd (x + y fp)"""

    pass


class Fsub(BinaryExpression):
    """cot_fsub (x - y fp)"""

    pass


class Fmul(BinaryExpression):
    """cot_fmul (x * y fp)"""

    pass


class Fdiv(BinaryExpression):
    """cot_fdiv (x / y fp)"""

    pass


class Fneg(UnaryExpression):
    """cot_fneg (-x fp)"""

    pass


class Neg(UnaryExpression):
    """cot_neg (-x)"""

    pass


class Cast(UnaryExpression):
    """cot_cast ((type)x)"""

    pass


class Lnot(UnaryExpression):
    """cot_lnot (!x)"""

    pass


class Bnot(UnaryExpression):
    """cot_bnot (~x)"""

    pass


class Ptr(Expression):
    """cot_ptr (*x, access size in 'ptrsize')"""

    def __init__(self, node_id: int, x: Expression, ptrsize: int):
        self.node_id = node_id
        self.x = x
        self.ptrsize = ptrsize

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Ptr":
        node_id = item.obj_id
        x = parse_hexrays_expression(item.x)
        return cls(node_id=node_id, x=x, ptrsize=item.ptrsize)


class Ref(UnaryExpression):
    """cot_ref (&x)"""

    pass


class Postinc(UnaryExpression):
    """cot_postinc (x++)"""

    pass


class Postdec(UnaryExpression):
    """cot_postdec (x--)"""

    pass


class Preinc(UnaryExpression):
    """cot_preinc (++x)"""

    pass


class Predec(UnaryExpression):
    """cot_predec (--x)"""

    pass


class Call(Expression):
    """cot_call (x(...))"""

    class Arg(Expression):
        """An argument"""

        def __init__(self, item: hr.carg_t):
            # FIXME: This is technically an expression,
            # so I think there should be more parsering
            self.node_id = item.obj_id
            self.is_vararg = item.is_vararg
            self.formal_type: TypeInfo = TypeLib.parse_ida_type(item.formal_type)

    def __init__(self, node_id: int, x: Expression, a: t.List[Call.Arg]):
        self.node_id = node_id
        self.x = x
        self.a = a

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Call":
        node_id = item.obj_id
        x = parse_hexrays_item(item.x)
        a = [Call.Arg(i) for i in item.a]
        return cls(node_id=node_id, x=item.x, a=a)


class Idx(BinaryExpression):
    """cot_idx (x[y])"""

    pass


class Memref(Expression):
    """cot_memref (x.m)"""

    def __init__(self, node_id: int, x: Expression, m: int):
        self.node_id = node_id
        self.x = x
        self.m = m

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Memref":
        node_id = item.obj_id
        x = parse_hexrays_expression(item.x)
        return cls(node_id=node_id, x=x, m=item.m)


class Memptr(Expression):
    """cot_memptr (x->m, access size in 'ptrsize')"""

    def __init__(self, node_id: int, x: Expression, m: int, ptrsize: int):
        self.node_id = node_id
        self.x = x
        self.m = m
        self.ptrsize = ptrsize

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Memptr":
        node_id = item.obj_id
        x = parse_hexrays_expression(item.x)
        return cls(node_id=node_id, x=x, m=item.m, ptrsize=item.ptrsize)


class Num(Expression):
    """cot_num (n: numeric constant)"""

    def __init__(self, node_id: int, n: int):
        self.node_id = node_id
        self.n = n

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Num":
        node_id = item.obj_id
        return cls(node_id=node_id, n=item.n)


class Fnum(Expression):
    """cot_fnum (fpc: floating point constant)"""

    def __init__(self, node_id: int, fpc: int):
        self.node_id = node_id
        self.fpc = fpc

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Fnum":
        node_id = item.obj_id
        return cls(node_id=node_id, fpc=item.fpc)


class Str(Expression):
    """cot_fnum (string constant)"""

    def __init__(self, node_id: int, string: str):
        self.node_id = node_id
        self.string = string

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Str":
        node_id = item.obj_id
        return cls(node_id=node_id, string=item.string)


class Obj(Expression):
    """cot_obj (obj_ea)"""

    def __init__(self, node_id: int, obj_ea: int):
        self.node_id = node_id
        self.obj_ea = obj_ea

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Obj":
        node_id = item.obj_id
        return cls(node_id=node_id, obj_ea=item.obj_ea)


class Var(Expression):
    """cot_var (v)

    idx is an offset into lvars_t, used by the AST to generate a Variable object.
    """

    def __init__(self, node_id: int, idx: int):
        self.node_id = node_id
        self.idx = idx

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Var":
        node_id = item.obj_id
        idx = item.v.idx
        return cls(node_id=node_id, idx=idx)


class Insn(Expression):
    """cot_insn (instruction in expression, internal representation only)"""

    pass


class Sizeof(UnaryExpression):
    """cot_sizeof (sizeof(x))"""

    pass


class Helper(Expression):
    """cot_helper (arbitrary name)"""

    pass


class Type(Expression):
    """cot_helper (arbitrary type)"""

    def __init__(self, node_id: int, typ: TypeInfo):
        self.node_id = node_id
        self.typ = typ

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Type":
        node_id = item.obj_id
        return cls(node_id=node_id, typ=TypeLib.parse_ida_type(item.type))


########## Statements ##########


class Block(Statement):
    """cit_block"""

    def __init__(self, node_id: int, statements: t.List[Statement]):
        self.node_id = node_id
        self.statements = statements

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Block":
        node_id = item.obj_id
        statements = [parse_hexrays_item(i) for i in item]
        return cls(node_id=node_id, statements=statements)


class ExprStatement(Statement):
    """A statement that has an expression"""

    def __init__(self, node_id: int, expr: Expression):
        self.node_id = node_id
        self.expr = expr

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "ExprStatement":
        node_id = item.obj_id
        expr = parse_hexrays_expression(item.expr)
        return cls(node_id=node_id, expr=expr)


class If(ExprStatement):
    """cit_if"""

    def __init__(
        self, node_id: int, expr: Expression, ithen: Statement, ielse: Statement
    ):
        self.node_id = node_id
        self.ithen = ithen
        self.ielse = ielse
        self.expr = expr

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "If":
        node_id = item.obj_id
        ithen = parse_hexrays_statement(item.ithen)
        ielse = parse_hexrays_statement(item.ielse)
        expr = parse_hexrays_expression(item.expr)
        return cls(node_id=node_id, ithen=ithen, ielse=ielse, expr=expr)


class Loop(ExprStatement):
    """A generic loop. body is the loop body, while expr is the guard expression"""

    def __init__(self, node_id: int, body: Statement, expr: Expression):
        self.node_id = node_id
        self.body = body
        self.expr = expr

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Loop":
        node_id = item.obj_id
        body = parse_hexrays_statement(item.body)
        expr = parse_hexrays_expression(item.expr)
        return cls(node_id=node_id, body=body, expr=expr)


class Do(Loop):
    """cit_do"""

    pass


class While(Loop):
    """cit_while"""

    pass


class For(Loop):
    """cit_for"""

    def __init__(
        self,
        node_id: int,
        body: Statement,
        expr: Expression,
        init: Expression,
        step: Expression,
    ):
        super().__init__(node_id, body, expr)
        self.init = init
        self.step = step

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "For":
        node_id = item.obj_id
        body = parse_hexrays_statement(item.body)
        expr = parse_hexrays_expression(item.expr)
        init = parse_hexrays_expression(item.init)
        step = parse_hexrays_expression(item.step)
        return cls(
            node_id=node_id,
            body=body,
            expr=expr,
            init=init,
            step=step,
        )


class Switch(ExprStatement):
    """cit_switch"""

    class Case(Statement):
        def __init__(self, item: hr.ccase_t):
            self.node_id = item.obj_id
            self.values = item.values

    def __init__(
        self,
        node_id: int,
        body: Statement,
        expr: Expression,
        mvnf: int,
        cases: t.List["Case"],
    ):
        super().__init__(node_id, expr)
        self.body = parse_hexrays_statement(body)
        self.mvnf = mvnf
        self.cases = cases

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Switch":
        node_id = item.obj_id
        body = parse_hexrays_statement(item.body)
        expr = parse_hexrays_expression(item.expr)
        cases = [Switch.Case(c) for c in item.cases]
        return cls(
            node_id=node_id,
            body=body,
            expr=expr,
            mvnf=item.mvnf,
            cases=cases,
        )


class Return(ExprStatement):
    """cit_return"""

    pass


class Goto(Statement):
    """cit_goto"""

    def __init__(self, node_id: int, label_num: int):
        self.node_id = node_id
        self.label_num = label_num

    @classmethod
    def from_item(cls, item: hr.citem_t) -> "Goto":
        node_id = item.obj_id
        return cls(node_id=node_id, label_num=item.label_num)


class Asm(Statement):
    """cit_asm, not supported"""

    pass


class Break(Statement):
    """cit_break"""

    pass


class Continue(Statement):
    """cit_continue"""

    pass

def parse_hexrays_expression(expr: hr.cexpr_t) -> Expression:
    """Parses a HexRays expression and returns an Expression object"""
    classes: t.Dict[hr.cexpr_t, t.Type["Expression"]] = {
        hr.cot_comma: Comma,
        hr.cot_asg: Asg,
        hr.cot_asgbor: Asgbor,
        hr.cot_asgxor: Asgxor,
        hr.cot_asgband: Asgband,
        hr.cot_asgadd: Asgadd,
        hr.cot_asgsub: Asgsub,
        hr.cot_asgmul: Asgmul,
        hr.cot_asgsshr: Asgsshr,
        hr.cot_asgushr: Asgushr,
        hr.cot_asgshl: Asgshl,
        hr.cot_asgsdiv: Asgsdiv,
        hr.cot_asgudiv: Asgudiv,
        hr.cot_asgsmod: Asgsmod,
        hr.cot_asgumod: Asgumod,
        hr.cot_tern: Tern,
        hr.cot_lor: Lor,
        hr.cot_land: Land,
        hr.cot_bor: Bor,
        hr.cot_xor: Xor,
        hr.cot_band: Band,
        hr.cot_eq: Eq,
        hr.cot_ne: Ne,
        hr.cot_sge: Sge,
        hr.cot_uge: Uge,
        hr.cot_sle: Sle,
        hr.cot_ule: Ule,
        hr.cot_sgt: Sgt,
        hr.cot_ugt: Ugt,
        hr.cot_slt: Slt,
        hr.cot_ult: Ult,
        hr.cot_sshr: Sshr,
        hr.cot_ushr: Ushr,
        hr.cot_shl: Shl,
        hr.cot_add: Add,
        hr.cot_sub: Sub,
        hr.cot_mul: Mul,
        hr.cot_sdiv: Sdiv,
        hr.cot_udiv: Udiv,
        hr.cot_smod: Smod,
        hr.cot_umod: Umod,
        hr.cot_fadd: Fadd,
        hr.cot_fsub: Fsub,
        hr.cot_fmul: Fmul,
        hr.cot_fdiv: Fdiv,
        hr.cot_fneg: Fneg,
        hr.cot_neg: Neg,
        hr.cot_cast: Cast,
        hr.cot_lnot: Lnot,
        hr.cot_bnot: Bnot,
        hr.cot_ptr: Ptr,
        hr.cot_ref: Ref,
        hr.cot_postinc: Postinc,
        hr.cot_postdec: Postdec,
        hr.cot_preinc: Preinc,
        hr.cot_predec: Predec,
        hr.cot_call: Call,
        hr.cot_idx: Idx,
        hr.cot_memref: Memref,
        hr.cot_memptr: Memptr,
        hr.cot_num: Num,
        hr.cot_fnum: Fnum,
        hr.cot_fnum: Str,
        hr.cot_obj: Obj,
        hr.cot_var: Var,
        hr.cot_insn: Insn,
        hr.cot_sizeof: Sizeof,
        hr.cot_helper: Helper,
        hr.cot_type: Type,
    }
    return classes[expr.op].from_item(expr)


def parse_hexrays_statement(stmt: hr.cinsn_t) -> Statement:
    """Parses a HexRays statement and returns a Statement object"""
    classes: t.Dict[hr.cinsn_t, t.Type[Statement]] = {
        hr.cit_block: Block,
        hr.cit_if: If,
        hr.cit_do: Do,
        hr.cit_while: While,
        hr.cit_for: For,
        hr.cit_switch: Switch,
        hr.cit_return: Return,
        hr.cit_goto: Goto,
        hr.cit_asm: Asm,
        hr.cit_break: Break,
        hr.cit_continue: Continue,
    }
    return classes[stmt.op].from_item(stmt)


def parse_hexrays_item(item: hr.citem_t) -> Statement:
    """Parses any HexRays item and returns a Statement object"""
    try:
        return parse_hexrays_statement(item)
    except KeyError:
        return parse_hexrays_expression(item)


class AST:
    def __init__(self, function: hr.cfunc_t):
        self.function = function
        # Statement should be a block
        self.root = parse_hexrays_statement(function.body)
