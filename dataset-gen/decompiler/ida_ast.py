"""HexRays AST

For the purposes of serialization, each node type corresponds to a HexRays
ctype_t, which is just an integer. This is stored in "meta"
"""


import typing as t

from collections import defaultdict

import idaapi as ida

from typeinfo import TypeLib, TypeInfo
from variable import Variable


#################### AST Nodes ####################

class Statement:
    """A statement. Corresponds to the Hex-Rays ctype_t types"""

    # A dummy meta. If this is hit during decoding there is a problem
    meta = -1

    def __init__(self, node_id: int):
        self.node_id = node_id

    def to_json(self):
        """Encodes the node as JSON"""
        return { "meta": self.meta }

    @classmethod
    def from_item(cls, item: ida.citem_t, lvars: ida.lvars_t) -> "Statement":
        node_id = item.obj_id
        return cls(node_id=node_id)

    def __repr__(self):
        return type(self).__name__

class Expression(Statement):
    """An expression, considered a special case of a statement. Corresponds to the
    Hex-Rays cot_ types"""

    @classmethod
    def from_item(cls, item: ida.citem_t, lvars: ida.lvars_t) -> "Expression":
        return cls(node_id=item.obj_id)


class UnaryExpression(Expression):
    """A unary expression. Has one operand stored in x"""

    def __init__(self, node_id: int, x: Expression):
        self.node_id = node_id
        self.x = x

    def to_json(self):
        x = self.x.to_json()
        return {
            "id": self.node_id,
            "x": x,
            "M": self.meta,
        }

    @classmethod
    def from_item(cls, item: ida.cexpr_t, lvars: ida.lvars_t) -> "UnaryExpression":
        node_id = item.obj_id
        x = parse_hexrays_expression(item.x, lvars)
        return cls(node_id=node_id, x=x)

    def __repr__(self):
        return f"{type(self).__name__} (x: {self.x})"

class BinaryExpression(Expression):
    """A binary expression. Has two operands stored in x and y"""

    def __init__(self, node_id: int, x: Expression, y: Expression):
        self.node_id = node_id
        self.x = x
        self.y = y

    def to_json(self):
        x = self.x.to_json()
        y = self.x.to_json()
        return {
            "id": self.node_id,
            "x": x,
            "y": y,
            "M": self.meta,
        }

    @classmethod
    def from_item(cls, item: ida.cexpr_t, lvars: ida.lvars_t) -> "BinaryExpression":
        node_id = item.obj_id
        x = parse_hexrays_expression(item.x, lvars)
        y = parse_hexrays_expression(item.y, lvars)
        return cls(node_id=node_id, x=x, y=y)

    def __repr__(self):
        return f"{type(self).__name__} (x: {self.x}, y: {self.y})"


class Comma(BinaryExpression):
    """cot_comma (x, y)"""

    meta = ida.cot_comma


class Asg(BinaryExpression):
    """cot_asg (x = y)"""

    meta = ida.cot_asg



class Asgbor(BinaryExpression):
    """cot_asgbor (x |= y)"""

    meta = ida.cot_asgbor


class Asgxor(BinaryExpression):
    """cot_asgxor (x ^= y)"""

    meta = ida.cot_asgxor


class Asgband(BinaryExpression):
    """cot_asgband (x &= y)"""

    meta = ida.cot_asgband


class Asgadd(BinaryExpression):
    """cot_asgadd (x += y)"""

    meta = ida.cot_asgadd


class Asgsub(BinaryExpression):
    """cot_asgsub (x -= y)"""

    meta = ida.cot_asgsub


class Asgmul(BinaryExpression):
    """cot_asgmul (x *= y)"""

    meta = ida.cot_asgmul


class Asgsshr(BinaryExpression):
    """cot_asgsshr (x >>= y signed)"""

    meta = ida.cot_asgsshr


class Asgushr(BinaryExpression):
    """cot_asgushr (x >>= y unsigned)"""

    meta = ida.cot_asgushr


class Asgshl(BinaryExpression):
    """cot_asgshl (x <<= y)"""

    meta = ida.cot_asgshl


class Asgsdiv(BinaryExpression):
    """cot_asgsdiv (x /= y signed)"""

    meta = ida.cot_asgsdiv


class Asgudiv(BinaryExpression):
    """cot_asgudiv (x /= y unsigned)"""

    meta = ida.cot_asgudiv


class Asgsmod(BinaryExpression):
    """cot_asgsmod (x %= y signed)"""

    meta = ida.cot_asgsmod


class Asgumod(BinaryExpression):
    """cot_asgumod (x %= y unsigned)"""

    meta = ida.cot_asgumod


class Tern(Expression):
    """cot_tern (x ? y : meta = ida.z)"""

    meta = ida.cot_tern

    def __init__(self, node_id: int, x: Expression, y: Expression, z: Expression):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.z = z

    def to_json(self):
        x = self.x.to_json()
        y = self.x.to_json()
        z = self.x.to_json()
        return {
            "id": self.node_id,
            "x": x,
            "y": y,
            "z": z,
            "M": self.meta
        }

    @classmethod
    def from_item(cls, item: ida.cexpr_t, lvars: ida.lvars_t) -> "Tern":
        node_id = item.obj_id
        x = parse_hexrays_expression(item.x, lvars)
        y = parse_hexrays_expression(item.y, lvars)
        z = parse_hexrays_expression(item.z, lvars)
        return cls(node_id=node_id, x=x, y=y, z=z)

    def __repr__(self):
        return f"{type(self).__name__} (x: {self.x}, y: {self.y}, z: {self.z})"

class Lor(BinaryExpression):
    """cot_lor (x || y)"""

    meta = ida.cot_lor


class Land(BinaryExpression):
    """cot_land (x && y)"""

    meta = ida.cot_land


class Bor(BinaryExpression):
    """cot_bor (x | y)"""

    meta = ida.cot_bor


class Xor(BinaryExpression):
    """cot_xor (x ^ y)"""

    meta = ida.cot_xor


class Band(BinaryExpression):
    """cot_band (x & y)"""

    meta = ida.cot_band


class Eq(BinaryExpression):
    """cot_eq (x == y int or fpu)"""

    meta = ida.cot_eq


class Ne(BinaryExpression):
    """cot_ne (x != y int or fpu)"""

    meta = ida.cot_ne


class Sge(BinaryExpression):
    """cot_sge (x >= y signed or fpu)"""

    meta = ida.cot_sge


class Uge(BinaryExpression):
    """cot_uge (x >= y unsigned)"""

    meta = ida.cot_uge


class Sle(BinaryExpression):
    """cot_sle (x <= y signed or fpu)"""

    meta = ida.cot_sle


class Ule(BinaryExpression):
    """cot_ule (x <= y unsigned)"""

    meta = ida.cot_ule


class Sgt(BinaryExpression):
    """cot_sgt (x > y signed or fpu)"""

    meta = ida.cot_sgt


class Ugt(BinaryExpression):
    """cot_ugt (x > y unsigned)"""

    meta = ida.cot_ugt


class Slt(BinaryExpression):
    """cot_slt (x < y signed or fpu)"""

    meta = ida.cot_slt


class Ult(BinaryExpression):
    """cot_ult (x < y unsigned)"""

    meta = ida.cot_ult


class Sshr(BinaryExpression):
    """cot_sshr (x >> y signed)"""

    meta = ida.cot_sshr


class Ushr(BinaryExpression):
    """cot_ushr (x >> y unsigned)"""

    meta = ida.cot_ushr


class Shl(BinaryExpression):
    """cot_shl (x << y)"""

    meta = ida.cot_shl


class Add(BinaryExpression):
    """cot_add (x + y)"""

    meta = ida.cot_add


class Sub(BinaryExpression):
    """cot_sub (x - y)"""

    meta = ida.cot_sub


class Mul(BinaryExpression):
    """cot_mul (x * y)"""

    meta = ida.cot_mul


class Sdiv(BinaryExpression):
    """cot_sdiv (x / y signed)"""

    meta = ida.cot_sdiv


class Udiv(BinaryExpression):
    """cot_udiv (x / y unsigned)"""

    meta = ida.cot_udiv


class Smod(BinaryExpression):
    """cot_smod (x % y signed)"""

    meta = ida.cot_smod


class Umod(BinaryExpression):
    """cot_umod (x % y unsigned)"""

    meta = ida.cot_umod


class Fadd(BinaryExpression):
    """cot_fadd (x + y fp)"""

    meta = ida.cot_fadd


class Fsub(BinaryExpression):
    """cot_fsub (x - y fp)"""

    meta = ida.cot_fsub


class Fmul(BinaryExpression):
    """cot_fmul (x * y fp)"""

    meta = ida.cot_fmul


class Fdiv(BinaryExpression):
    """cot_fdiv (x / y fp)"""

    meta = ida.cot_fdiv


class Fneg(UnaryExpression):
    """cot_fneg (-x fp)"""

    meta = ida.cot_fneg


class Neg(UnaryExpression):
    """cot_neg (-x)"""

    meta = ida.cot_neg


class Cast(UnaryExpression):
    """cot_cast ((type)x)"""

    meta = ida.cot_cast


class Lnot(UnaryExpression):
    """cot_lnot (!x)"""

    meta = ida.cot_lnot


class Bnot(UnaryExpression):
    """cot_bnot (~x)"""

    meta = ida.cot_bnot


class Ptr(Expression):
    """cot_ptr (*x, access size in 'ptrsize')"""

    meta = ida.cot_ptr

    def __init__(self, node_id: int, x: Expression, ptrsize: int):
        self.node_id = node_id
        self.x = x
        self.ptrsize = ptrsize

    def to_json(self):
        x = self.x.to_json()
        return {
            "id": self.node_id,
            "x": x,
            "M": self.meta,
        }

    @classmethod
    def from_item(cls, item: ida.cexpr_t, lvars: ida.lvars_t) -> "Ptr":
        node_id = item.obj_id
        x = parse_hexrays_expression(item.x, lvars)
        return cls(node_id=node_id, x=x, ptrsize=item.ptrsize)

    def __repr__(self):
        return f"{type(self).__name__} (size: {self.ptrsize}, x: {self.x})"


class Ref(UnaryExpression):
    """cot_ref (&x)"""

    meta = ida.cot_ref


class Postinc(UnaryExpression):
    """cot_postinc (x++)"""

    meta = ida.cot_postinc


class Postdec(UnaryExpression):
    """cot_postdec (x--)"""

    meta = ida.cot_postdec


class Preinc(UnaryExpression):
    """cot_preinc (++x)"""

    meta = ida.cot_preinc


class Predec(UnaryExpression):
    """cot_predec (--x)"""

    meta = ida.cot_predec


class Call(Expression):
    """cot_call (x(...))"""

    meta = ida.cot_call

    class Arg(Expression):
        """An argument"""

        def __init__(self, item: ida.carg_t, lvars: ida.lvars_t):
            # FIXME: This is technically an expression,
            # so I think there should be more parsering
            self.node_id = item.obj_id
            self.is_vararg = item.is_vararg
            self.idx = None
            self.name = None
            if item.v:
                self.idx = item.v.idx
                self.name = lvars[self.idx].name
            self.formal_type: TypeInfo = TypeLib.parse_ida_type(item.formal_type)

        def to_json(self):
            return {
                "id": self.node_id,
                "va": self.is_vararg,
                "i": self.idx,
                "n": self.name,
            }

        def __repr__(self):
            if self.is_vararg:
                return "Vararg"
            return f"{self.formal_type} {self.name}"

    def __init__(self, node_id: int, x: Expression, a: t.List["Call.Arg"]):
        self.node_id = node_id
        self.x = x
        self.a = a

    def to_json(self):
        x = self.x.to_json()
        a = [ arg.to_json() for arg in self.a ]
        return {
            "id": self.node_id,
            "x": x,
            "a": a,
            "M": self.meta,
        }

    @classmethod
    def from_item(cls, item: ida.cexpr_t, lvars: ida.lvars_t) -> "Call":
        node_id = item.obj_id
        x = parse_hexrays_expression(item.x, lvars)
        a = [Call.Arg(i, lvars) for i in item.a]
        return cls(node_id=node_id, x=x, a=a)

    def __repr__(self):
        return f"{type(self).__name__} (x: {self.x}, args: {self.a})"


class Idx(BinaryExpression):
    """cot_idx (x[y])"""

    meta = ida.cot_idx


class Memref(Expression):
    """cot_memref (x.m)"""

    meta = ida.cot_memref

    def __init__(self, node_id: int, x: Expression, m: int):
        self.node_id = node_id
        self.x = x
        self.m = m

    def to_json(self):
        x = self.x.to_json()
        return {
            "id": self.node_id,
            "x": x,
            "m": self.m,
            "M": self.meta,
        }

    @classmethod
    def from_item(cls, item: ida.cexpr_t, lvars: ida.lvars_t) -> "Memref":
        node_id = item.obj_id
        x = parse_hexrays_expression(item.x, lvars)
        return cls(node_id=node_id, x=x, m=item.m)

    def __repr__(self):
        return f"Memref (x: {self.x}, m: {self.m})"


class Memptr(Expression):
    """cot_memptr (x->m, access size in 'ptrsize')"""

    meta = ida.cot_memptr

    def __init__(self, node_id: int, x: Expression, m: int, ptrsize: int):
        self.node_id = node_id
        self.x = x
        self.m = m
        self.ptrsize = ptrsize

    def to_json(self):
        x = self.x.to_json()
        return {
            "id": self.node_id,
            "x": x,
            "m": self.m,
            "p": self.ptrsize,
            "M": self.meta,
        }

    @classmethod
    def from_item(cls, item: ida.cexpr_t, lvars: ida.lvars_t) -> "Memptr":
        node_id = item.obj_id
        x = parse_hexrays_expression(item.x, lvars)
        return cls(node_id=node_id, x=x, m=item.m, ptrsize=item.ptrsize)

    def __repr__(self):
        return f"Memptr (x: {self.x}, m: {self.m} size: {self.ptrsize})"


class Num(Expression):
    """cot_num (n: numeric constant)"""

    meta = ida.cot_num

    def __init__(self, node_id: int, n: int):
        self.node_id = node_id
        self.n = n

    def to_json(self):
        return {
            "id": self.node_id,
            "n": self.n,
            "M": self.meta,
        }

    @classmethod
    def from_item(cls, item: ida.cexpr_t, lvars: ida.lvars_t) -> "Num":
        node_id = item.obj_id
        return cls(node_id=node_id, n=item.n._value)

    def __repr__(self):
        return f"Num ({self.n})"

class Fnum(Expression):
    """cot_fnum (fpc: floating point constant)"""

    meta = ida.cot_fnum

    def __init__(self, node_id: int, fpc: int):
        self.node_id = node_id
        self.fpc = fpc

    def to_json(self):
        return {
            "id": self.node_id,
            "f": self.fpc,
            "M": self.meta,
        }

    @classmethod
    def from_item(cls, item: ida.cexpr_t, lvars: ida.lvars_t) -> "Fnum":
        node_id = item.obj_id
        return cls(node_id=node_id, fpc=item.fpc.fnum)

    def __repr__(self):
        return f"Fnum ({self.fpc})"

class Str(Expression):
    """cot_str (string constant)"""

    meta = ida.cot_str

    def __init__(self, node_id: int, string: str):
        self.node_id = node_id
        self.string = string

    def to_json(self):
        return {
            "id": self.node_id,
            "s": self.string,
            "M": self.meta,
        }

    @classmethod
    def from_item(cls, item: ida.cexpr_t, lvars: ida.lvars_t) -> "Str":
        node_id = item.obj_id
        return cls(node_id=node_id, string=item.string)

    def __repr__(self):
        return f"Str ({self.string})"

class Obj(Expression):
    """cot_obj (obj_ea)"""

    meta = ida.cot_obj

    def __init__(self, node_id: int, obj_ea: int):
        self.node_id = node_id
        self.obj_ea = obj_ea
        func_name = ida.get_func_name(self.obj_ea)
        self.func_name = func_name if func_name else None

    def to_json(self):
        return {
            "id": self.node_id,
            "e": self.obj_ea,
            "n": self.func_name,
            "M": self.meta,
        }

    @classmethod
    def from_item(cls, item: ida.cexpr_t, lvars: ida.lvars_t) -> "Obj":
        node_id = item.obj_id
        return cls(node_id=node_id, obj_ea=item.obj_ea)

    def __repr__(self):
        # If this is a function show its name, otherwise show the offset
        return self.func_name if self.func_name else f"Obj ({self.obj_ea})"


class Var(Expression):
    """cot_var (v)

    idx is an offset into lvars_t, used by the AST to generate a Variable object.
    """

    meta = ida.cot_var

    def __init__(self, node_id: int, idx: int, lvars: ida.lvars_t):
        self.node_id = node_id
        self.idx = idx
        self.lvar = lvars[self.idx]

    def to_json(self):
        return {
            "id": self.node_id,
            "i": self.idx,
            "n": self.lvar.name,
            "M": self.meta,
        }

    @classmethod
    def from_item(cls, item: ida.cexpr_t, lvars: ida.lvars_t) -> "Var":
        node_id = item.obj_id
        idx = item.v.idx
        return cls(node_id=node_id, idx=idx, lvars=lvars)

    def __repr__(self):
        # FIXME
        # typ = TypeLib.parse_ida_type(self.lvar.type)
        return f"{self.lvar.name}"

class Insn(Expression):
    """cot_insn (instruction in expression, internal representation only)"""

    meta = ida.cot_insn


class Sizeof(UnaryExpression):
    """cot_sizeof (sizeof(x))"""

    meta = ida.cot_sizeof


class Helper(Expression):
    """cot_helper (arbitrary name)"""

    meta = ida.cot_helper


class Type(Expression):
    """cot_type (arbitrary type)"""

    meta = ida.cot_type

    def __init__(self, node_id: int, typ: TypeInfo):
        self.node_id = node_id
        self.typ = typ

    def to_json(self):
        return {
            "id": self.node_id,
            "t": self.typ,
            "M": self.meta,
        }

    @classmethod
    def from_item(cls, item: ida.cexpr_t, lvars: ida.lvars_t) -> "Type":
        node_id = item.obj_id
        return cls(node_id=node_id, typ=TypeLib.parse_ida_type(item.type))

    def __repr__(self):
        return f"Type ({self.typ})"


########## Statements ##########


class Block(Statement):
    """cit_block"""

    meta = ida.cit_block

    def __init__(self, node_id: int, statements: t.List[Statement]):
        self.node_id = node_id
        self.statements = statements

    def to_json(self):
        return {
            "id": self.node_id,
            "M": self.meta,
            "s": [ stmt.to_json() for stmt in self.statements ],
        }

    @classmethod
    def from_item(cls, item: ida.citem_t, lvars: ida.lvars_t) -> "Block":
        node_id = item.obj_id
        # Ignore cit_empty
        statements = [
            parse_hexrays_statement(i, lvars) for i in item.cblock if i.op != ida.cit_empty
        ]
        return cls(node_id=node_id, statements=statements)

    def __repr__(self):
        return f"{self.statements}"

class ExprStatement(Statement):
    """A statement that has an expression"""

    def __init__(self, node_id: int, expr: Expression):
        self.node_id = node_id
        self.expr = expr


class If(ExprStatement):
    """cit_if"""

    meta = ida.cit_if

    def __init__(
        self,
        node_id: int,
        expr: Expression,
        ithen: t.Optional[Statement],
        ielse: t.Optional[Statement],
    ):
        self.node_id = node_id
        self.ithen = ithen
        self.ielse = ielse
        self.expr = expr

    def to_json(self):
        ithen = self.ithen.to_json() if self.ithen else None
        ielse = self.ielse.to_json() if self.ielse else None
        expr = self.expr.to_json()
        return {
            "id": self.node_id,
            "M": self.meta,
            "e": expr,
            "t": ithen,
            "f": ielse,
        }

    @classmethod
    def from_item(cls, item: ida.citem_t, lvars: ida.lvars_t) -> "If":
        node_id = item.obj_id
        stmt = item.cif
        ithen = parse_hexrays_statement(stmt.ithen, lvars) if stmt.ithen else None
        ielse = parse_hexrays_statement(stmt.ielse, lvars) if stmt.ielse else None
        expr = parse_hexrays_expression(stmt.expr, lvars)
        return cls(node_id=node_id, ithen=ithen, ielse=ielse, expr=expr)

    def __repr__(self):
        return f"If (expr: {self.expr}, ithen: {self.ithen}, ielse: {self.ielse})"

class Loop(ExprStatement):
    """A generic loop. body is the loop body, while expr is the guard expression"""

    def __init__(self, node_id: int, body: Statement, expr: Expression):
        self.node_id = node_id
        self.body = body
        self.expr = expr


class Do(Loop):
    """cit_do"""

    meta = ida.cit_do

    def to_json(self):
        body = self.body.to_json()
        expr = self.expr.to_json()
        return {
            "id": self.node_id,
            "M": self.meta,
            "e": expr,
            "b": body,
        }

    @classmethod
    def from_item(cls, item: ida.citem_t, lvars: ida.lvars_t) -> "Loop":
        node_id = item.obj_id
        stmt = item.cdo
        body = parse_hexrays_statement(stmt.body, lvars)
        expr = parse_hexrays_expression(stmt.expr, lvars)
        return cls(node_id=node_id, body=body, expr=expr)

    def __repr__(self):
        return f"Do (expr: {self.expr}, body: {self.body})"

class While(Loop):
    """cit_while"""

    meta = ida.cit_while

    def to_json(self):
        body = self.body.to_json()
        expr = self.expr.to_json()
        return {
            "id": self.node_id,
            "M": self.meta,
            "e": expr,
            "b": body,
        }

    @classmethod
    def from_item(cls, item: ida.citem_t, lvars: ida.lvars_t) -> "Loop":
        node_id = item.obj_id
        stmt = item.cwhile
        body = parse_hexrays_statement(stmt.body, lvars)
        expr = parse_hexrays_expression(stmt.expr, lvars)
        return cls(node_id=node_id, body=body, expr=expr)

    def __repr__(self):
        return f"While (expr: {self.expr}, body: {self.body})"


class For(Loop):
    """cit_for"""

    meta = ida.cit_for

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

    def to_json(self):
        body = self.body.to_json()
        expr = self.expr.to_json()
        init = self.init.to_json()
        step = self.step.to_json()
        return {
            "id": self.node_id,
            "M": self.meta,
            "e": expr,
            "b": body,
            "i": init,
            "s": step,
        }

    @classmethod
    def from_item(cls, item: ida.citem_t, lvars: ida.lvars_t) -> "For":
        node_id = item.obj_id
        stmt = item.cfor
        body = parse_hexrays_statement(stmt.body, lvars)
        expr = parse_hexrays_expression(stmt.expr, lvars)
        init = parse_hexrays_expression(stmt.init, lvars)
        step = parse_hexrays_expression(stmt.step, lvars)
        return cls(node_id=node_id, body=body, expr=expr, init=init, step=step)

    def __repr__(self):
        return f"For (expr: {self.expr}, init: {self.init}, step: {self.step}, body: {self.body})"

class Switch(ExprStatement):
    """cit_switch"""

    meta = ida.cit_switch

    class Case(Statement):
        def __init__(self, item: ida.ccase_t, lvars: ida.lvars_t):
            self.node_id = item.obj_id
            # Have to manually convert this into a list
            self.values = list(item.values)
            self.stmt = parse_hexrays_statement(item, lvars)

        def to_json(self):
            return {
                "id": self.node_id,
                "v": self.values,
                "s": self.stmt.to_json(),
            }

        def __repr__(self):
            return f"Case (values: {self.values}, stmt: {self.stmt})"

    def __init__(
        self,
        node_id: int,
        expr: Expression,
        # mvnf: int,
        cases: t.List["Case"],
    ):
        super().__init__(node_id, expr)
        # self.mvnf = mvnf
        self.cases = cases

    def to_json(self):
        expr = self.expr.to_json()
        cases = [ case.to_json() for case in self.cases ]
        return {
            "id": self.node_id,
            "M": self.meta,
            "e": expr,
            "c": cases,
        }

    @classmethod
    def from_item(cls, item: ida.citem_t, lvars: ida.lvars_t) -> "Switch":
        node_id = item.obj_id
        stmt = item.cswitch
        expr = parse_hexrays_expression(stmt.expr, lvars)
        # mvnf = ...
        cases = [Switch.Case(c, lvars) for c in stmt.cases]
        return cls(
            node_id=node_id,
            expr=expr,
            # mvnf=item.mvnf,
            cases=cases,
        )

    def __repr__(self):
        return f"Switch: (expr: {self.expr}, cases: {self.cases})"


class Return(ExprStatement):
    """cit_return"""

    meta = ida.cit_return

    @classmethod
    def from_item(cls, item: ida.citem_t, lvars: ida.lvars_t) -> "Return":
        node_id = item.obj_id
        stmt = item.creturn
        expr = parse_hexrays_expression(stmt.expr, lvars)
        return cls(node_id=node_id, expr=expr)

    def to_json(self):
        expr = self.expr.to_json()
        return {
            "id": self.node_id,
            "M": self.meta,
            "e": expr,
        }

    def __repr__(self):
        return f"Return (expr: {self.expr})"

class Goto(Statement):
    """cit_goto"""

    meta = ida.cit_goto

    def __init__(self, node_id: int, label_num: int):
        self.node_id = node_id
        self.label_num = label_num

    def to_json(self):
        return {
            "id": self.node_id,
            "M": self.meta,
            "l": self.label_num,
        }

    @classmethod
    def from_item(cls, item: ida.citem_t, lvars: ida.lvars_t) -> "Goto":
        node_id = item.obj_id
        stmt = item.cgoto
        return cls(node_id=node_id, label_num=stmt.label_num)

    def __repr__(self):
        return f"Goto (label: {self.label_num})"

class Asm(Statement):
    """cit_asm, not supported"""

    meta = ida.cit_asm


class Break(Statement):
    """cit_break"""

    meta = ida.cit_break


class Continue(Statement):
    """cit_continue"""

    meta = ida.cit_continue


#################### Utilities ####################

def _get_all_classes(cls) -> set:
    all_classes = set()
    for subclass in cls.__subclasses__():
        if hasattr(subclass, "meta"):
            all_classes.add(subclass)
        all_classes.update(_get_all_classes(subclass))
    return all_classes


all_classes = _get_all_classes(Statement)
expressions = { c.meta: c for c in all_classes if c.meta <= ida.cot_last }
statements = { c.meta: c for c in all_classes if c.meta > ida.cot_last }

def parse_hexrays_expression(expr: ida.cexpr_t, lvars: ida.lvars_t) -> Expression:
    """Parses a HexRays expression and returns an Expression object"""
    return expressions[expr.op].from_item(expr, lvars)  # type: ignore


def parse_hexrays_statement(stmt: ida.cinsn_t, lvars: ida.lvars_t) -> Statement:
    """Parses a HexRays statement and returns a Statement object"""
    if stmt.op == ida.cit_expr:
        return parse_hexrays_expression(stmt.cexpr, lvars)
    return statements[stmt.op].from_item(stmt, lvars)  # type: ignore


#################### AST ####################
class AST:
    def __init__(self, function: ida.cfunc_t):
        print("Creating AST")
        self.function = function
        print(ida.get_func_name(self.function.entry_ea))
        self.root = parse_hexrays_statement(function.body, function.lvars)
        print(self.root)
        print("Done creating AST")
        print(self.root.to_json())
