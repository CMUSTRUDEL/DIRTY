import pytest

import csvnpm
from csvnpm.binary import dire_types, function, ida_ast
from csvnpm.dataset_gen import generate, lexer
from csvnpm.ida import idaapi
from csvnpm.dataset_gen.decompiler import collect, debug, dump_trees


@pytest.mark.commit
def test_donothing():
    assert True
