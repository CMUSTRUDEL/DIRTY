import pytest

from dirty.baselines import copy_decompiler, most_common, most_common_decomp
from dirty.model import (
    beam,
    decoder,
    encoder,
    model,
    simple_decoder,
    xfmr_decoder,
    xfmr_mem_encoder,
    xfmr_sequential_encoder,
    xfmr_subtype_decoder,
)
from dirty.utils import (
    case_study,
    code_processing,
    compute_mi,
    dataset,
    dataset_statistics,
    evaluate,
    lexer,
    preprocess,
    util,
    vocab,
)


@pytest.mark.commit
def test_imports():
    assert True
