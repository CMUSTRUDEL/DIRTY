import pytest

from dirty.baselines import copy_decompiler, most_common_decomp, most_common
from dirty.model import (
    beam, decoder, encoder, model, simple_decoder,
    xfmr_decoder, xfmr_mem_encoder, xfmr_sequential_encoder, xfmr_subtype_decoder
)
from dirty.utils import (
    case_study,
    code_processing,
    compute_mi,
    dataset_statistics,
    dataset,
    evaluate,
    lexer,
    preprocess,
    util,
    vocab
)

@pytest.mark.commit
def test_imports():
    assert True