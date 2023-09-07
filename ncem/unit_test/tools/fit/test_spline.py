import anndata
import numpy as np
import pandas as pd
import pytest

from ncem.tl.fit.backend.utils import read_uns
from ncem.tl.fit.constants import UNS_KEY_CELL_TYPES, VARM_KEY_FDR_PVALS_SPLINE, VARM_KEY_PVALS_SPLINE
from ncem.tl.fit.glm import (
    get_spline_interpolation,
    spline_differential_ncem,
    spline_differential_ncem_deconvoluted,
    spline_linear_ncem,
    spline_linear_ncem_deconvoluted,
)
from ncem.unit_test.data_for_tests import KEY_1D, KEY_ADJACENCY, KEY_BATCH, KEY_COND, KEY_DECONV, KEY_TYPE, get_adata_1d
from ncem.unit_test.tools.fit.test_glm import _assert_slot_dimension, _assert_slot_domain, _assert_slot_keys

HYPERPARAMS_SPLINE = {"df": 3, "spline_family": "cr", "key_1d_coord": KEY_1D}


def _assert_slot_keys_spline(adata: anndata.AnnData, differential: bool):
    """Asserts that all relevant slots in adata were set in NCEM method."""
    _assert_slot_keys(adata, differential=differential)
    # Spline-specific:
    assert VARM_KEY_PVALS_SPLINE in adata.varm.keys()
    assert VARM_KEY_FDR_PVALS_SPLINE in adata.varm.keys()


def _assert_slot_domain_spline(adata: anndata.AnnData, differential: bool):
    """Asserts numerical domain of slot entries, e.g. positive."""
    _assert_slot_domain(adata, differential=differential)
    # Spline-specific:
    assert np.all(adata.varm[VARM_KEY_PVALS_SPLINE] >= 0.0) and np.all(adata.varm[VARM_KEY_PVALS_SPLINE] <= 1.0)
    assert np.all(adata.varm[VARM_KEY_FDR_PVALS_SPLINE] >= 0.0) and np.all(adata.varm[VARM_KEY_FDR_PVALS_SPLINE] <= 1.0)


def _assert_slot_dimension_spline(adata: anndata.AnnData, differential: bool):
    """Asserts that slots have correct dimensions and dimension names."""
    _assert_slot_dimension(adata, differential=differential)


def _test_spline_extrapolation(adata):
    """Checks extrapolation based on spline fit."""
    cell_types = read_uns(adata, k=UNS_KEY_CELL_TYPES)
    cell_type = cell_types[1]
    genes = adata.var_names[[0]]
    x = get_spline_interpolation(adata=adata, genes=genes, cell_type=cell_type)
    assert isinstance(x, pd.DataFrame)
    assert x.shape[1] == len(genes), x.shape
    assert np.all(x.columns == genes), x.columns
    assert x.shape[0] == 100, x.shape


@pytest.mark.parametrize("n_conds", [2, 4])
@pytest.mark.parametrize("confounders", [[], [KEY_BATCH]])
def test_spline_differential_ncem(n_conds, confounders):
    adata = get_adata_1d(simulate_deconvoluted=False, n_conds=n_conds)
    adata = spline_differential_ncem(
        adata=adata,
        formula=f"~0",
        key_graph=KEY_ADJACENCY,
        key_type="type",
        key_differential=KEY_COND,
        type_specific_confounders=confounders,
        **HYPERPARAMS_SPLINE,
    )
    _assert_slot_keys_spline(adata=adata, differential=True)
    _assert_slot_domain_spline(adata=adata, differential=True)
    _assert_slot_dimension_spline(adata=adata, differential=True)
    _test_spline_extrapolation(adata=adata)


@pytest.mark.parametrize("n_conds", [2, 4])
@pytest.mark.parametrize("confounders", [[], [KEY_BATCH]])
def test_spline_differential_ncem_deconvoluted(n_conds, confounders):
    adata = get_adata_1d(simulate_deconvoluted=True, n_conds=n_conds)
    adata = spline_differential_ncem_deconvoluted(
        adata=adata,
        formula=f"~0",
        key_differential=KEY_COND,
        key_deconvolution=KEY_DECONV,
        type_specific_confounders=confounders,
        **HYPERPARAMS_SPLINE,
    )
    _assert_slot_keys_spline(adata=adata, differential=True)
    _assert_slot_domain_spline(adata=adata, differential=True)
    _assert_slot_dimension_spline(adata=adata, differential=True)
    _test_spline_extrapolation(adata=adata)


@pytest.mark.parametrize("confounders", [[], [KEY_BATCH]])
def test_spline_linear_ncem(confounders):
    adata = get_adata_1d(simulate_deconvoluted=False)
    adata = spline_linear_ncem(
        adata=adata,
        formula=f"~0",
        key_graph=KEY_ADJACENCY,
        key_type=KEY_TYPE,
        type_specific_confounders=confounders,
        **HYPERPARAMS_SPLINE,
    )
    _assert_slot_keys_spline(adata=adata, differential=False)
    _assert_slot_domain_spline(adata=adata, differential=False)
    _assert_slot_dimension_spline(adata=adata, differential=False)
    _test_spline_extrapolation(adata=adata)


@pytest.mark.parametrize("confounders", [[], [KEY_BATCH]])
def test_spline_linear_ncem_deconvoluted(confounders):
    adata = get_adata_1d(simulate_deconvoluted=True)
    adata = spline_linear_ncem_deconvoluted(
        adata=adata,
        formula=f"~0",
        key_deconvolution=KEY_DECONV,
        type_specific_confounders=confounders,
        **HYPERPARAMS_SPLINE,
    )
    _assert_slot_keys_spline(adata=adata, differential=False)
    _assert_slot_domain_spline(adata=adata, differential=False)
    _assert_slot_dimension_spline(adata=adata, differential=False)
    _test_spline_extrapolation(adata=adata)
