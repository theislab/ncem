import numpy as np
import pytest

from ncem.tools.fit.constants import PREFIX_INDEX, PREFIX_NEIGHBOR, VARM_KEY_FDR_PVALS, VARM_KEY_PARAMS, \
    VARM_KEY_PVALS, VARM_KEY_TESTED_PARAMS
from ncem.tools.fit.glm import differential_ncem, differential_ncem_deconvoluted, linear_ncem, linear_ncem_deconvoluted

from ncem.unit_test.data_for_tests import get_adata, KEY_ADJACENCY, KEY_BATCH, KEY_COND, KEY_DECONV, KEY_TYPE


def _assert_slot_keys(adata):
    """Asserts that all relevant slots in adata were set in NCEM method."""
    assert VARM_KEY_PARAMS in adata.varm.keys()
    assert VARM_KEY_PVALS in adata.varm.keys()
    assert VARM_KEY_FDR_PVALS in adata.varm.keys()
    assert VARM_KEY_TESTED_PARAMS in adata.varm.keys()


def _assert_slot_domain(adata):
    """Asserts numerical domain of slot entries, e.g. positive."""
    assert np.all(adata.varm[VARM_KEY_PVALS] >= 0.) and np.all(adata.varm[VARM_KEY_PVALS] <= 1.)
    assert np.all(adata.varm[VARM_KEY_FDR_PVALS] >= 0.) and np.all(adata.varm[VARM_KEY_FDR_PVALS] <= 1.)


def _assert_slot_dimension(adata):
    """Asserts that slots have correct dimensions and dimension names."""
    if KEY_DECONV in adata.obsm.keys():
        cell_types = np.sort(np.unique(adata.obsm[KEY_DECONV].columns))
    else:
        cell_types = np.sort(np.unique(adata.obs[KEY_TYPE].values))
    couplings = [f"{PREFIX_INDEX}{x}:{PREFIX_NEIGHBOR}{y}" for y in cell_types for x in cell_types]
    assert np.all(adata.varm[VARM_KEY_PVALS].shape[1] == len(couplings))
    assert np.all(adata.varm[VARM_KEY_FDR_PVALS].shape[1] == len(couplings))
    assert np.all(adata.varm[VARM_KEY_TESTED_PARAMS].shape[1] == len(couplings))
    assert np.all(np.sort(couplings) == np.sort(list(adata.varm[VARM_KEY_PVALS].columns)))
    assert np.all(np.sort(couplings) == np.sort(list(adata.varm[VARM_KEY_FDR_PVALS].columns)))
    assert np.all(np.sort(couplings) == np.sort(list(adata.varm[VARM_KEY_TESTED_PARAMS].columns)))


@pytest.mark.parametrize("n_conds", [2, 4])
@pytest.mark.parametrize("confounders", [[], [KEY_BATCH]])
def test_differential_ncem(n_conds, confounders):
    adata = get_adata(simulate_deconvoluted=False, n_conds=n_conds)
    adata = differential_ncem(adata=adata, formula=f"~0", key_graph=KEY_ADJACENCY, key_type="type",
                              key_differential=KEY_COND, type_specific_confounders=confounders)
    _assert_slot_keys(adata=adata)
    _assert_slot_domain(adata=adata)
    _assert_slot_dimension(adata=adata)


@pytest.mark.parametrize("n_conds", [2, 4])
@pytest.mark.parametrize("confounders", [[], [KEY_BATCH]])
def test_differential_ncem_deconvoluted(n_conds, confounders):
    adata = get_adata(simulate_deconvoluted=True, n_conds=n_conds)
    adata = differential_ncem_deconvoluted(adata=adata, formula=f"~0", key_differential=KEY_COND,
                                           key_deconvolution=KEY_DECONV, type_specific_confounders=confounders)
    _assert_slot_keys(adata=adata)
    _assert_slot_domain(adata=adata)
    _assert_slot_dimension(adata=adata)


@pytest.mark.parametrize("confounders", [[], [KEY_BATCH]])
def test_linear_ncem(confounders):
    adata = get_adata(simulate_deconvoluted=False)
    adata = linear_ncem(adata=adata, formula=f"~0", key_graph=KEY_ADJACENCY, key_type=KEY_TYPE,
                        type_specific_confounders=confounders)
    _assert_slot_keys(adata=adata)
    _assert_slot_domain(adata=adata)
    _assert_slot_dimension(adata=adata)


@pytest.mark.parametrize("confounders", [[], [KEY_BATCH]])
def test_linear_ncem_deconvoluted(confounders):
    adata = get_adata(simulate_deconvoluted=True)
    adata = linear_ncem_deconvoluted(adata=adata, formula=f"~0", key_deconvolution=KEY_DECONV,
                                     type_specific_confounders=confounders)
    _assert_slot_keys(adata=adata)
    _assert_slot_domain(adata=adata)
    _assert_slot_dimension(adata=adata)
