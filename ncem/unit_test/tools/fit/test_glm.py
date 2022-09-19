import numpy as np

from ncem.tools.fit.constants import VARM_KEY_PARAMS, VARM_KEY_PVALs, VARM_KEY_FDR_PVALs
from ncem.tools.fit.glm import differential_ncem, differential_ncem_deconvoluted, linear_ncem, linear_ncem_deconvoluted

from ncem.unit_test.data_for_tests import get_adata, KEY_BATCH, KEY_COND, KEY_DECONV, KEY_TYPE


def _slot_asserts(adata):
    """Asserts that all relevant slots in adata were set in NCEM method."""
    assert VARM_KEY_PARAMS in adata.var.keys()
    assert VARM_KEY_PVALs in adata.var.keys()
    assert VARM_KEY_FDR_PVALs in adata.var.keys()


def _slot_domain(adata):
    """Asserts numerical domain of slot entries, e.g. positive."""
    assert np.all(adata.var[VARM_KEY_PVALs] >= 0.) and np.all(adata.var[VARM_KEY_PVALs] <= 1.)
    assert np.all(adata.var[VARM_KEY_FDR_PVALs] >= 0.) and np.all(adata.var[VARM_KEY_FDR_PVALs] <= 1.)


def test_differential_ncem():
    adata = get_adata(report_deconvolution=False)
    adata = differential_ncem(adata=adata, formula=f"~0+{KEY_BATCH}", key_type="type", key_differential=KEY_COND)
    _slot_asserts(adata=adata)
    _slot_domain(adata=adata)


def test_differential_ncem_deconvoluted():
    adata = get_adata(report_deconvolution=True)
    adata = differential_ncem_deconvoluted(adata=adata, formula=f"~0+{KEY_BATCH}", key_differential=KEY_COND,
                                           key_deconvolution=KEY_DECONV)
    _slot_asserts(adata=adata)
    _slot_domain(adata=adata)


def test_linear_ncem():
    adata = get_adata(report_deconvolution=False)
    adata = linear_ncem(adata=adata, formula=f"~0+{KEY_BATCH}", key_type=KEY_TYPE)
    _slot_asserts(adata=adata)
    _slot_domain(adata=adata)


def test_linear_ncem_deconvoluted():
    adata = get_adata(report_deconvolution=True)
    adata = linear_ncem_deconvoluted(adata=adata, formula=f"~0+{KEY_BATCH}", key_deconvolution=KEY_DECONV)
    _slot_asserts(adata=adata)
    _slot_domain(adata=adata)
