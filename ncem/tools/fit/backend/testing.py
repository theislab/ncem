from typing import List

import anndata
from diffxpy.testing.correction import correct

from ncem.tools.fit.constants import VARM_KEY_PARAMS, VARM_KEY_PVALs, VARM_KEY_FDR_PVALs
from ncem.utils.wald_test import get_fim_inv, wald_test


def _test_base(adata: anndata.AnnData, coef_to_test: List[str]) -> anndata.AnnData:
    """
    Base test function independent of NCEM variant that interfaces Wald test.

    Args:
        adata: AnnData instance with fits saved.
        coef_to_test:

    Returns:
        Anndata instance with test output saved. Test output is one p-value, FDR-corrected p-value and log-fold change
        per gene and type x type pair.

    """
    pvals = None
    adata.varm[VARM_KEY_PVALs] = pvals
    fdr_pvals = correct(pvals=pvals, method="bh")
    adata.varm[VARM_KEY_FDR_PVALs] = fdr_pvals
    return adata


def test_linear_ncem(adata: anndata.AnnData, term_type: str) -> anndata.AnnData:
    """
    Test for linear NCEM.

    Args:
        adata: AnnData instance with fits saved.
        term_type: Name of cell type term used in formula, this is used to extract type coefficients that will
            be used to define couplings.

    Returns:
        Anndata instance with test output saved. Test output is one p-value, FDR-corrected p-value and log-fold change
        per gene and type x type pair. The test signifies the coupling between any two cell types.

    """
    # TODO
    coef_to_test = None
    adata = _test_base(adata=adata, coef_to_test=coef_to_test)
    return adata


def test_differential_ncem(adata: anndata.AnnData, term_condition: str, term_type: str) -> anndata.AnnData:
    """
    Test for differential NCEM.

    Args:
        adata: AnnData instance with fits saved.
        term_condition: Name of condition term used in formula, this is used to extract condition coefficients that will
            be tested.
        term_type: Name of cell type term used in formula, this is used to extract type coefficients that will
            be used to define couplings.

    Returns:
        Anndata instance with test output saved. Test output is one p-value, FDR-corrected p-value and log-fold change
        per gene and type x type pair. The test signifies the differential coupling between any two cell types.

    """
    # TODO
    coef_to_test = None
    adata = _test_base(adata=adata, coef_to_test=coef_to_test)
    return adata
