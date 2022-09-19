"""
Global ToDos:

    - at which point is spatial graph built in squidy and how is this communicated to these methods?
"""

import anndata

from ncem.tools.fit.backend.testing import test_linear_ncem, test_differential_ncem
from ncem.tools.fit.constants import VARM_KEY_PARAMS
from ncem.utils.ols_fit import ols_fit


def differential_ncem(adata: anndata.AnnData, formula: str, key_type: str, key_differential):
    """
    Fit a differential NCEM based on an adata instance and save fits in instance.

    Saves fits and Wald test output into instance.
    TODO requires spatial graph to have been built or coordinates to be in fixed slots?

    Args:
        adata: AnnData instance with data and annotation.
        formula: Description of batch covariates as linear model. Do not include intercept, cell type, niche, or the
            differential term as this is automatically added.
        key_differential: Key of condition annotation in .obs. This will be used for testing.
        key_type: Key of type annotation in .obs.

    Returns:

    """
    # TODO
    adata = None
    term_condition = None
    term_type = None
    adata = test_differential_ncem(adata=adata, term_condition=term_condition, term_type=term_type)
    return adata


def differential_ncem_deconvoluted(adata: anndata.AnnData, formula: str, key_deconvolution: str):
    """
    Fit a differential NCEM based on deconvoluted data in an adata instance and save fits in instance.

    Saves fits and Wald test output into instance.
    TODO requires spatial graph to have been built or coordinates to be in fixed slots?

    Args:
        adata: AnnData instance with data and annotation.
        formula: Description of batch covariates as linear model. Do not include intercept, cell type, niche, or the
            differential term as this is automatically added.
        key_deconvolution: Key of type deconvolution in .obsm.
        key_differential: Key of condition annotation in .obs. This will be used for testing.

    Returns:

    """
    # TODO
    adata = None
    term_condition = None
    term_type = None
    adata = test_differential_ncem(adata=adata, term_condition=term_condition, term_type=term_type)
    return adata


def linear_ncem(adata: anndata.AnnData, formula: str, key_type: str):
    """
    Fit a linear NCEM based on an adata instance and save fits in instance.

    Saves fits and Wald test output into instance.
    TODO requires spatial graph to have been built or coordinates to be in fixed slots?

    Args:
        adata: AnnData instance with data and annotation.
        formula: Description of batch covariates as linear model. Do not include intercept, cell type, or niche as
            this is automatically added.
        key_type: Key of type annotation in .obs.

    Returns:

    """
    dmat = None
    ols = ols_fit(x_=dmat, y_=y)
    params = ols.squeeze()
    adata.varm[VARM_KEY_PARAMS] = params
    adata = test_linear_ncem(adata=adata, term_type=key_type)
    return adata


def linear_ncem_deconvoluted(adata: anndata.AnnData, formula: str, key_deconvolution: str):
    """
    Fit a linear NCEM based on deconvoluted data in an adata instance and save fits in instance.

    Saves fits and Wald test output into instance.
    TODO requires spatial graph to have been built or coordinates to be in fixed slots?

    Args:
        adata: AnnData instance with data and annotation.
        formula: Description of batch covariates as linear model. Do not include intercept, cell type, or niche as
            this is automatically added.
        key_deconvolution: Key of type deconvolution in .obsm.

    Returns:

    """
    dmat = None
    ols = ols_fit(x_=dmat, y_=y)
    params = ols.squeeze()
    adata.varm[VARM_KEY_PARAMS] = params
    adata = test_linear_ncem(adata=adata, term_type=term_type)
    return adata
