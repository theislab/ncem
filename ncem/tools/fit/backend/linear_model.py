"""
Global ToDos:

    - at which point is spatial graph built in squidy and how is this communicated to these methods?
"""
from typing import List

import anndata
import numpy as np
import pandas as pd

from ncem.tools.fit.backend.design_matrix import extend_formula_ncem, extend_formula_differential_ncem, \
    get_obs_niche_from_graph, get_dmat_from_deconvoluted, get_dmat_from_obs
from ncem.tools.fit.backend.ols_fit import ols_fit
from ncem.tools.fit.backend.testing import test_linear_ncem, test_differential_ncem
from ncem.tools.fit.constants import VARM_KEY_PARAMS, OBSM_KEY_NICHE


def _validate_formula(formula: str, auto_keys: List[str] = []):
    # Check formula format:
    assert formula.startswith("~0+"), "formula describing batch needs to start with '~0+'"


def differential_ncem(adata: anndata.AnnData, formula: str, key_differential: str, key_graph: str, key_type: str):
    """
    Fit a differential NCEM based on an adata instance and save fits in instance.

    Saves fits and Wald test output into instance.
    TODO requires spatial graph to have been built or coordinates to be in fixed slots?

    Args:
        adata: AnnData instance with data and annotation.
        formula: Description of batch covariates as linear model. Do not include intercept, cell type, niche, or the
            differential term as this is automatically added.
        key_differential: Key of condition annotation in .obs. This will be used for testing.
        key_graph: Key of spatial neighborhood graph in .obsp.
        key_type: Key of type annotation in .obs.

    Returns:

    """
    # TODO extract obs_niche into obsm, eg. using squidpy or using precomputed.
    _validate_formula(formula=formula, auto_keys=[key_differential])
    groups = np.sort(np.unique(adata.obs[key_type].values)).tolist()
    formula = extend_formula_differential_ncem(formula=formula, groups=groups, key_cond=key_differential)
    adata.obsm[OBSM_KEY_NICHE] = get_obs_niche_from_graph(adata=adata, obs_key_type=key_type,
                                                          obsp_key_graph=key_graph, marginalisation="binary")
    dmat = get_dmat_from_obs(formula=formula, key_type=key_type, obs=adata.obs, obs_niche=adata.obsm[OBSM_KEY_NICHE])
    ols = ols_fit(x_=dmat, y_=adata.X)
    params = ols.squeeze()
    adata.varm[VARM_KEY_PARAMS] = params
    term_condition = None
    term_type = None
    adata = test_differential_ncem(adata=adata, term_condition=term_condition, term_type=term_type)
    return adata


def differential_ncem_deconvoluted(adata: anndata.AnnData, formula: str, key_differential: str, key_deconvolution: str):
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
    _validate_formula(formula=formula, auto_keys=[key_differential, key_deconvolution])
    groups = np.sort(adata.obsm[key_deconvolution].columns).tolist()
    formula = extend_formula_differential_ncem(formula=formula, groups=groups, key_cond=key_differential)
    dmat = get_dmat_from_deconvoluted(deconv=adata.obsm[key_deconvolution], formula=formula, obs=adata.obs)
    ols = ols_fit(x_=dmat, y_=adata.X)
    params = ols.squeeze()
    adata.varm[VARM_KEY_PARAMS] = params
    term_condition = None
    term_type = None
    adata = test_differential_ncem(adata=adata, term_condition=term_condition, term_type=term_type)
    return adata


def linear_ncem(adata: anndata.AnnData, formula: str, key_type: str, key_graph: str):
    """
    Fit a linear NCEM based on an adata instance and save fits in instance.

    Saves fits and Wald test output into instance.
    TODO requires spatial graph to have been built or coordinates to be in fixed slots?

    Args:
        adata: AnnData instance with data and annotation.
        formula: Description of batch covariates as linear model. Do not include intercept, cell type, or niche as
            this is automatically added.
        key_type: Key of type annotation in .obs.
        key_graph: Key of spatial neighborhood graph in .obsp.

    Returns:

    """
    # TODO extract obs_niche into obsm, eg. using squidpy or using precomputed.
    _validate_formula(formula=formula, auto_keys=[])
    groups = np.sort(np.unique(adata.obs[key_type].values)).tolist()
    formula = extend_formula_ncem(formula=formula, groups=groups)
    adata.obsm[OBSM_KEY_NICHE] = get_obs_niche_from_graph(adata=adata, obs_key_type=key_type,
                                                          obsp_key_graph=key_graph, marginalisation="binary")
    dmat = get_dmat_from_obs(formula=formula, key_type=key_type, obs=adata.obs, obs_niche=adata.obsm[OBSM_KEY_NICHE])
    ols = ols_fit(x_=dmat, y_=adata.X)
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
    _validate_formula(formula=formula, auto_keys=[key_deconvolution])
    groups = np.sort(adata.obsm[key_deconvolution].columns).tolist()
    formula = extend_formula_ncem(formula=formula, groups=groups)
    dmat = get_dmat_from_deconvoluted(deconv=adata.obsm[key_deconvolution], formula=formula, obs=adata.obs)
    ols = ols_fit(x_=dmat, y_=adata.X)
    params = ols.squeeze()
    adata.varm[VARM_KEY_PARAMS] = params
    adata = test_linear_ncem(adata=adata, term_type=term_type)
    return adata
