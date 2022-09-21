from typing import List

import anndata
import numpy as np
import pandas as pd

from ncem.tools.fit.backend.design_matrix import extend_formula_ncem, extend_formula_differential_ncem, \
    get_binary_sample_annotation_conditions, get_dmats_from_deconvoluted, get_dmat_from_obs, get_obs_niche_from_graph
from ncem.tools.fit.backend.ols_fit import ols_fit
from ncem.tools.fit.backend.testing import test_ncem, test_ncem_deconvoluted
from ncem.tools.fit.constants import VARM_KEY_PARAMS, OBSM_KEY_DMAT, OBSM_KEY_NICHE


def _validate_formula(formula: str, auto_keys: List[str] = []):
    # Check formula format:
    assert formula.startswith("~0"), "base formula needs to start with '~0'"


def differential_ncem(adata: anndata.AnnData, formula: str, key_differential: str, key_graph: str, key_type: str):
    """
    Fit a differential NCEM based on an adata instance and save fits in instance.

    Saves fits and Wald test output into instance.

    Args:
        adata: AnnData instance with data and annotation.
        formula: Description of batch covariates as linear model. Do not include intercept, cell type, niche, or the
            differential term as this is automatically added.
        key_differential: Key of condition annotation in .obs. This will be used for testing.
        key_graph: Key of spatial neighborhood graph in .obsp.
        key_type: Key of type annotation in .obs.

    Returns:

    """
    _validate_formula(formula=formula, auto_keys=[key_differential])
    cell_types = np.sort(np.unique(adata.obs[key_type].values)).tolist()
    # Simulate intercept in this auxiliary design matrix so that first condition is absorbed into intercept.
    obs_condition = get_binary_sample_annotation_conditions(obs=adata.obs, formula=f"~1+{key_differential}")
    conditions = obs_condition.columns
    # Add one-hot encoded condition assignments into sample description so that they are available as terms for
    # formula.
    obs = pd.concat([adata.obs, obs_condition], axis=1)
    formula, coef_to_test = extend_formula_differential_ncem(formula=formula, cell_types=cell_types,
                                                             conditions=conditions, per_index_cell=False)
    adata.obsm[OBSM_KEY_NICHE] = get_obs_niche_from_graph(adata=adata, obs_key_type=key_type,
                                                          obsp_key_graph=key_graph, marginalisation="binary")
    adata.obsm[OBSM_KEY_DMAT] = get_dmat_from_obs(formula=formula, key_type=key_type, obs=obs,
                                                  obs_niche=adata.obsm[OBSM_KEY_NICHE])
    params = ols_fit(x_=adata.obsm[OBSM_KEY_DMAT].values, y_=adata.X)
    params = pd.DataFrame(params.squeeze(), index=adata.var_names, columns=adata.obsm[OBSM_KEY_DMAT].columns)
    adata.varm[VARM_KEY_PARAMS] = params
    adata = test_ncem(adata=adata, coef_to_test=coef_to_test)
    return adata


def differential_ncem_deconvoluted(adata: anndata.AnnData, formulas: str, key_differential: str, key_deconvolution: str):
    """
    Fit a differential NCEM based on deconvoluted data in an adata instance and save fits in instance.

    Saves fits and Wald test output into instance.

    Args:
        adata: AnnData instance with data and annotation. Note on placement of deconvolution output:

                - type abundances must in be in .obsm[key_deconvolution] with cell type names as columns
                - spot- and type-specific gene expression results must be layers named after types
        formulas: Description of batch covariates as linear model. Do not include intercept, cell type, niche, or the
            differential term as this is automatically added.
        key_deconvolution: Key of type deconvolution in .obsm.
        key_differential: Key of condition annotation in .obs. This will be used for testing.

    Returns:

    """
    _validate_formula(formula=formulas, auto_keys=[key_differential, key_deconvolution])
    cell_types = np.sort(adata.obsm[key_deconvolution].columns).tolist()
    # Simulate intercept in this auxiliary design matrix so that first condition is absorbed into intercept.
    obs_condition = get_binary_sample_annotation_conditions(obs=adata.obs, formula=f"~1+{key_differential}")
    conditions = obs_condition.columns
    # Add one-hot encoded condition assignments into sample description so that they are available as terms for
    # formula.
    obs = pd.concat([adata.obs, obs_condition], axis=1)
    formulas, coef_to_test = extend_formula_differential_ncem(formula=formulas, cell_types=cell_types,
                                                              conditions=conditions, per_index_cell=True)
    dmats = get_dmats_from_deconvoluted(deconv=adata.obsm[key_deconvolution], formulas=formulas, obs=obs)
    for k, v in dmats.items():
        dmat_key = f"{OBSM_KEY_DMAT}_{k}"
        adata.obsm[dmat_key] = v
        params = ols_fit(x_=adata.obsm[dmat_key].values, y_=adata.layers[k])
        params = pd.DataFrame(params.squeeze(), index=adata.var_names, columns=adata.obsm[dmat_key].columns)
        if VARM_KEY_PARAMS in adata.varm.keys():
            adata.varm[VARM_KEY_PARAMS] = pd.concat([adata.varm[VARM_KEY_PARAMS], params], axis=1)
        else:
            adata.varm[VARM_KEY_PARAMS] = params
    adata = test_ncem_deconvoluted(adata=adata, coef_to_test=coef_to_test, cell_types=cell_types)
    return adata


def linear_ncem(adata: anndata.AnnData, formula: str, key_type: str, key_graph: str):
    """
    Fit a linear NCEM based on an adata instance and save fits in instance.

    Saves fits and Wald test output into instance.

    Args:
        adata: AnnData instance with data and annotation.
        formula: Description of batch covariates as linear model. Do not include intercept, cell type, or niche as
            this is automatically added.
        key_type: Key of type annotation in .obs.
        key_graph: Key of spatial neighborhood graph in .obsp.

    Returns:

    """
    _validate_formula(formula=formula, auto_keys=[])
    cell_types = np.sort(np.unique(adata.obs[key_type].values)).tolist()
    formula, coef_to_test = extend_formula_ncem(formula=formula, cell_types=cell_types, per_index_cell=False)
    adata.obsm[OBSM_KEY_NICHE] = get_obs_niche_from_graph(adata=adata, obs_key_type=key_type,
                                                          obsp_key_graph=key_graph, marginalisation="binary")
    adata.obsm[OBSM_KEY_DMAT] = get_dmat_from_obs(formula=formula, key_type=key_type, obs=adata.obs,
                                                  obs_niche=adata.obsm[OBSM_KEY_NICHE])
    params = ols_fit(x_=adata.obsm[OBSM_KEY_DMAT].values, y_=adata.X)
    params = pd.DataFrame(params.squeeze(), index=adata.var_names, columns=adata.obsm[OBSM_KEY_DMAT].columns)
    adata.varm[VARM_KEY_PARAMS] = params
    adata = test_ncem(adata=adata, coef_to_test=coef_to_test)
    return adata


def linear_ncem_deconvoluted(adata: anndata.AnnData, formulas: str, key_deconvolution: str):
    """
    Fit a linear NCEM based on deconvoluted data in an adata instance and save fits in instance.

    Saves fits and Wald test output into instance.

    Args:
        adata: AnnData instance with data and annotation. Note on placement of deconvolution output:

                - type abundances must in be in .obsm[key_deconvolution] with cell type names as columns
                - spot- and type-specific gene expression results must be layers named after types
        formulas: Description of batch covariates as linear model. Do not include intercept, cell type, or niche as
            this is automatically added.
        key_deconvolution: Key of type deconvolution in .obsm.

    Returns:

    """
    _validate_formula(formula=formulas, auto_keys=[key_deconvolution])
    cell_types = np.sort(adata.obsm[key_deconvolution].columns).tolist()
    assert np.all([x in adata.layers.keys() for x in cell_types])
    formulas, coef_to_test = extend_formula_ncem(formula=formulas, cell_types=cell_types, per_index_cell=True)
    dmats = get_dmats_from_deconvoluted(deconv=adata.obsm[key_deconvolution], formulas=formulas, obs=adata.obs)
    for k, v in dmats.items():
        dmat_key = f"{OBSM_KEY_DMAT}_{k}"
        adata.obsm[dmat_key] = v
        params = ols_fit(x_=adata.obsm[dmat_key].values, y_=adata.layers[k])
        params = pd.DataFrame(params.squeeze(), index=adata.var_names, columns=adata.obsm[dmat_key].columns)
        if VARM_KEY_PARAMS in adata.varm.keys():
            adata.varm[VARM_KEY_PARAMS] = pd.concat([adata.varm[VARM_KEY_PARAMS], params], axis=1)
        else:
            adata.varm[VARM_KEY_PARAMS] = params
    adata = test_ncem_deconvoluted(adata=adata, coef_to_test=coef_to_test, cell_types=cell_types)
    return adata
