from typing import Dict, List, Union

import anndata
import numpy as np
import pandas as pd
from diffxpy.stats.stats import wald_test, wald_test_chisq
from diffxpy.testing.correction import correct

from ncem.tl.fit.constants import OBSM_KEY_DMAT, VARM_KEY_PARAMS
from ncem.utils.wald_test import get_fim_inv


def test_standard(
    adata: anndata.AnnData,
    coef_to_test: Union[Dict[str, List[str]], List[str]],
    key_coef: str,
    key_pval: str,
    key_fdr_pval: str,
) -> anndata.AnnData:
    """
        Test for model with individual spatially localised entities, e.g. cell-resolution.

    Args:
        adata: AnnData instance with fits saved.
        coef_to_test: Names of coefficients to test, or named groups of coefficients for multi-parameter tests (dict).
        key_coef: .varm key to write tested coefficient fits into.
        key_pval: .varm key to write p-values into.
        key_fdr_pval: .varm key to write FDR-corrected p-values into.

    Returns:
        Anndata instance with test output saved. Test output is one p-value, FDR-corrected p-value and log-fold change
        per gene and type x type pair. The test signifies the coupling between any two cell types.

    """
    dmat = adata.obsm[OBSM_KEY_DMAT]
    params = adata.varm[VARM_KEY_PARAMS]
    fisher_inv = get_fim_inv(x=dmat, y=adata.X)
    # Run multi-parameter Wald test:
    parameter_names = params.columns.tolist()
    pvals = {}
    tested_coefficients = {}
    multi_parameter_tests = isinstance(coef_to_test, dict)
    if multi_parameter_tests:
        test_keys = list(coef_to_test.keys())
        for k, x in coef_to_test.items():
            idx = [parameter_names.index(y) for y in x]
            theta_mle = params.values[:, idx]
            theta_covar = fisher_inv[:, idx, :][:, :, idx]
            pvals[k] = np.expand_dims(wald_test_chisq(theta_mle=theta_mle.T, theta_covar=theta_covar), axis=0)
            if len(idx) == 1:
                tested_coefficients[k] = np.expand_dims(theta_mle[:, 0], axis=0)
            else:
                tested_coefficients[k] = np.zeros_like(theta_mle[:, 0]) + np.nan
    else:
        test_keys = coef_to_test
        for x in coef_to_test:
            idx = parameter_names.index(x)
            theta_mle = params.values[:, idx]
            theta_sd = fisher_inv[:, idx, idx]
            theta_sd = np.nextafter(0, np.inf, out=theta_sd, where=theta_sd < np.nextafter(0, np.inf))
            theta_sd = np.sqrt(theta_sd)
            theta_sd = np.expand_dims(theta_sd, axis=0)
            theta_mle = np.expand_dims(theta_mle, axis=0)
            pvals[x] = wald_test(theta_mle=theta_mle, theta_sd=theta_sd, theta0=0)
            tested_coefficients[x] = theta_mle
    # Run FDR correction:
    pvals_flat = np.hstack(list(pvals.values()))
    qvals_flat = correct(pvals_flat.flatten())
    qvals = qvals_flat.reshape((-1, len(test_keys)))
    # Write results to object:
    if key_coef is not None:
        tested_coefficients = np.concatenate(list(tested_coefficients.values()), axis=0).T
        # print(tested_coefficients.shape)
        tested_coefficients = pd.DataFrame(tested_coefficients, index=adata.var_names)
        # print(tested_coefficients)
        adata.varm[key_coef] = tested_coefficients
    pvals = np.concatenate(list(pvals.values())).T
    adata.varm[key_pval] = pd.DataFrame(pvals, index=adata.var_names, columns=test_keys)
    adata.varm[key_fdr_pval] = pd.DataFrame(qvals, index=adata.var_names, columns=test_keys)
    return adata


def test_deconvoluted(
    adata: anndata.AnnData,
    coef_to_test: Union[Dict[str, List[str]], List[str]],
    cell_types: List[str],
    key_coef: str,
    key_pval: str,
    key_fdr_pval: str,
) -> anndata.AnnData:
    """
    Test for model of deconvoluted spots.

    Args:
        adata: AnnData instance with fits saved.
        coef_to_test: Names of coefficients to test, or named groups of coefficients for multi-parameter tests (dict).
        cell_types: Cell types that were deconvoluted to.
        key_coef: .varm key to write tested coefficient fits into.
        key_pval: .varm key to write p-values into.
        key_fdr_pval: .varm key to write FDR-corrected p-values into.

    Returns:
        Anndata instance with test output saved. Test output is one p-value, FDR-corrected p-value and log-fold change
        per gene and type x type pair. The test signifies the coupling between any two cell types.

    """
    # Run multi-parameter Wald test for each individually fit model (note that one linear model was fit for each index
    # cell):
    pvals = {}
    qvals = {}
    tested_coefficients = {}
    multi_parameter_tests = isinstance(coef_to_test, dict)
    if multi_parameter_tests:
        test_keys = list(coef_to_test.keys())
    else:
        test_keys = coef_to_test
    # Loop over models (cell types) and coefficients, each coefficient will appear in only one model.
    # This is checked by an assert statement in the inner loop.
    for x in cell_types:
        dmat_key = f"{OBSM_KEY_DMAT}_{x}"
        dmat = adata.obsm[dmat_key]
        # Subset parameter matrix to parameters of sub-model at hand (dmat).
        params = adata.varm[VARM_KEY_PARAMS].loc[:, dmat.columns]
        parameter_names = params.columns.tolist()
        fisher_inv = get_fim_inv(x=dmat, y=adata.layers[x])
        if multi_parameter_tests:
            for k, y in coef_to_test.items():
                if np.all([z in parameter_names for z in y]):
                    idx = np.sort([parameter_names.index(z) for z in y])
                    theta_mle = params.values[:, idx]
                    fisher_inv_subset = fisher_inv[:, idx, :][:, :, idx]
                    assert k not in pvals.keys()
                    pvals[k] = wald_test_chisq(theta_mle=theta_mle.T, theta_covar=fisher_inv_subset)
                    if len(idx) == 1:
                        tested_coefficients[k] = theta_mle[:, 0]
                    else:
                        tested_coefficients[k] = np.zeros_like(theta_mle[:, 0]) + np.nan
        else:
            for y in coef_to_test:
                if y in parameter_names:
                    idx = parameter_names.index(y)
                    theta_mle = params.values[:, idx]
                    theta_sd = fisher_inv[:, idx, idx]
                    theta_sd = np.nextafter(0, np.inf, out=theta_sd, where=theta_sd < np.nextafter(0, np.inf))
                    theta_sd = np.sqrt(theta_sd)
                    assert y not in pvals.keys()
                    pvals[y] = wald_test(theta_mle=theta_mle, theta_sd=theta_sd)
                    qvals[y] = correct(pvals[y])
                    tested_coefficients[y] = theta_mle
    # Run FDR correction across all models:
    pvals_flat = np.hstack(list(pvals.values()))
    qvals_flat = correct(pvals_flat)
    # qvals = qvals_flat.reshape((-1, len(test_keys)))
    # Write results to object:
    if key_coef is not None:
        adata.varm[key_coef] = pd.DataFrame(tested_coefficients, index=adata.var_names)
    adata.varm[key_pval] = pd.DataFrame(pvals, index=adata.var_names)
    adata.varm[key_fdr_pval] = pd.DataFrame(qvals, index=adata.var_names, columns=test_keys)
    return adata
