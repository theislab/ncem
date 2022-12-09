from typing import List, Union

import anndata
import numpy as np
import pandas as pd
import patsy

from ncem.tl.fit.backend.linear_model import differential_ncem, differential_ncem_deconvoluted, linear_ncem, \
    linear_ncem_deconvoluted
from ncem.tl.fit.backend.testing import test_standard, test_deconvoluted
from ncem.tl.fit.backend.utils import read_uns, write_uns
from ncem.tl.fit.constants import PREFIX_INDEX, VARM_KEY_PARAMS, VARM_KEY_PVALS_SPLINE, VARM_KEY_FDR_PVALS_SPLINE, \
    UNS_KEY_CELL_TYPES, UNS_KEY_PER_INDEX, UNS_KEY_SPLINE_COEFS, UNS_KEY_SPLINE_DF, UNS_KEY_SPLINE_FAMILY, \
    UNS_KEY_SPLINE_KEY_1D


def get_spline_basis(df: int, key_1d_coord: str, obs: pd.DataFrame, spline_family: str):
    if spline_family.lower() == "bs":
        dmat_spline = patsy.dmatrix(
            "bs(" + key_1d_coord + ", df=" + str(df) + ", degree=3, include_intercept=False) - 1",
            obs
        )
    elif spline_family.lower() == "cr":
        dmat_spline = patsy.dmatrix(
            "cr(" + key_1d_coord + ", df=" + str(df) + ", constraints='center') - 1",
            obs
        )
    elif spline_family.lower() == "cc":
        dmat_spline = patsy.dmatrix(
            "cc(" + key_1d_coord + ", df=" + str(df) + ", constraints='center') - 1",
            obs
        )
    else:
        raise ValueError("spline basis %s not recognized" % spline_family)
    dmat_spline = pd.DataFrame(np.asarray(dmat_spline), index=obs.index, columns=dmat_spline.design_info.column_names)
    return dmat_spline


def _spline_ncem_base(adata: anndata.AnnData, f, df: int, key_1d_coord: str, spline_family: str,
                      type_specific_confounders: List[str] = [], **kwargs):
    """
    Add a spline fit to any NCEM model.

    Saves fits and Wald test output into instance.

    Args:
        adata: AnnData instance with data and annotation.
        f: NCEM fit function.
        df: Degrees of freedom of the spline model, i.e. the number of spline basis vectors.
        kwargs: Kwargs for NCEM function f.
        spline_family: The type of sline basis to use, refer also to:
            https://patsy.readthedocs.io/en/latest/spline-regression.html

                - "bs": B-splines
                - "cr": natural cubic splines
                - "cc": natural cyclic splines
        type_specific_confounders: List of confounding terms in .obs to be added with an interaction term to cell
            types, ie confounders that act on the cell type level. Global confounders can be added in the formula.

    Returns:

    """
    # Prepare spline basis:
    dmat_spline = get_spline_basis(df=df, key_1d_coord=key_1d_coord, obs=adata.obs, spline_family=spline_family)
    adata.obs = pd.concat([adata.obs, dmat_spline], axis=1)
    spline_coefs = dmat_spline.columns.tolist()
    # Save key spline hyper-parameters:
    write_uns(adata, k=UNS_KEY_SPLINE_COEFS, v=spline_coefs)
    write_uns(adata, k=UNS_KEY_SPLINE_DF, v=df)
    write_uns(adata, k=UNS_KEY_SPLINE_FAMILY, v=spline_family)
    write_uns(adata, k=UNS_KEY_SPLINE_KEY_1D, v=key_1d_coord)
    type_specific_confounders.extend(spline_coefs)
    # Fit NCEM with added spline terms:
    adata = f(adata=adata, type_specific_confounders=type_specific_confounders, **kwargs)
    # Perform differential expression along spline:
    # This is saved in addition to the differential expression from the NCEM.
    cell_types = read_uns(adata, k=UNS_KEY_CELL_TYPES)
    spline_coefs_by_type = {f"spline_{x}": [f"{PREFIX_INDEX}{x}:{y}" for y in spline_coefs]for x in cell_types}
    per_index_type = read_uns(adata, k=UNS_KEY_PER_INDEX)
    if per_index_type:
        adata = test_deconvoluted(adata=adata, cell_types=cell_types, coef_to_test=spline_coefs_by_type, key_coef=None,
                                  key_pval=VARM_KEY_PVALS_SPLINE, key_fdr_pval=VARM_KEY_FDR_PVALS_SPLINE)
    else:
        adata = test_standard(adata=adata, coef_to_test=spline_coefs_by_type, key_coef=None,
                              key_pval=VARM_KEY_PVALS_SPLINE, key_fdr_pval=VARM_KEY_FDR_PVALS_SPLINE)
    return adata


def spline_differential_ncem(adata: anndata.AnnData, df: int, key_1d_coord: str, key_differential: str, key_graph: str,
                             key_type: str, formula: str = "~0", spline_family: str = "cr",
                             type_specific_confounders: List[str] = []):
    """
    Fit a differential NCEM based on an adata instance and save fits in instance.

    Saves fits and Wald test output into instance.

    Args:
        adata: AnnData instance with data and annotation.
        df: Degrees of freedom of the spline model, i.e. the number of spline basis vectors.
        formula: Description of batch covariates as linear model. Do not include intercept, cell type, niche, or the
            differential term as this is automatically added.
        key_1d_coord: Key of numeric 1D coordinate of each observation in .obs. This will be used to build the spline.
        key_differential: Key of condition annotation in .obs. This will be used for testing.
        key_graph: Key of spatial neighborhood graph in .obsp.
        key_type: Key of type annotation in .obs.
        spline_family: The type of sline basis to use, refer also to:
            https://patsy.readthedocs.io/en/latest/spline-regression.html

                - "bs": B-splines
                - "cr": natural cubic splines
                - "cc": natural cyclic splines
        type_specific_confounders: List of confounding terms in .obs to be added with an interaction term to cell
            types, ie confounders that act on the cell type level. Global confounders can be added in the formula.

    Returns:

    """
    adata = _spline_ncem_base(adata=adata, df=df, f=differential_ncem, formula=formula, key_1d_coord=key_1d_coord,
                              key_differential=key_differential, key_graph=key_graph, key_type=key_type,
                              spline_family=spline_family, type_specific_confounders=type_specific_confounders)
    return adata


def spline_differential_ncem_deconvoluted(adata: anndata.AnnData, df: int, key_1d_coord: str, key_differential: str,
                                          key_deconvolution: str, formula: str = "~0", spline_family: str = "cr",
                                          type_specific_confounders: List[str] = []):
    """
    Fit a differential NCEM based on deconvoluted data in an adata instance and save fits in instance.

    Saves fits and Wald test output into instance.

    Args:
        adata: AnnData instance with data and annotation. Note on placement of deconvolution output:

                - type abundances must in be in .obsm[key_deconvolution] with cell type names as columns
                - spot- and type-specific gene expression results must be layers named after types
        df: Degrees of freedom of the spline model, i.e. the number of spline basis vectors.
        formula: Description of batch covariates as linear model. Do not include intercept, cell type, niche, or the
            differential term as this is automatically added.
        key_1d_coord: Key of numeric 1D coordinate of each observation in .obs. This will be used to build the spline.
        key_deconvolution: Key of type deconvolution in .obsm.
        key_differential: Key of condition annotation in .obs. This will be used for testing.
        spline_family: The type of sline basis to use, refer also to:
            https://patsy.readthedocs.io/en/latest/spline-regression.html

                - "bs": B-splines
                - "cr": natural cubic splines
                - "cc": natural cyclic splines
        type_specific_confounders: List of confounding terms in .obs to be added with an interaction term to cell
            types, ie confounders that act on the cell type level. As the formula is used for each index cell, this
            is equivalent to adding these terms into the formula.

    Returns:

    """
    adata = _spline_ncem_base(adata=adata, df=df, f=differential_ncem_deconvoluted, formula=formula,
                              key_1d_coord=key_1d_coord, key_deconvolution=key_deconvolution,
                              key_differential=key_differential, spline_family=spline_family,
                              type_specific_confounders=type_specific_confounders)
    return adata


def spline_linear_ncem(adata: anndata.AnnData, df: int, key_1d_coord: str, key_graph: str, key_type: str,
                       formula: str = "~0", spline_family: str = "cr", type_specific_confounders: List[str] = []):
    """
    Fit a linear NCEM based on an adata instance and save fits in instance.

    Saves fits and Wald test output into instance.

    Args:
        adata: AnnData instance with data and annotation.
        df: Degrees of freedom of the spline model, i.e. the number of spline basis vectors.
        formula: Description of batch covariates as linear model. Do not include intercept, cell type, or niche as
            this is automatically added.
        key_1d_coord: Key of numeric 1D coordinate of each observation in .obs. This will be used to build the spline.
        key_type: Key of type annotation in .obs.
        key_graph: Key of spatial neighborhood graph in .obsp.
        spline_family: The type of sline basis to use, refer also to:
            https://patsy.readthedocs.io/en/latest/spline-regression.html

                - "bs": B-splines
                - "cr": natural cubic splines
                - "cc": natural cyclic splines
        type_specific_confounders: List of confounding terms in .obs to be added with an interaction term to cell
            types, ie confounders that act on the cell type level. Global confounders can be added in the formula.

    Returns:

    """

    adata = _spline_ncem_base(adata=adata, df=df, f=linear_ncem, formula=formula, key_1d_coord=key_1d_coord,
                              key_graph=key_graph, key_type=key_type, spline_family=spline_family,
                              type_specific_confounders=type_specific_confounders)
    return adata


def spline_linear_ncem_deconvoluted(adata: anndata.AnnData, df: int, key_1d_coord: str, key_deconvolution: str,
                                    formula: str = "~0", spline_family: str = "cr",
                                    type_specific_confounders: List[str] = []):
    """
    Fit a linear NCEM based on deconvoluted data in an adata instance and save fits in instance.

    Saves fits and Wald test output into instance.

    Args:
        adata: AnnData instance with data and annotation. Note on placement of deconvolution output:

                - type abundances must in be in .obsm[key_deconvolution] with cell type names as columns
                - spot- and type-specific gene expression results must be layers named after types
        df: Degrees of freedom of the spline model, i.e. the number of spline basis vectors.
        formula: Description of batch covariates as linear model. Do not include intercept, cell type, or niche as
            this is automatically added.
        key_1d_coord: Key of numeric 1D coordinate of each observation in .obs. This will be used to build the spline.
        key_deconvolution: Key of type deconvolution in .obsm.
        spline_family: The type of sline basis to use, refer also to:
            https://patsy.readthedocs.io/en/latest/spline-regression.html

                - "bs": B-splines
                - "cr": natural cubic splines
                - "cc": natural cyclic splines
        type_specific_confounders: List of confounding terms in .obs to be added with an interaction term to cell
            types, ie confounders that act on the cell type level. As the formula is used for each index cell, this
            is equivalent to adding these terms into the formula.

    Returns:

    """
    adata = _spline_ncem_base(adata=adata, df=df, f=linear_ncem_deconvoluted, formula=formula,
                              key_1d_coord=key_1d_coord, key_deconvolution=key_deconvolution,
                              spline_family=spline_family, type_specific_confounders=type_specific_confounders)
    return adata


def get_spline_interpolation(adata: anndata.AnnData, genes: Union[str, List[str]], cell_type: str) -> pd.DataFrame:
    """
    # Spline fit only, including type-wise intercept but not confounding- or coupling-coefficients.

    Args:
        adata: AnnData instance after NCEM fit.
        genes: Genes to return fit for.
        cell_type: Cell type to return fit for.

    Returns: Data frame with spline fit at evaluation points.

    """
    if isinstance(genes, str):
        genes = [genes]
    assert isinstance(cell_type, str), "only supply one cell types as string"
    spline_coefs = read_uns(adata, k=UNS_KEY_SPLINE_COEFS)
    # Generated interpolated spline bases to replace cells (many and varying density!) as evaluation points.
    key_1d_coord = read_uns(adata, k=UNS_KEY_SPLINE_KEY_1D)
    df = read_uns(adata, k=UNS_KEY_SPLINE_DF)
    spline_family = read_uns(adata, k=UNS_KEY_SPLINE_FAMILY)
    obs_auxiliary = pd.DataFrame({
        key_1d_coord: np.linspace(
            np.min(adata.obs[key_1d_coord].values),
            np.max(adata.obs[key_1d_coord].values),
            100
        )
    })
    dmat_spline = get_spline_basis(df=df, key_1d_coord=key_1d_coord, obs=obs_auxiliary, spline_family=spline_family)
    dmat = np.hstack([
        np.ones([100, 1]),  # intercept
        dmat_spline,  # spline
    ])
    # We have one spline per cell type and iterate over cell types here:
    # Cell-type wise intercept is called directly after cell type.
    spline_coefs_type = [f"{PREFIX_INDEX}{cell_type}:{y}" for y in spline_coefs]
    coef_names = [f"{PREFIX_INDEX}{cell_type}"] + spline_coefs_type
    theta = adata.varm[VARM_KEY_PARAMS].loc[genes, coef_names].T
    yhat = np.matmul(dmat, theta)
    yhat = pd.DataFrame(yhat.T, index=obs_auxiliary[key_1d_coord].values, columns=genes)
    return yhat
