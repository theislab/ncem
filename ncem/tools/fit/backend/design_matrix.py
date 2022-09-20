from typing import Dict, List

import anndata
import numpy as np
import pandas as pd
import patsy

PREFIX_INDEX = "index_"
PREFIX_NEIGHBOR = "neighbor_"


def _make_type_categorical(obs, key_type):
    obs[key_type] = pd.Categorical(obs[key_type].values, categories=np.sort(np.unique(obs[key_type].values)))
    return obs


def extend_formula_ncem(formula: str, groups: List[str]):
    """
    Adds linear NCEM terms into formula.

    Example for cell types A, B, C:
        "~0+batch" -> "~0+batch+TYPE-A+...+TYPE-C+"
                      "index_TYPE-A:neighbor_TYPE-A+...+index_TYPE-C:neighbor_TYPE-C"
    """
    # Add type-wise intercept:
    formula = formula + "+" + "+".join([f"{PREFIX_INDEX}{x}" for x in groups])
    # Add couplings (type-type interactions):
    coef_couplings = [f"{PREFIX_INDEX}{x}:{PREFIX_NEIGHBOR}{y}" for y in groups for x in groups]
    formula = formula + "+" + "+".join(coef_couplings)
    return formula


def extend_formula_differential_ncem(formula: str, key_cond: str, groups: List[str]):
    """
    Adds linear NCEM terms into formula.

    Example for cell types A, B, C:
        "~0+batch" -> "~0+batch+TYPE-A+...+TYPE-C+"
                      "index_TYPE-A:neighbor_TYPE-A+...+index_TYPE-C:neighbor_TYPE-C+"
                      "condition+condition:TYPE-A+...+condition:TYPE-C+"
                      "condition:index_TYPE-A:neighbor_TYPE-A+...+condition:index_TYPE-C:neighbor_TYPE-C"
    """
    # Add type-wise intercept:
    formula = formula + "+" + "+".join([f"{PREFIX_INDEX}{x}" for x in groups])
    # Add couplings (type-type interactions):
    coef_couplings = [f"{PREFIX_INDEX}{x}:{PREFIX_NEIGHBOR}{y}" for y in groups for x in groups]
    formula = formula + "+" + "+".join(coef_couplings)
    # Add condition interaction to type-wise intercept:
    formula = formula + "+" + "+".join([f"{key_cond}:{PREFIX_INDEX}{x}" for x in groups])
    # Add differential couplings (differential type-type interactions):
    coef_diff_couplings = [f"{key_cond}:{PREFIX_INDEX}{x}:{PREFIX_NEIGHBOR}{y}" for y in groups for x in groups]
    formula = formula + "+" + "+".join(coef_diff_couplings)
    return formula


def get_obs_niche_from_graph(adata: anndata.AnnData, obs_key_type, obsp_key_graph: str,
                             marginalisation: str = "binary") -> pd.DataFrame:
    """
    Create niche sample annotation table from graph.

    Niche sample annotation indicates presence or abundance of cell types in neighborhood of each cell.

    Args:
        adata: AnnData with cell type annotation in .obs and spatial neighborhood graph in .obsp.
        marginalisation: Type of marginalisation across neighborhood for each type:

            - "binary": 0 or 1 for presence of absence of type in neighborhood.
            - "sum": number of cells of given type in neighborhood.
        obs_key_type: Key of cell type annotation in adata.obs.
        obsp_key_graph: Key of neighborhood graph in adata.obsp.

    Returns: Design matrix describing niche.
    """
    assert obs_key_type in adata.obs.keys()
    assert obsp_key_graph in adata.obsp.keys()

    # Auxiliary one-hot encoding of cell types:
    obs = _make_type_categorical(obs=adata.obs, key_type=obs_key_type)
    onehot_type = pd.get_dummies(obs[[obs_key_type]], columns=[obs_key_type], drop_first=False)
    # Make sure diagonal of graph is zero:
    g = adata.obsp[obsp_key_graph].copy()
    g[np.arange(0, g.shape[0]), np.arange(0, g.shape[1])] = 0.
    # Cell type counts in each neighborhood:
    counts = g.dot(onehot_type.values)
    if marginalisation == "binary":
        obs_niche = np.asarray(np.asarray(counts > 0, dtype="int32"), dtype="float32")
    elif marginalisation == "sum":
        obs_niche = counts
    else:
        raise ValueError(marginalisation)
    obs_niche = pd.DataFrame(obs_niche, index=adata.obs_names, columns=obs[obs_key_type].values.categories)
    return obs_niche


def get_dmat_from_obs(obs: pd.DataFrame, obs_niche: pd.DataFrame, formula: str, key_type: str) -> pd.DataFrame:
    """
    Create a design matrix from a sample description table according to a patsy style formula.

    Example for cell types A, B, C:
        columns([batch, type]) -> columns([batch, type, index_A, index_B, index_C, neighbor_A, neighbor_B, neighbor_C])

    Args:
        obs: Observation-indexed table.
        formula: Model formula.
        key_type: Key of cell type annotation in .obs.
        obs_niche: Observation-indexed niche table with summary of occurrence of cell types in niche of each cell
            (observation x types).

    Returns: Design matrix.
    """
    obs = _make_type_categorical(obs=obs, key_type=key_type)
    assert np.all(obs.index == obs_niche.index)
    assert np.all([x in obs[key_type].values.categories for x in obs_niche.columns])
    # One-hot encode index cell:
    obs_index_type = pd.get_dummies(obs[[key_type]], prefix=PREFIX_INDEX, prefix_sep='', columns=[key_type],
                                    drop_first=False)
    # Process niche table:
    obs_niche.columns = [PREFIX_NEIGHBOR + x for x in obs_niche.columns]
    # Merge sample annotation:
    obs_full = pd.concat([obs, obs_index_type, obs_niche], axis=1)
    dmat = patsy.dmatrix(formula, obs_full)
    dmat = pd.DataFrame(np.asarray(dmat), index=obs.index, columns=dmat.design_info.column_names)
    return dmat


def get_dmats_from_deconvoluted(obs: pd.DataFrame, deconv: pd.DataFrame, formula: str) -> Dict[str, pd.DataFrame]:
    """
    Create a design matrix per index cell from a sample description table and deconvolution results according to a
    patsy style formula.


    Example for cell types A, B, C:
        columns([batch]) + deconv -> columns([batch, type, index_A, index_B, index_C, neighbor_A, neighbor_B,
                                              neighbor_C])

    Args:
        obs: Observation-indexed table. Note: does not need spot annotation as obsm carries deconvolution table.
        deconv: Deconvolution result table. Indexed by spots and cell types. E.g. a value from adata.obsm.
        formula: Model formula.

    Returns: Design matrix.
    """
    assert obs.shape[0] == deconv.shape[0]
    assert np.all(obs.index == deconv.index)
    # 1. Create spot x cell-type-wise design matrix.
    type_index_key = "type"
    cell_types = deconv.columns
    # Abundance of cell types in spot (niche):
    obs_niche = {
        x: pd.DataFrame(deconv.values, columns=[PREFIX_NEIGHBOR + x for x in deconv.columns])
        for x in cell_types
    }
    # One-hot encode index cell:
    dummy_type_annotation = pd.DataFrame({type_index_key: deconv.columns})
    dummy_type_annotation = _make_type_categorical(obs=dummy_type_annotation, key_type=type_index_key)
    obs_index_type = pd.get_dummies(dummy_type_annotation, prefix=PREFIX_INDEX, prefix_sep='',
                                    columns=[type_index_key], drop_first=False)
    obs_index_type = {x: pd.concat([obs_index_type.iloc[[i], :] for _ in range(obs.shape[0])], axis=0)
                      for i, x in enumerate(cell_types)}
    # Remaining spot annotation:
    obs_unsqueezed = {x: obs for x in cell_types}
    # 2. Get design matrix
    dmats = {}
    for x in cell_types:
        # Merge sample annotation:
        obs_unsqueezed[x].index = obs.index
        obs_index_type[x].index = obs.index
        obs_niche[x].index = obs.index
        obs_full = pd.concat([obs_unsqueezed[x], obs_index_type[x], obs_niche[x]], axis=1)
        dmats[x] = patsy.dmatrix(formula, obs_full)
        dmats[x] = pd.DataFrame(np.asarray(dmats[x]), index=obs.index, columns=dmats[x].design_info.column_names)
    return dmats
