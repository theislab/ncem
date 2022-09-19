import numpy as np
import pandas as pd
import patsy

PREFIX_INDEX = "index_"
PREFIX_NEIGHBOR = "neighbor_"


def _make_type_categorical(obs, key_type):
    obs[key_type] = pd.Categorical(obs[key_type].values, categories=np.sort(np.unique(obs[key_type].values)))
    return obs


def extend_formula_ncem(formula: str, key_type: str, obs: pd.DataFrame):
    """Adds linear NCEM terms into formula."""
    groups = np.unique(obs[key_type].values).tolist()
    # Add type-wise intercept:
    formula = formula + "+" + "+".join(groups)
    # Add couplings (type-type interactions):
    coef_couplings = [f"{x}:neighbor_{y}" for y in groups for x in groups]
    formula = formula + "+" + "+".join(coef_couplings)
    return formula


def extend_formula_differential_ncem(formula: str, key_cond: str, key_type: str, obs: pd.DataFrame):
    """Adds linear NCEM terms into formula."""
    conds = np.unique(obs[key_cond].values).tolist()
    groups = np.unique(obs[key_type].values).tolist()
    # Add type-wise intercept:
    formula = formula + "+" + "+".join(groups)
    # Add couplings (type-type interactions):
    coef_couplings = [f"{x}:{y}" for y in groups for x in groups]
    formula = formula + "+" + "+".join(coef_couplings)
    # Add differential couplings (differential type-type interactions):
    coef_couplings = [f"{x}:{y}:neighbor_{z}" for z in groups for y in groups for x in conds]
    formula = formula + "+" + "+".join(coef_couplings)
    return formula


def get_dmat_from_obs(obs: pd.DataFrame, obs_niche: pd.DataFrame, formula: str, key_type: str) -> pd.DataFrame:
    """
    Create a design matrix from a sample description table according to a patsy style formula.

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
    assert np.all([x in obs[key_type].categories for x in obs_niche.columns])
    # One-hot encode index cell:
    obs_index_type = pd.get_dummies(obs, prefix=PREFIX_INDEX, prefix_sep='', columns=key_type, drop_first=False)
    # Process niche table:
    obs_niche.columns = [PREFIX_NEIGHBOR + x for x in obs_niche.columns]
    # Merge sample annotation:
    obs_full = pd.concat([obs, obs_index_type, obs_niche], axis=1)
    dmat = patsy.dmatrix(formula, obs_full)
    return dmat


def get_dmat_from_deconvoluted(obs: pd.DataFrame, deconv: pd.DataFrame, formula: str) -> pd.DataFrame:
    """
    Create a design matrix from a sample description table and deconvolution results according to a patsy style formula.

    Args:
        obs: Observation-indexed table. Note: does not need spot annotation as obsm carries deconvolution table.
        deconv: Deconvolution result table. Indexed by spots and cell types. E.g. a value from adata.obsm.
        formula: Model formula.

    Returns: Design matrix.
    """
    # 1. Create spot x cell-type-wise design matrix.
    type_index_key = "type"
    obs_niche = pd.concat([
        pd.concat([pd.DataFrame(deconv.iloc[[i], :].values, columns=[PREFIX_NEIGHBOR + x])
                   for x in deconv.columns], axis=0)
        for i in range(deconv.shape[0])], axis=0)
    # One-hot encode index cell:
    dummy_type_annotation = pd.DataFrame({type_index_key: deconv.columns})
    dummy_type_annotation = _make_type_categorical(obs=dummy_type_annotation, key_type=type_index_key)
    obs_index_type = pd.concat([
        pd.concat([pd.get_dummies(dummy_type_annotation.iloc[[i], :], prefix=PREFIX_INDEX, prefix_sep='',
                                  columns=type_index_key, drop_first=False)
                   for _ in range(deconv.shape[1])], axis=0)
        for i in range(deconv.shape[0])], axis=0)
    obs_unsqueezed = pd.concat([pd.concat([obs.iloc[i, :] for _ in range(deconv.shape[1])], axis=0)
                                for i in range(deconv.shape[0])], axis=0)
    # Merge sample annotation:
    obs_full = pd.concat([obs_unsqueezed, obs_index_type, obs_niche], axis=1)
    # 2. Get design matrix
    dmat = patsy.dmatrix(formula, obs_full)
    return dmat
