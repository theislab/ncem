from typing import Dict, List, Tuple

import anndata
import numpy as np
import pandas as pd
import patsy

from ncem.tl.fit.constants import PREFIX_INDEX, PREFIX_NEIGHBOR


def _make_type_categorical(obs, key_type):
    obs[key_type] = pd.Categorical(obs[key_type].values, categories=np.sort(np.unique(obs[key_type].values)))
    return obs


def extend_formula_ncem(
    formula: str, cell_types: List[str], per_index_type: bool = False, type_specific_confounders: List[str] = []
) -> Tuple[str, List[str]]:
    """
    Adds linear NCEM terms into formula.

    Example for cell types A, B, C:
        "~0+batch" -> "~0+batch+TYPE-A+...+TYPE-C+"
                      "index_TYPE-A:neighbor_TYPE-A+...+index_TYPE-C:neighbor_TYPE-C"

    Args:
        formula: Base formula, may describe confounding for example.
        cell_types: Cell type labels.
        per_index_type: Whether to yield formula per index cell type, ie if one separate linear model is fit for each
            index cell type.
        type_specific_confounders: List of confounding terms in obs to be added with an interaction term to cell
            types, ie confounders that act on the cell type level. Global confounders can be added in the formula.

    Returns:

        - Full NCEM formula, or dictionary over index cell-wise formulas
        - List of coefficient names to test.
    """
    if per_index_type:
        formula_out = {}
        coef_couplings = []
        for x in cell_types:
            formula_x = formula
            # Add type-specific confounders:
            if len(type_specific_confounders) > 0:
                formula_x = formula_x + "+" + "+".join([f"{PREFIX_INDEX}{x}:{y}" for y in type_specific_confounders])
            # Add type-wise intercept:
            formula_x = formula_x + "+" + f"{PREFIX_INDEX}{x}"
            # Add couplings (type-type interactions):
            coef_couplings_x = [f"{PREFIX_INDEX}{x}:{PREFIX_NEIGHBOR}{y}" for y in cell_types]
            formula_out[x] = formula_x + "+" + "+".join(coef_couplings_x)
            coef_couplings.extend(coef_couplings_x)
    else:
        # Add type-specific confounders:
        if len(type_specific_confounders) > 0:
            formula = (
                formula
                + "+"
                + "+".join([f"{PREFIX_INDEX}{x}:{y}" for y in type_specific_confounders for x in cell_types])
            )
        # Add type-wise intercept:
        formula = formula + "+" + "+".join([f"{PREFIX_INDEX}{x}" for x in cell_types])
        # Add couplings (type-type interactions):
        coef_couplings = [f"{PREFIX_INDEX}{x}:{PREFIX_NEIGHBOR}{y}" for y in cell_types for x in cell_types]
        formula_out = formula + "+" + "+".join(coef_couplings)
    return formula_out, coef_couplings


def extend_formula_differential_ncem(
    formula: str,
    conditions: List[str],
    cell_types: List[str],
    per_index_type: bool = False,
    type_specific_confounders: List[str] = [],
) -> Tuple[str, List[str], Dict[str, List[str]]]:
    """
    Adds linear NCEM terms into formula.

    Example for cell types A, B, C:
        "~0+batch" -> "~0+batch+TYPE-A+...+TYPE-C+"
                      "index_TYPE-A:neighbor_TYPE-A+...+index_TYPE-C:neighbor_TYPE-C+"
                      "condition_1+TYPE-A:condition_1+...+TYPE-C:condition_1+"
                      "index_TYPE-A:neighbor_TYPE-A:condition_1+...+index_TYPE-C:neighbor_TYPE-C:condition_1"

    Args:
        formula: Base formula, may describe confounding for example.
        cell_types: Cell type labels.
        conditions: List of condition names with first one dropped (ie the one that is absorbed into intercept).
        per_index_type: Whether to yield formula per index cell type, ie if one separate linear model is fit for each
            index cell type.
        type_specific_confounders: List of confounding terms in obs to be added with an interaction term to cell
            types, ie confounders that act on the cell type level. Global confounders can be added in the formula.

    Returns:

        - Full NCEM formula, or dictionary over index cell-wise formulas
        - Coefficients to test for NCEM: List of coefficient names to test.
        - Coefficients to test for differential NCEM: Dictionary over coefficient names to test grouped by index-target
            cell type pair. Each value represents all interaction coefficients of that pair to all modelled conditions.
    """
    coef_diff_couplings_grouped = {}
    if per_index_type:
        coef_couplings = []
        formula_out = {}
        for x in cell_types:
            formula_x = formula
            # Add type-specific confounders:
            if len(type_specific_confounders) > 0:
                formula_x = formula_x + "+" + "+".join([f"{PREFIX_INDEX}{x}:{y}" for y in type_specific_confounders])
            # Add type-wise intercept:
            formula_x = formula_x + "+" + f"{PREFIX_INDEX}{x}"
            # Add couplings (type-type interactions):
            coef_couplings_x = [f"{PREFIX_INDEX}{x}:{PREFIX_NEIGHBOR}{y}" for y in cell_types]
            coef_couplings.extend(coef_couplings_x)
            formula_x = formula_x + "+" + "+".join(coef_couplings_x)
            for c in conditions:
                # Add condition interaction to type-wise intercept:
                formula_x = formula_x + "+" + f"{PREFIX_INDEX}{x}:{c}"
                # Add differential couplings (differential type-type interactions):
                coef_diff_couplings_x = [f"{PREFIX_INDEX}{x}:{PREFIX_NEIGHBOR}{y}:{c}" for y in cell_types]
                formula_x = formula_x + "+" + "+".join(coef_diff_couplings_x)
                # Group coefficients across conditions by interaction pair:
                for y in cell_types:
                    pair_name = f"{PREFIX_INDEX}{x}:{PREFIX_NEIGHBOR}{y}"
                    if pair_name not in coef_diff_couplings_grouped:
                        coef_diff_couplings_grouped[pair_name] = []
                    coef_diff_couplings_grouped[pair_name].append(f"{PREFIX_INDEX}{x}:{PREFIX_NEIGHBOR}{y}:{c}")
                formula_out[x] = formula_x
    else:
        # Add type-specific confounders:
        if len(type_specific_confounders) > 0:
            formula = (
                formula
                + "+"
                + "+".join([f"{PREFIX_INDEX}{x}:{y}" for y in type_specific_confounders for x in cell_types])
            )
        # Add type-wise intercept:
        formula = formula + "+" + "+".join([f"{PREFIX_INDEX}{x}" for x in cell_types])
        # Add couplings (type-type interactions):
        coef_couplings = [f"{PREFIX_INDEX}{x}:{PREFIX_NEIGHBOR}{y}" for y in cell_types for x in cell_types]
        formula = formula + "+" + "+".join(coef_couplings)
        for c in conditions:
            # Add condition interaction to type-wise intercept:
            formula = formula + "+" + "+".join([f"{PREFIX_INDEX}{x}:{c}" for x in cell_types])
            # Add differential couplings (differential type-type interactions):
            coef_diff_couplings = [
                f"{PREFIX_INDEX}{x}:{PREFIX_NEIGHBOR}{y}:{c}" for y in cell_types for x in cell_types
            ]
            formula = formula + "+" + "+".join(coef_diff_couplings)
            # Group coefficients across conditions by interaction pair:
            for x in cell_types:
                for y in cell_types:
                    pair_name = f"{PREFIX_INDEX}{x}:{PREFIX_NEIGHBOR}{y}"
                    if pair_name not in coef_diff_couplings_grouped:
                        coef_diff_couplings_grouped[pair_name] = []
                    coef_diff_couplings_grouped[pair_name].append(f"{PREFIX_INDEX}{x}:{PREFIX_NEIGHBOR}{y}:{c}")
        formula_out = formula
    return formula_out, coef_couplings, coef_diff_couplings_grouped


def get_obs_niche_from_graph(
    adata: anndata.AnnData, obs_key_type, obsp_key_graph: str, marginalisation: str = "binary"
) -> pd.DataFrame:
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
    g[np.arange(0, g.shape[0]), np.arange(0, g.shape[1])] = 0.0
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


def get_binary_sample_annotation_conditions(obs: pd.DataFrame, formula: str) -> pd.DataFrame:
    """
    Create a design matrix from a sample description table according to a patsy style formula.


    Note that we are switching from addressing "condition" as a multi-factor categorical to multiple binary
    terms here so that coefficient names are easier to control in formula assembly and testing.

    Args:
        obs: Observation-indexed table.
        formula: Model formula.

    Returns: Design matrix.
    """
    dmat = patsy.dmatrix(formula, obs)
    # Simplify condition names, this is necessary for patsy to accept these as terms later.
    # Leave out intercept:
    conditions = [x.split("[T.")[-1].replace("]", "") for x in dmat.design_info.column_names[1:]]
    dmat = pd.DataFrame(np.asarray(dmat)[:, 1:], index=obs.index, columns=conditions)
    return dmat


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
    obs_index_type = pd.get_dummies(
        obs[[key_type]], prefix=PREFIX_INDEX, prefix_sep="", columns=[key_type], drop_first=False
    )
    # Process niche table:
    obs_niche.columns = [PREFIX_NEIGHBOR + x for x in obs_niche.columns]
    # Merge sample annotation:
    obs_full = pd.concat([obs, obs_index_type, obs_niche], axis=1)
    dmat = patsy.dmatrix(formula, obs_full)
    dmat = pd.DataFrame(np.asarray(dmat), index=obs.index, columns=dmat.design_info.column_names)
    return dmat


def get_dmats_from_deconvoluted(
    obs: pd.DataFrame, deconv: pd.DataFrame, formulas: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Create a design matrix per index cell from a sample description table and deconvolution results according to a
    patsy style formula.


    Example for cell types A, B, C:
        columns([batch]) + deconv -> columns([batch, type, index_A, index_B, index_C, neighbor_A, neighbor_B,
                                              neighbor_C])

    See also get_dmat_global_from_deconvoluted(), coefficient names match between both functions.

    Args:
        obs: Observation-indexed table. Note: does not need spot annotation as obsm carries deconvolution table.
        deconv: Deconvolution result table. Indexed by spots and cell types. E.g. a value from adata.obsm.
        formulas: Model formula per index cell.

    Returns: Design matrix.
    """
    assert obs.shape[0] == deconv.shape[0]
    assert np.all(obs.index == deconv.index)
    # 1. Create spot x cell-type-wise design matrix.
    type_index_key = "type"
    cell_types = deconv.columns
    # Abundance of cell types in spot (niche):
    obs_niche = pd.DataFrame(deconv.values, columns=[PREFIX_NEIGHBOR + x for x in deconv.columns], index=obs.index)
    # 2. Get design matrices
    # One hot encode index cells:
    dummy_type_annotation = pd.DataFrame({type_index_key: deconv.columns})
    dummy_type_annotation = _make_type_categorical(obs=dummy_type_annotation, key_type=type_index_key)
    obs_index_type = pd.get_dummies(
        dummy_type_annotation, prefix=PREFIX_INDEX, prefix_sep="", columns=[type_index_key], drop_first=False
    )
    dmats = {}
    for i, x in enumerate(cell_types):
        # Create index cell annotation:
        obs_index_type_x = pd.concat([obs_index_type.iloc[[i], [i]] for _ in range(obs.shape[0])], axis=0)
        obs_index_type_x.index = obs.index
        # Merge sample annotation:
        obs_full = pd.concat([obs, obs_index_type_x, obs_niche], axis=1)
        dmats[x] = patsy.dmatrix(formulas[x], obs_full)
        dmats[x] = pd.DataFrame(np.asarray(dmats[x]), index=obs.index, columns=dmats[x].design_info.column_names)
    return dmats


def get_dmat_global_from_deconvoluted(obs: pd.DataFrame, deconv: pd.DataFrame, formula: str) -> pd.DataFrame:
    """
    Create a global design matrix from a sample description table and deconvolution results according to a
    patsy style formula.


    Example for cell types A, B, C:
        columns([batch]) + deconv -> columns([batch, type, index_A, index_B, index_C, neighbor_A, neighbor_B,
                                              neighbor_C])
        and (cells, columns) -> (cells * types, columns), ie the design matrix treats each cell type per spot as an
        observation. This broadcasting works as:
            ([all spots for first index cell, ..., all spots for last index cell], columns)

    See also get_dmats_from_deconvoluted(), coefficient names match between both functions.

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
    obs_niche = pd.DataFrame(deconv.values, columns=[PREFIX_NEIGHBOR + x for x in deconv.columns], index=obs.index)
    # 2. Get design matrices
    # One hot encode index cells:
    dummy_type_annotation = pd.DataFrame({type_index_key: deconv.columns})
    dummy_type_annotation = _make_type_categorical(obs=dummy_type_annotation, key_type=type_index_key)
    obs_index_type = pd.get_dummies(
        dummy_type_annotation, prefix=PREFIX_INDEX, prefix_sep="", columns=[type_index_key], drop_first=False
    )
    dmats = []
    for i, x in enumerate(cell_types):
        # Create index cell annotation:
        # Note that in contrast to get_dmats_from_deconvoluted(), we keep the full column space of the onehot encoding
        # of index cells here as all index cells will be fit in one linear model.
        obs_index_type_x = pd.concat([obs_index_type.iloc[[i], :] for _ in range(obs.shape[0])], axis=0)
        obs_index_type[x].index = obs.index
        # Merge sample annotation:
        obs_full = pd.concat([obs, obs_index_type_x, obs_niche], axis=1)
        dmat_x = patsy.dmatrix(formula, obs_full)
        dmats.append(pd.DataFrame(np.asarray(dmat_x), index=obs.index, columns=dmat_x.design_info.column_names))
    dmat = pd.concat(dmats, axis=0)
    return dmat
