import pandas as pd
import patsy


def get_dmat_from_obs(obs: pd.DataFrame, formula: str) -> pd.DataFrame:
    """
    Create a design matrix from a sample description table according to a patsy style formula.

    Args:
        obs: Observation-indexed table.
        formula: Model formula.

    Returns: Design matrix.
    """
    # TODO
    pass


def get_dmat_from_deconvoluted(obs: pd.DataFrame, obsm: pd.DataFrame, formula: str) -> pd.DataFrame:
    """
    Create a design matrix from a sample description table and deconvolution results according to a patsy style formula.

    Args:
        obs: Observation-indexed table. Note: does not need spot annotation as obsm carries deconvolution table.
        obsm: Deconvolution result table. Indexed by spots and cell types.
        formula: Model formula.

    Returns: Design matrix.
    """
    # TODO
    # 1. Create spot x cell-type-wise design matrix.
    obs_full = None
    # 2. Get design matrix
    dmat = get_dmat_from_obs(obs=obs_full, formula=formula)
    return dmat


