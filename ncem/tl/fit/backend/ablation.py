import warnings
from typing import List, Optional

import anndata
import numpy as np
import pandas as pd
import squidpy as sq
from scipy.stats import linregress

from ncem.tl.fit.constants import OBS_KEY_SPLIT, OBSM_KEY_DMAT, UNS_KEY_ABLATION, VARM_KEY_PARAMS
from ncem.tl.fit.glm import linear_ncem


def _train_split(adata: anndata.AnnData, test_split: float = 0.1, seed: int = 0):
    """
        Train-validation-test split for abalation study

    Args:
        adata: AnnData instance with fits saved.
        test_split: Float indicating faction of nodes used for testing.
        seed: Seed used for random split.

    Returns:
        Anndata instance with added split in .obs.

    """
    np.random.seed(seed)
    node_idx = np.arange(adata.shape[0])

    n_test_nodes = int(adata.shape[0] * test_split)
    test_nodes = np.random.choice(node_idx, n_test_nodes, replace=False)

    adata.obs[f"{OBS_KEY_SPLIT}{seed}"] = "train"
    adata.obs.iloc[test_nodes, adata.obs.columns.get_loc(f"{OBS_KEY_SPLIT}{seed}")] = "test"
    return adata


def ablation(
    adata: anndata.AnnData,
    resolutions: List[float],
    key_type: str,
    library_key: Optional[str] = None,
    coord_type: str = "generic",
    key_graph: str = "spatial_connectivities",
    test_split: float = 0.1,
    n_cvs: int = 3,
):
    """
        Run ablation study for multiply different resolutions and assess performance on test split.

    Args:
        adata: AnnData instance with fits saved.
        resolutions: List of different resolutions that should be evaluated in ablation study.
        key_type: Key of type annotation in .obs.
        library_key: Key of library annotation in .obs which will be used for building the connectivity matrix.
        coord_type: Type of coordinate system.
        key_graph: Key of spatial neighborhood graph in .obsp.
        test_split: Float indicating faction of nodes used for testing.
        n_cvs: Number of cross validations in ablation study.

    Returns:
        Anndata instance with ablation output saved. Ablation output is a dataframe of R squared values across all cells in the test set.

    """

    warnings.filterwarnings("ignore")

    cv_ids = np.arange(n_cvs)
    res_ablation = []
    for cv in cv_ids:
        adata = _train_split(adata, test_split=test_split, seed=cv)
        for res in resolutions:
            sq.gr.spatial_neighbors(
                adata, spatial_key="spatial", library_key=library_key, coord_type=coord_type, radius=res
            )
            train_ad = adata[adata.obs[f"{OBS_KEY_SPLIT}{cv}"] == "train"].copy()
            train_ad = linear_ncem(adata=train_ad, key_type=key_type, key_graph=key_graph)

            test_ad = adata[adata.obs[f"{OBS_KEY_SPLIT}{cv}"] == "test"].copy()
            test_ad = linear_ncem(adata=test_ad, key_type=key_type, key_graph=key_graph)

            pred = np.matmul(test_ad.obsm[OBSM_KEY_DMAT], train_ad.varm[VARM_KEY_PARAMS].T)
            _, _, r, _, _ = linregress(test_ad.X.flatten(), np.array(pred).flatten())
            res_ablation.append([cv, res, r**2])
    adata.uns[UNS_KEY_ABLATION] = pd.DataFrame(res_ablation, columns=["cv", "resolution", "r_squared"])
