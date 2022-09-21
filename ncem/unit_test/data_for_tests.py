import anndata
import numpy as np
import pandas as pd
import squidpy as sq

KEY_1D = "spatial_coord"
KEY_ADJACENCY = "spatial_connectivities"
KEY_BATCH = "batch"
KEY_COND = "condition"
KEY_DECONV = "deconv"
KEY_TYPE = "type"


def get_adata(simulate_deconvoluted: bool = False, n_conds: int = 2) -> anndata.AnnData:
    # TODO enables conditions > 2 to be used for more complex differential NCEM testing later on.
    n_obs = 200
    n_var = 10
    n_batches = 2
    n_types = 5
    type_prefix = "type_"
    cell_types = [f"{type_prefix}{i}" for i in range(n_types)]

    x = np.random.randint(low=0, high=10, size=(n_obs, n_var))
    a = np.random.randint(low=0, high=1, size=(n_obs, n_obs))
    a[np.arange(0, a.shape[0]), np.arange(0, a.shape[1])] = 1
    obs = pd.DataFrame({
        KEY_BATCH: [f"batch_{i % n_batches}" for i in range(n_obs)],
        KEY_COND: [f"cond_{i % n_conds}" for i in range(n_obs)],
    }, index=[f"cell_{i}" for i in range(n_obs)])
    obsp = {KEY_ADJACENCY: a}
    var = pd.DataFrame({}, index=[f"gene_{i}" for i in range(n_var)])
    adata = anndata.AnnData(X=x, obs=obs, obsp=obsp, var=var)
    if simulate_deconvoluted:
        # Deconvoluted abundances are not normalized, this is not necessary.
        deconv = np.random.uniform(low=0.01, high=1., size=(n_obs, n_types))
        adata.obsm[KEY_DECONV] = pd.DataFrame(deconv, columns=cell_types, index=adata.obs_names)
        # Spot- and type-specific gene expression vectors:
        for x in cell_types:
            adata.layers[x] = np.random.randint(low=0, high=10, size=(n_obs, n_var))
    else:
        adata.obs[KEY_TYPE] = [cell_types[i % n_types] for i in range(n_obs)]
        # Add spatial coordinates of individual cells:
        spatial_key = "spatial"
        key_x = "x"
        key_y = "y"
        # Segment locations:
        adata.obsm[spatial_key] = pd.DataFrame({
            key_x: np.random.uniform(low=0., high=1., size=(n_obs,)),
            key_y: np.random.uniform(low=0., high=1., size=(n_obs,))
        }, index=adata.obs_names)
        sq.gr.spatial_neighbors(adata, spatial_key=spatial_key, coord_type="generic", radius=0.1)
    return adata


def get_adata_1d(simulate_deconvoluted: bool = False, n_conds: int = 2) -> anndata.AnnData:
    adata = get_adata(simulate_deconvoluted=simulate_deconvoluted, n_conds=n_conds)
    # Add 1-coordinate in:
    adata.obs[KEY_1D] = np.random.uniform(low=0., high=1., size=(adata.n_obs,))
    return adata
