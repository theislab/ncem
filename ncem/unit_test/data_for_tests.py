import anndata
import numpy as np
import pandas as pd

KEY_BATCH = "batch"
KEY_COND = "condition"
KEY_DECONV = "deconv"
KEY_TYPE = "type"


def get_adata(report_deconvolution: bool = False, n_conds: int = 2) -> anndata.AnnData:
    # TODO enables conditions > 2 to be used for more complex differential NCEM testing later on.
    n_obs = 200
    n_var = 10
    n_batches = 2
    n_types = 5
    x = np.random.randint(low=0, high=10, size=(n_obs, n_var))
    obs = pd.DataFrame({
        KEY_BATCH: [f"batch_{i % n_batches}" for i in range(n_obs)],
        KEY_COND: [f"cond_{i % n_conds}" for i in range(n_obs)],
    }, index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame({}, index=[f"gene_{i}" for i in range(n_var)])
    adata = anndata.AnnData(X=x, obs=obs, var=var)
    if report_deconvolution:
        # Deconvolution is not normalized, this is not necessary.
        deconv = np.random.uniform(low=0.01, high=1., size=(n_obs, n_types))
        adata.obs[KEY_DECONV] = deconv
    else:
        adata.obs[KEY_TYPE] = [f"type_{i % n_types}" for i in range(n_obs)]
    return adata
