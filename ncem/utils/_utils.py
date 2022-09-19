from anndata import AnnData
from pandas.api.types import infer_dtype, is_categorical_dtype


def _assert_categorical_obs(adata: AnnData, key: str) -> None:
    if key not in adata.obs:
        raise KeyError(f"Cluster key `{key}` not found in `adata.obs`.")

    if not is_categorical_dtype(adata.obs[key]):
        raise TypeError(f"Expected `adata.obs[{key!r}]` to be `categorical`, found `{infer_dtype(adata.obs[key])}`.")
