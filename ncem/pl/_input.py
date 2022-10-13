from pathlib import Path
from types import MappingProxyType
from typing import Union, Optional, Tuple  # noqa: F401
from typing import Any, Literal, Mapping, Sequence, TYPE_CHECKING

from anndata import AnnData
import scanpy as sc
import numpy as np

from matplotlib import colors as mcolors, pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

from ncem.utils._utils import _assert_categorical_obs

# ToDo
# degree versus distance
# interaction matrices & nhood enrichment -> squidpy
# noise structure per cluster
# umap per image -> scanpy
# spatial allocation -> squidpy
# cluster enrichment (needs a function in tools that produces this)
# umaps of cluster enrichment
# ligrec -> squidpy
# ligrec barplot
# variance decomposition (needs function in tools that produces this


def cluster_freq(
    adata: AnnData,
    cluster_key: str,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, Path]] = None,
    ax: Optional[Axes] = None,
) -> None:
    """
    Plot cluster frequencies.
    Parameters
    ----------
    """
    _assert_categorical_obs(adata, key=cluster_key)
    if title is None:
        title = "Cluster frequencies"

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True, dpi=dpi, figsize=figsize)
    else:
        fig = ax.figure

    fig = adata.obs[cluster_key].value_counts().sort_index(ascending=False).plot(
        kind='barh', ax=ax, title=title
    )
    if save is not None:
        fig.savefig(save)


def noise_structure(
    adata: AnnData,
    cluster_key: str,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
) -> None:
    _assert_categorical_obs(adata, key=cluster_key)
    if title is None:
        title = "Noise structure"

    plotdf = sc.get.obs_df(
        adata,
        keys=list(adata.var_names) + [cluster_key],
    )
    x = np.log(plotdf.groupby(cluster_key).mean() + 1)
    y = np.log(plotdf.groupby(cluster_key).var() + 1)

    nrows = x.shape[0] // 12 + int(x.shape[0] % 12 > 0)
    fig, ax = plt.subplots(
        ncols=12, nrows=nrows, constrained_layout=True, dpi=dpi, figsize=figsize, sharex="all", sharey="all"
    )
    ax = ax.flat
    for axis in ax[x.shape[0]:]:
        axis.remove()

    for i in range(x.shape[0]):
        sns.scatterplot(x=x.iloc[i, :], y=y.iloc[i, :], ax=ax[i])
