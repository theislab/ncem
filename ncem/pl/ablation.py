import anndata
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple  # noqa: F401

from ncem.src.tl.fit.constants import UNS_KEY_ABLATION


def ablation(
    adata: anndata.AnnData,
    figsize: Tuple[float, float] = (3.5, 4.0)
):

    """
        Plot of ablation study results

    Args:
        adata: AnnData instance with fits saved.
        figsize:

    Returns:
        Plot

    """
    sns.set_palette("colorblind")
    plt.ioff()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    sns.lineplot(
        x='resolution',
        y='r_squared',
        style='cv',
        data=adata.uns[UNS_KEY_ABLATION],
        ax=ax,
        markers=True,
        color='orange'
    )
    ax.set_xscale("symlog", linthresh=10)
    plt.axvline(10, color='limegreen', linewidth=3.)
    # add saving
    plt.tight_layout()
    plt.show()
