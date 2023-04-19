import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def _get_edge_weights(
    adata,
    alpha,
    pvals_key,
    params_key,
):
    """
    Plot cluster frequencies.

    Args:
        adata: AnnData instance with data and annotation.
        alpha:
        pvals_key:
        params_key:

    """
    de_genes = adata.varm[pvals_key].T < alpha
    de_genes = pd.DataFrame(de_genes.sum(axis=1), columns=["de_genes"])

    de_genes["index"] = [re.search("index_(.*):", de_genes.iloc[i].name).group(1) for i in range(de_genes.shape[0])]
    de_genes["neighbor"] = [
        re.search(":neighbor_(.*)", de_genes.iloc[i].name).group(1) for i in range(de_genes.shape[0])
    ]
    if params_key:
        magnitude = pd.DataFrame(np.log10((adata.varm[params_key].T ** 2).sum(axis=1)), columns=["magnitude"])
        edge_weights = de_genes.join(magnitude)
    else:
        edge_weights = de_genes
    return edge_weights


def _get_graph(x, edge_width_scale, edge_type, clip_edges):
    """
    Plot cluster frequencies.

    Args:
        x: AnnData instance with data and annotation.
        edge_width_scale:
        edge_type:
        clip_edges:

    """
    G = nx.from_pandas_edgelist(
        x, target="index", source="neighbor", edge_attr=["magnitude", "de_genes"], create_using=nx.DiGraph()
    )
    G.remove_edges_from(nx.selfloop_edges(G))
    nodes = np.unique(x["index"])
    pos = nx.circular_layout(G)

    # adjust edge scaling
    if edge_type == "magnitude":
        width = [e["magnitude"] * edge_width_scale for u, v, e in G.edges(data=True) if e["de_genes"] > clip_edges]
    elif edge_type == "de_genes":
        width = [e["de_genes"] * edge_width_scale for u, v, e in G.edges(data=True) if e["de_genes"] > clip_edges]

    return G, pos, nodes, width


def _draw_graph(G, pos, nodes, width, ax, palette, clip_edges):
    """
    Plot cluster frequencies.

    Args:
        G: AnnData instance with data and annotation.
        pos:
        nodes:
        width:
        ax:
        palette:
        clip_edges:

    """
    nx.set_node_attributes(G, dict([(x, palette[i]) for i, x in enumerate(nodes)]), "color")
    node_color = nx.get_node_attributes(G, "color")

    selected_edges = [(u, v) for u, v, e in G.edges(data=True) if e["de_genes"] > clip_edges]

    nx.draw_networkx(
        G,
        pos,
        with_labels=False,
        edgelist=selected_edges,
        width=width,
        arrowstyle="-|>",
        node_color=list(node_color.values()),
        ax=ax,
        connectionstyle="arc3, rad = 0.1",
    )


def circular(
    adata,
    alpha,
    scale_edge,
    pvals_key: str = "ncem_fdr_pvals",
    params_key: str = "ncem_params",
    figsize=(10, 5),
    edge_type: str = "magnitude",
    clip_edges: int = 0,
):
    """
    Plot cluster frequencies.

    Args:
        adata: AnnData instance with data and annotation.
        alpha:
        scale_edge:
        params_key:
        pvals_key:
        figsize:
        edge_type:
        clip_edges:

    """
    from matplotlib import rcParams
    from scanpy.plotting import palettes

    edge_weights = _get_edge_weights(adata, alpha, pvals_key, params_key)
    G, pos, nodes, width = _get_graph(edge_weights, scale_edge, edge_type, clip_edges)

    length = len(nodes)
    # check if default matplotlib palette has enough colors
    if len(rcParams["axes.prop_cycle"].by_key()["color"]) >= length:
        cc = rcParams["axes.prop_cycle"]()
        palette = [next(cc)["color"] for _ in range(length)]

    else:
        if length <= 20:
            palette = palettes.default_20
        elif length <= 28:
            palette = palettes.default_28
        elif length <= len(palettes.default_102):  # 103 colors
            palette = palettes.default_102
        else:
            palette = ["grey" for _ in range(length)]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax[0].axis("off")
    ax[1].axis("off")

    _draw_graph(G, pos, nodes, width, ax=ax[0], palette=palette, clip_edges=clip_edges)
    node_color = nx.get_node_attributes(G, "color")

    values = sorted(list(set(node_color.values())))
    for k, v in node_color.items():
        # make dummy scatterplot to generate labels
        ax[1].scatter([], [], color=v, label=k)

    ax[1].legend(loc="center left", frameon=False)
    plt.tight_layout()
    plt.show()
    return edge_weights


def circular_rotated_labels(
    adata,
    alpha,
    scale_edge,
    pvals_key: str = "ncem_fdr_pvals",
    params_key: str = "ncem_params",
    figsize=(10, 5),
    edge_type: str = "magnitude",
    clip_edges: int = 0,
    text_space: float = 1.15,
):
    """
    Plot cluster frequencies.

    Args:
        adata: AnnData instance with data and annotation.
        alpha:
        scale_edge:
        params_key:
        pvals_key:
        figsize:
        edge_type:
        clip_edges:
        text_space:

    """
    from matplotlib import axes, gridspec, rcParams, ticker
    from scanpy.plotting import palettes

    edge_weights = _get_edge_weights(adata, alpha, pvals_key, params_key)
    G, pos, nodes, width = _get_graph(edge_weights, scale_edge, edge_type, clip_edges)

    length = len(nodes)
    # check if default matplotlib palette has enough colors
    if len(rcParams["axes.prop_cycle"].by_key()["color"]) >= length:
        cc = rcParams["axes.prop_cycle"]()
        palette = [next(cc)["color"] for _ in range(length)]

    else:
        if length <= 20:
            palette = palettes.default_20
        elif length <= 28:
            palette = palettes.default_28
        elif length <= len(palettes.default_102):  # 103 colors
            palette = palettes.default_102
        else:
            palette = ["grey" for _ in range(length)]
            logg.info(
                f"the obs value {value_to_plot!r} has more than 103 categories. Uniform "
                "'grey' color will be used for all categories."
            )

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.axis("off")

    nx.set_node_attributes(G, dict([(x, palette[i]) for i, x in enumerate(nodes)]), "color")
    node_color = nx.get_node_attributes(G, "color")

    selected_edges = [(u, v) for u, v, e in G.edges(data=True) if e["de_genes"] > clip_edges]

    description = nx.draw_networkx_labels(G, pos, font_size=17)
    n = len(nodes)
    node_list = sorted(G.nodes())
    angle = []
    angle_dict = {}
    for i, node in zip(range(n), node_list):
        theta = 2.0 * np.pi * i / n
        angle.append((np.cos(theta), np.sin(theta)))
        angle_dict[node] = theta
    pos = {}
    for node_i, node in enumerate(node_list):
        pos[node] = angle[node_i]

    r = fig.canvas.get_renderer()
    trans = plt.gca().transData.inverted()
    for node, t in description.items():
        bb = t.get_window_extent(renderer=r)
        bbdata = bb.transformed(trans)
        radius = text_space + bbdata.width / 2.0
        position = (radius * np.cos(angle_dict[node]), radius * np.sin(angle_dict[node]))
        t.set_position(position)
        t.set_rotation(angle_dict[node] * 360.0 / (2.0 * np.pi))
        t.set_clip_on(False)

    nx.draw_networkx(
        G,
        pos,
        with_labels=False,
        edgelist=selected_edges,
        width=width,
        arrowstyle="-|>",
        node_color=list(node_color.values()),
        ax=ax,
        connectionstyle="arc3, rad = 0.1",
    )
    node_color = nx.get_node_attributes(G, "color")

    plt.tight_layout()
    plt.show()
