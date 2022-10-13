import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import re
import scanpy as sc
import numpy as np

from ncem.tools.fit.constants import VARM_KEY_PARAMS, OBSM_KEY_DMAT, OBSM_KEY_DMAT_NICHE, VARM_KEY_FDR_PVALS, \
    VARM_KEY_FDR_PVALS_DIFFERENTIAL,  VARM_KEY_PVALS, VARM_KEY_PVALS_DIFFERENTIAL, VARM_KEY_TESTED_PARAMS, \
    VARM_KEY_TESTED_PARAMS_DIFFERENTIAL, UNS_KEY_CELL_TYPES, UNS_KEY_CONDITIONS, UNS_KEY_PER_INDEX


def _get_edge_weights(
    adata,
    alpha,
    pvals_key,
    params_key,

):
    de_genes = adata.varm[pvals_key].T < alpha
    de_genes = pd.DataFrame(de_genes.sum(axis=1), columns=['de_genes'])

    de_genes['index'] = [re.search('index_(.*):', de_genes.iloc[i].name).group(1) for i in range(de_genes.shape[0])]
    de_genes['neighbor'] = [re.search(':neighbor_(.*)', de_genes.iloc[i].name).group(1) for i in
                             range(de_genes.shape[0])]
    if params_key:
        magnitude = pd.DataFrame(
            (adata.varm[params_key].T ** 2).sum(axis=1),
            columns=['magnitude']
        )
        magnitude = magnitude.iloc[8:, :]
        edge_weights = de_genes.join(magnitude)
    else:
        edge_weights = de_genes
    return edge_weights


def _get_graph(
    x,
    edge_width_scale,
    edge_type,
clip_edges
):
    G = nx.from_pandas_edgelist(
        x, source='index', target='neighbor',
        edge_attr=["magnitude", "de_genes"],
        create_using=nx.DiGraph()
    )
    G.remove_edges_from(nx.selfloop_edges(G))
    nodes = np.unique(x['index'])
    pos = nx.circular_layout(G)

    # adjust edge scaling
    if edge_type == 'magnitude':
        width = [e["magnitude"] * edge_width_scale for u, v, e in G.edges(data=True) if
                 e['de_genes'] > clip_edges]
    elif edge_type == 'de_genes':
        width = [e["de_genes"] * edge_width_scale for u, v, e in G.edges(data=True) if
                 e['de_genes'] > clip_edges]

    return G, pos, nodes, width


def _draw_graph(
    G, pos, nodes, width, ax, palette, clip_edges
):
    nx.set_node_attributes(G, dict([(x, palette[x]) for x in nodes]), "color")
    node_color = nx.get_node_attributes(G, 'color')

    selected_edges = [(u, v) for u, v, e in G.edges(data=True) if e['de_genes'] > clip_edges]

    nx.draw_networkx(
        G, pos, with_labels=False, edgelist=selected_edges,
        width=width, arrowstyle='-|>', node_color=list(node_color.values()),
        ax=ax, connectionstyle='arc3, rad = 0.1'
    )


def circular(
    adata,
    alpha,
    pvals_key,
    params_key,
    scale_edge,
    figsize=(10, 5),
    edge_type: str = 'magnitude',
    clip_edges: int = 0,
):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax[0].axis('off')
    palette = sc.plotting._tools.scatterplots._get_palette(adata, "Cluster")

    edge_weights = _get_edge_weights(adata, alpha, pvals_key, params_key)

    G_control, pos_control, nodes_control, width_control = _get_graph(edge_weights, scale_edge, edge_type, clip_edges)

    _draw_graph(G_control, pos_control, nodes_control, width_control, ax=ax[0], palette=palette, clip_edges=clip_edges)
    node_color = nx.get_node_attributes(G_control, 'color')

    ax[1].axis('off')
    values = sorted(list(set(node_color.values())))
    for k, v in node_color.items():
        # make dummy scatterplot to generate labels
        ax[1].scatter([], [], color=v, label=k)

    ax[1].legend(loc='center left', frameon=False)
    plt.tight_layout()
    plt.show()
