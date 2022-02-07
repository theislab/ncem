import abc
import warnings
from collections import OrderedDict
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from anndata import AnnData, read_h5ad
from diffxpy.testing.correction import correct
from matplotlib.ticker import FormatStrFormatter
from matplotlib.tri import Triangulation
from omnipath.interactions import import_intercell_network
from pandas import read_csv, read_excel, DataFrame
from scipy import sparse, stats
from tqdm import tqdm


class GraphTools:
    """GraphTools class."""

    celldata: AnnData
    img_celldata: Dict[str, AnnData]

    def compute_adjacency_matrices(
        self, radius: int, coord_type: str = 'generic', n_rings: int = 1, transform: str = None
    ):
        """Compute adjacency matrix for each image in dataset (uses `squidpy.gr.spatial_neighbors`).

        Parameters
        ----------
        radius : int
            Radius of neighbors for non-grid data.
        coord_type : str
            Type of coordinate system.
        n_rings : int
            Number of rings of neighbors for grid data.
        transform : str
            Type of adjacency matrix transform. Valid options are:

            - `spectral` - spectral transformation of the adjacency matrix.
            - `cosine` - cosine transformation of the adjacency matrix.
            - `None` - no transformation of the adjacency matrix.
        """
        pbar_total = len(self.img_celldata.keys())
        with tqdm(total=pbar_total) as pbar:
            for _k, adata in self.img_celldata.items():
                sq.gr.spatial_neighbors(
                    adata=adata,
                    coord_type=coord_type,
                    radius=radius,
                    n_rings=n_rings,
                    transform=transform,
                    key_added="adjacency_matrix"
                )
                pbar.update(1)

    @staticmethod
    def _transform_a(a):
        """Compute degree transformation of adjacency matrix.

        Computes D^(-1) * (A+I), with A an adjacency matrix, I the identity matrix and D the degree matrix.

        Parameters
        ----------
        a
            sparse adjacency matrix.

        Returns
        -------
        degree transformed sparse adjacency matrix
        """
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        degrees = 1 / a.sum(axis=0)
        degrees[a.sum(axis=0) == 0] = 0
        degrees = np.squeeze(np.asarray(degrees))
        deg_matrix = sparse.diags(degrees)
        a_out = deg_matrix * a
        return a_out

    def _transform_all_a(self, a_dict: dict):
        """Compute degree transformation for dictionary of adjacency matrices.

        Computes D^(-1) * (A+I), with A an adjacency matrix, I the identity matrix and D the degree matrix for all
        matrices in a dictionary.

        Parameters
        ----------
        a_dict : dict
            a_dict

        Returns
        -------
        dictionary of degree transformed sparse adjacency matrices
        """
        a_transformed = {i: self._transform_a(a) for i, a in a_dict.items()}
        return a_transformed

    @staticmethod
    def _compute_distance_matrix(pos_matrix):
        """Compute distance matrix.

        Parameters
        ----------
        pos_matrix
            Position matrix.

        Returns
        -------
        distance matrix
        """
        diff = pos_matrix[:, :, None] - pos_matrix[:, :, None].T
        return (diff * diff).sum(1)

    def _get_degrees(self, max_distances: list):
        """Get degrees.

        Parameters
        ----------
        max_distances : list
            List of maximal distances.

        Returns
        -------
        degrees
        """
        degs = {}
        degrees = {}
        for i, adata in self.img_celldata.items():
            pm = np.array(adata.obsm["spatial"])
            dist_matrix = self._compute_distance_matrix(pm)
            degs[i] = {dist: np.sum(dist_matrix < dist * dist, axis=0) for dist in max_distances}
        for dist in max_distances:
            degrees[dist] = [deg[dist] for deg in degs.values()]
        return degrees

    def prepare_spectral_clusters(
            self,
            a_dict,
            n_cluster,
            k_neighbors=10,
    ):
        """
        Computes spectral clusterings for all graphs of the dataset.

        Parameters
        ----------
        a_dict
            A dict of adjacency matrices.
        n_cluster : int
            The number of spectral clusters to be produced for the self-supervision task.
        k_neighbors : int
            The number of neighbors used of the knn graph construction.

        Returns
        -------
        node_to_cluster_mapping
            Dictionary mapping image keys to a one hot encoded matrix (n_nodes, n_clusters) assigning graph nodes
            to their cluster.
        within_cluster_a
            Dictionary mapping image keys to transformed adjacency matrices with edges between clusters removed.
        between_cluster_a
            Dictionary mapping image keys to adjacency matrices describing the connectivity of the clusters.

        """
        from sklearn.cluster import SpectralClustering
        from sklearn.neighbors import kneighbors_graph

        # Compute knn matrices
        knn_matrices = {
            image_key: kneighbors_graph(
                adata.obsm['spatial'],
                n_neighbors=k_neighbors,
                mode='connectivity',  # also 'distance' possible
                include_self=True
            )
            for image_key, adata in self.img_celldata.items()
        }

        # Compute spectral clusters and one-hot encoded assignments from graph nodes to clusters
        clusterer = SpectralClustering(
            n_clusters=n_cluster,
            affinity='precomputed',
        )

        def to_one_hot(a):
            res = np.zeros((len(a), np.max(a) + 1))
            res[np.arange(len(a)), a] = 1
            return res

        node_to_cluster_mapping = {
            image_key: to_one_hot(clusterer.fit_predict(X=value))
            for image_key, value in knn_matrices.items()
        }

        # Compute adjacency matrices containing only within-cluster edges
        within_cluster_a = {
            image_key: a_dict[image_key].multiply(value @ np.transpose(value))
            for image_key, value in node_to_cluster_mapping.items()
        }

        # Compute connectivity of clusters
        between_cluster_a = {
            key: (np.transpose(node_to_cluster_mapping[key]) @ value @ node_to_cluster_mapping[key] > 0).astype(float)
                 - np.eye(node_to_cluster_mapping[key].shape[1])
            for key, value in knn_matrices.items()
        }

        return node_to_cluster_mapping, within_cluster_a, between_cluster_a

    def get_self_supervision_label(
        self,
        label,
        node_to_cluster_mapping,
        between_cluster_a,

    ):
        """
        Computes a label per cluster used for a self-supervision task. This is usually some form of description of the
        surrounding of a cluster.

        Parameters
        ----------
        label : str
            Name of the supervision label to be prepared. Valid options are:

            - 'relative_cell_types' - the cell type frequency of all clusters connected to one cluster.
        node_to_cluster_mapping
            Dictionary mapping image keys to a one hot encoded matrix (n_nodes, n_clusters) assigning graph nodes
            to their cluster.
        between_cluster_a
            Dictionary mapping image keys to adjacency matrices describing the connectivity of the clusters.

        Returns
        -------
        A dict mapping image keys to matrix (n_clusters, n_types) containing the cell type frequencies of all nodes
        within clusters connected to one cluster for all the cluster.
        """

        if label == 'relative_cell_types':
            surrounding_cell_types = {
                image_key: between_cluster_a[image_key] @ np.transpose(node_to_cluster_mapping[image_key])
                           @ self.img_celldata[image_key].obsm['node_types']
                for image_key in node_to_cluster_mapping.keys()
            }
            rel_cell_types = {
                key: value / np.maximum(np.sum(value, axis=1, keepdims=True), np.ones((value.shape[0], 1)))
                for key, value in surrounding_cell_types.items()
            }
            return rel_cell_types
        else:
            raise ValueError(f'Self-supervision label {label} not recognized')

    def process_node_features(
        self,
        node_feature_transformation: str,
    ):
        # Process node-wise features:
        if node_feature_transformation == 'standardize_per_image':
            self._standardize_features_per_image()
        elif node_feature_transformation == 'standardize_globally':
            self._standardize_overall()
        elif node_feature_transformation == 'scale_observations':
            self._scale_observations(n=100)
        elif node_feature_transformation is None or node_feature_transformation == "none":
            pass
        else:
            raise ValueError('Feature transformation %s not recognized!' % node_feature_transformation)

    def _standardize_features_per_image(self):
        for adata in self.img_celldata.values():
            sc.pp.scale(adata)

    def _standardize_overall(self):
        data = np.concatenate([adata.X for adata in self.img_celldata.values()], axis=0)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        for adata in self.img_celldata.values():
            adata.X = adata.X - mean / std

    def _scale_observations(self, n: int):
        """
        TPM-like scaling of observation vectors.
        Only makes sense with positive input.
        :param n: Total feature count to linearly scale observations into.
        :return:
        """
        for adata in self.img_celldata.values():
            adata.X = n * adata.X / adata.X.mean(axis=1)

    def plot_degree_vs_dist(
        self,
        degree_matrices: Optional[list] = None,
        max_distances: Optional[list] = None,
        lateral_resolution: float = 1.0,
        save: Optional[str] = None,
        suffix: str = "_degree_vs_dist.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        """Plot degree versus distances.

        Parameters
        ----------
        degree_matrices : list, optional
            List of degree matrices
        max_distances : list, optional
            List of maximal distances.
        lateral_resolution : float
            Lateral resolution
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        show : bool
            Whether to display plot.
        return_axs : bool
            Whether to return axis objects.

        Returns
        -------
        axis if `return_axs` is True.

        Raises
        ------
        ValueError
            If `degree_matrices` and `max_distances` are `None`.
        """
        if degree_matrices is None:
            if max_distances is None:
                raise ValueError("Provide either distance matrices or distance values!")
            else:
                degree_matrices = self._get_degrees(max_distances)

        plt.ioff()
        fig = plt.figure(figsize=(4, 3))

        mean_degree = []
        distances = []

        for dist, degrees in degree_matrices.items():
            mean_d = [np.mean(degree) for degree in degrees]
            print(np.mean(mean_d))
            mean_degree += mean_d
            distances += [np.int(dist * lateral_resolution)] * len(mean_d)

        sns_data = pd.DataFrame(
            {
                "dist": distances,
                "mean_degree": mean_degree,
            }
        )
        ax = fig.add_subplot(111)
        sns.boxplot(data=sns_data, x="dist", color="steelblue", y="mean_degree", ax=ax)
        ax.set_yscale("log", basey=10)
        ax.grid(False)
        plt.ylabel("")
        plt.xlabel("")
        plt.xticks(rotation=90)

        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None


class PlottingTools:
    """PlottingTools class."""

    celldata: AnnData
    img_celldata: Dict[str, AnnData]

    def celldata_interaction_matrix(
        self,
        fontsize: Optional[int] = None,
        figsize: Tuple[float, float] = (5, 5),
        title: Optional[str] = None,
        save: Optional[str] = None,
        suffix: str = "_celldata_interaction_matrix.pdf",
    ):
        """Compute and plot interaction matrix of celldata.

        The interaction matrix is computed by `squidpy.gr.interaction_matrix()`.

        Parameters
        ----------
        fontsize : int, optional
            Font size.
        figsize : tuple
            Figure size.
        title : str, optional
            Figure title.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        """
        interaction_matrix = []
        cluster_key = self.celldata.uns["metadata"]["cluster_col_preprocessed"]
        with tqdm(total=len(self.img_celldata.keys())) as pbar:
            for adata in self.img_celldata.values():
                im = sq.gr.interaction_matrix(
                    adata, cluster_key=cluster_key, connectivity_key="adjacency_matrix", normalized=False, copy=True
                )
                im = pd.DataFrame(
                    im, columns=list(np.unique(adata.obs[cluster_key])), index=list(np.unique(adata.obs[cluster_key]))
                )
                interaction_matrix.append(im)
                pbar.update(1)
        df_concat = pd.concat(interaction_matrix)
        by_row_index = df_concat.groupby(df_concat.index)
        df_means = by_row_index.sum().sort_index(axis=1)
        interactions = np.array(df_means).T
        self.celldata.uns[f"{cluster_key}_interactions"] = interactions/np.sum(interactions, axis=1)[:, np.newaxis]

        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        if save:
            save = save + suffix
        sq.pl.interaction_matrix(
            self.celldata,
            cluster_key=cluster_key,
            connectivity_key="adjacency_matrix",
            figsize=figsize,
            title=title,
            save=save,
        )

    def celldata_nhood_enrichment(
        self,
        fontsize: Optional[int] = None,
        figsize: Tuple[float, float] = (5, 5),
        title: Optional[str] = None,
        save: Optional[str] = None,
        suffix: str = "_celldata_nhood_enrichment.pdf",
    ):
        """Compute and plot neighbourhood enrichment of celldata.

        The enrichment is computed by `squidpy.gr.nhood_enrichment()`.

        Parameters
        ----------
        fontsize : int, optional
            Font size.
        figsize : tuple
            Figure size.
        title : str, optional
            Figure title.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        """
        zscores = []
        counts = []
        cluster_key = self.celldata.uns["metadata"]["cluster_col_preprocessed"]
        with tqdm(total=len(self.img_celldata.keys())) as pbar:
            for adata in self.img_celldata.values():
                im = sq.gr.nhood_enrichment(
                    adata,
                    cluster_key=cluster_key,
                    connectivity_key="adjacency_matrix",
                    copy=True,
                    show_progress_bar=False,
                )
                zscore = pd.DataFrame(
                    im[0],
                    columns=list(np.unique(adata.obs[cluster_key])),
                    index=list(np.unique(adata.obs[cluster_key])),
                )
                count = pd.DataFrame(
                    im[1],
                    columns=list(np.unique(adata.obs[cluster_key])),
                    index=list(np.unique(adata.obs[cluster_key])),
                )
                zscores.append(zscore)
                counts.append(count)
                pbar.update(1)
        df_zscores = pd.concat(zscores)
        by_row_index = df_zscores.groupby(df_zscores.index)
        df_zscores = by_row_index.mean().sort_index(axis=1)

        df_counts = pd.concat(counts)
        by_row_index = df_counts.groupby(df_counts.index)
        df_counts = by_row_index.sum().sort_index(axis=1)

        self.celldata.uns[f"{cluster_key}_nhood_enrichment"] = {
            "zscore": np.array(df_zscores).T,
            "count": np.array(df_counts).T,
        }
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        if save:
            save = save + suffix
        sq.pl.nhood_enrichment(
            self.celldata,
            cluster_key=cluster_key,
            connectivity_key="adjacency_matrix",
            figsize=figsize,
            title=title,
            save=save,
        )

    def celltype_frequencies(
        self,
        figsize: Tuple[float, float] = (5.0, 6.0),
        fontsize: Optional[int] = None,
        save: Optional[str] = None,
        suffix: str = "_noise_structure.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        """Plot cell type frequencies from celldata on the complete dataset.

        Parameters
        ----------
        fontsize : int, optional
           Font size.
        figsize : tuple
           Figure size.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        show : bool
            Whether to display plot.
        return_axs : bool
            Whether to return axis objects.

        Returns
        -------
        axis
            If `return_axs` is True.
        """
        plt.ioff()
        cluster_id = self.celldata.uns["metadata"]["cluster_col_preprocessed"]
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.barplot(
            y=self.celldata.obs[cluster_id].value_counts().index,
            x=list(self.celldata.obs[cluster_id].value_counts()),
            color="steelblue",
            ax=ax,
        )
        ax.grid(False)
        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None

    def noise_structure(
        self,
        undefined_type: Optional[str] = None,
        merge_types: Optional[Tuple[list, list]] = None,
        min_x: Optional[float] = None,
        max_x: Optional[float] = None,
        panelsize: Tuple[float, float] = (2.0, 2.3),
        fontsize: Optional[int] = None,
        save: Optional[str] = None,
        suffix: str = "_noise_structure.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        """Plot cell type frequencies grouped by cell type.

        Parameters
        ----------
        undefined_type : str, optional
            Undefined cell type.
        merge_types : tuple, optional
            Merge cell types.
        min_x : float, optional
            Minimal x value.
        max_x : float, optional
            Maximal x value.
        fontsize : int, optional
           Font size.
        panelsize : tuple
           Panel size.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        show : bool
            Whether to display plot.
        return_axs : bool
            Whether to return axis objects.

        Returns
        -------
        axis
            If `return_axs` is True.
        """
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        feature_mat = pd.concat(
            [
                pd.concat(
                    [
                        pd.DataFrame(
                            {
                                "image": [k for _i in range(adata.shape[0])],
                            }
                        ),
                        pd.DataFrame(adata.X, columns=list(adata.var_names)),
                        pd.DataFrame(
                            np.asarray(list(adata.uns["node_type_names"].values()))[
                                np.argmax(adata.obsm["node_types"], axis=1)
                            ],
                            columns=["cell_type"],
                        ),
                    ],
                    axis=1,
                ).melt(value_name="expression", var_name="gene", id_vars=["cell_type", "image"])
                for k, adata in self.img_celldata.items()
            ]
        )
        feature_mat["log_expression"] = np.log(feature_mat["expression"].values + 1)
        if undefined_type:
            feature_mat = feature_mat[feature_mat["cell_type"] != undefined_type]

        if merge_types:
            for mt in merge_types[0]:
                feature_mat = feature_mat.replace(mt, merge_types[-1])

        plt.ioff()
        ct = np.unique(feature_mat["cell_type"].values)
        nrows = len(ct) // 12 + int(len(ct) % 12 > 0)
        fig, ax = plt.subplots(
            ncols=12, nrows=nrows, figsize=(12 * panelsize[0], nrows * panelsize[1]), sharex="all", sharey="all"
        )
        ax = ax.flat
        for axis in ax[len(ct) :]:
            axis.remove()
        for i, ci in enumerate(ct):
            tab = feature_mat.loc[feature_mat["cell_type"].values == ci, :]
            x = np.log(tab.groupby(["gene"])["expression"].mean() + 1)
            y = np.log(tab.groupby(["gene"])["expression"].var() + 1)
            sns.scatterplot(x=x, y=y, ax=ax[i])
            min_x = np.min(x) if min_x is None else min_x
            max_x = np.max(x) if max_x is None else max_x
            sns.lineplot(x=[min_x, max_x], y=[2 * min_x, 2 * max_x], color="black", ax=ax[i])
            ax[i].grid(False)
            ax[i].set_title(ci, fontsize=fontsize)
            ax[i].set_xlabel("")
            ax[i].set_ylabel("")
            ax[i].yaxis.set_major_formatter(FormatStrFormatter("%0.1f"))
        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None

    def umap(
        self,
        image_key: str,
        target_cell_type: Optional[str] = None,
        undefined_type: Optional[str] = None,
        n_neighbors: int = 15,
        n_pcs: Optional[int] = None,
        figsize: Tuple[float, float] = (4.0, 4.0),
        fontsize: Optional[int] = None,
        size: Optional[int] = None,
        palette: Optional[str] = None,
        save: Union[str, None] = None,
        suffix: str = "_umap.pdf",
        show: bool = True,
        copy: bool = True,
    ):
        """Plot the umap for one image and optionally for a specific target cell type.

        Parameters
        ----------
        image_key : str
            Image key.
        target_cell_type : str, optional
            Target cell type.
        undefined_type : str, optional
            Undefined cell type.
        n_neighbors : int
            The size of local neighborhood (in terms of number of neighboring data points) used for manifold
            approximation. Larger values result in more global views of the manifold, while smaller values result in
            more local data being preserved. In general values should be in the range 2 to 100.
        n_pcs : int, optional
            Use this many PCs.
        fontsize : int, optional
           Font size.
        figsize : tuple
           Figure size.
        size : int, optional
            Point size. If `None`, is automatically computed as 120000 / n_cells.
        palette : str, optional
            Colors to use for plotting categorical annotation groups. The palette can be a valid `ListedColormap`
            name (`'Set2'`, `'tab20'`, …). If `None`, `mpl.rcParams["axes.prop_cycle"]` is used unless the categorical
            variable already has colors stored in `adata.uns["{var}_colors"]`. If provided, values of
            `adata.uns["{var}_colors"]` will be set.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        show : bool
            Whether to display plot.
        copy : bool
            Whether to return a copy of the AnnaData object.

        Returns
        -------
        AnnData
            If `copy` is True.
        """
        temp_adata = self.img_celldata[image_key].copy()
        cluster_id = temp_adata.uns["metadata"]["cluster_col_preprocessed"]
        if undefined_type:
            temp_adata = temp_adata[temp_adata.obs[cluster_id] != undefined_type]
        if target_cell_type:
            temp_adata = temp_adata[temp_adata.obs[cluster_id] == target_cell_type]
        sc.pp.neighbors(temp_adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.louvain(temp_adata)
        sc.tl.umap(temp_adata)
        print("n cells: ", temp_adata.shape[0])
        if target_cell_type:
            temp_adata.obs[f"{target_cell_type} substates"] = (
                target_cell_type + " " + temp_adata.obs.louvain.astype(str)
            )
            temp_adata.obs[f"{target_cell_type} substates"] = temp_adata.obs[f"{target_cell_type} substates"].astype(
                "category"
            )
            print(temp_adata.obs[f"{target_cell_type} substates"].value_counts())
            color = [f"{target_cell_type} substates"]
        else:
            color = [cluster_id]

        plt.ioff()
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=figsize,
        )
        sc.pl.umap(temp_adata, color=color, ax=ax, show=False, size=size, palette=palette, title="")
        # Save, show and return figure.
        if save is not None:
            plt.savefig(save + image_key + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if copy:
            return temp_adata.copy()

    def spatial(
        self,
        image_key: str,
        undefined_type: Optional[str] = None,
        figsize: Tuple[float, float] = (7.0, 7.0),
        spot_size: int = 30,
        fontsize: Optional[int] = None,
        legend_loc: str = "right margin",
        save: Union[str, None] = None,
        suffix: str = "_spatial.pdf",
        clean_view: bool = False,
        show: bool = True,
        copy: bool = True,
    ):
        """Plot spatial allocation of cells of one image for all cell types.

        Parameters
        ----------
        image_key : str
            Image key.
        undefined_type : str, optional
            Undefined cell type.
        fontsize : int, optional
           Font size.
        figsize : tuple
           Figure size.
        spot_size : int
            Diameter of spot (in coordinate space) for each point. Diameter in pixels of the spots will be
            `size * spot_size * scale_factor`. This argument is required if it cannot be resolved from library info.
        legend_loc : str
            Location of legend, either `'on data'`, `'right margin'` or a valid keyword for the loc parameter of Legend.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        show : bool
            Whether to display plot.
        clean_view : bool
            Whether to show cleaned view.
        copy : bool
            Whether to return a copy of the AnnaData object.

        Returns
        -------
        AnnData
            If `copy` is True.
        """
        temp_adata = self.img_celldata[image_key].copy()
        cluster_id = temp_adata.uns["metadata"]["cluster_col_preprocessed"]
        if undefined_type:
            temp_adata = temp_adata[temp_adata.obs[cluster_id] != undefined_type]

        if clean_view:
            temp_adata = temp_adata[np.argwhere(np.array(temp_adata.obsm["spatial"])[:, 1] < 0).squeeze()]

        plt.ioff()
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sc.pl.spatial(
            temp_adata, color=cluster_id, spot_size=spot_size, legend_loc=legend_loc, ax=ax, show=False, title=""
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + image_key + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if copy:
            return temp_adata

    def compute_cluster_enrichment(
        self,
        image_key: list,
        target_cell_type: str,
        undefined_type: Optional[str] = None,
        filter_titles: Optional[List[str]] = None,
        n_neighbors: Optional[int] = None,
        n_pcs: Optional[int] = None,
        clip_pvalues: Optional[int] = -5,
    ):
        """Compute cluster enrichment for one image and one target cell type.

        Parameters
        ----------
        image_key : list
            Image key.
        target_cell_type : str
            Target cell type.
        undefined_type : str, optional
            Undefined cell type.
        filter_titles : list, optional
            Filter certain titles.
        n_neighbors : int
            The size of local neighborhood (in terms of number of neighboring data points) used for manifold
            approximation. Larger values result in more global views of the manifold, while smaller values result in
            more local data being preserved. In general values should be in the range 2 to 100.
        n_pcs : int, optional
            Use this many PCs.
        clip_pvalues : int, optional
            Clipping value for p-values.

        Returns
        -------
        adata, adata_substates, log_pval, fold_change
        """
        titles = list(self.celldata.uns["node_type_names"].values())
        sorce_type_names = [f"source type {x.replace('_', ' ')}" for x in titles]

        pbar_total = len(self.img_celldata.keys()) + len(self.img_celldata.keys()) + len(titles)
        with tqdm(total=pbar_total) as pbar:
            for adata in self.img_celldata.values():
                source_type = np.matmul(
                    np.asarray(adata.obsp["adjacency_matrix_connectivities"].todense() > 0, dtype="int"),
                    adata.obsm["node_types"],
                )
                source_type = (
                    pd.DataFrame((source_type > 0).astype(str), columns=sorce_type_names)
                    .replace({"True": "in neighbourhood", "False": "not in neighbourhood"}, regex=True)
                    .astype("category")
                )

                for col in source_type.columns:
                    adata.obs[col] = list(source_type[col])
                    adata.obs[col] = adata.obs[col].astype("category")

                pbar.update(1)
                pbar.update(1)

            adata_list = list(self.img_celldata.values())
            adata = adata_list[0].concatenate(adata_list[1:], uns_merge="same")

            cluster_col = self.celldata.uns["metadata"]["cluster_col_preprocessed"]
            image_col = self.celldata.uns["metadata"]["image_col"]
            if undefined_type:
                adata = adata[adata.obs[cluster_col] != undefined_type]

            adata_substates = adata[
                (adata.obs[cluster_col] == target_cell_type) & (adata.obs[image_col].isin(image_key))
            ]
            sc.pp.neighbors(adata_substates, n_neighbors=n_neighbors, n_pcs=n_pcs)
            sc.tl.louvain(adata_substates)
            sc.tl.umap(adata_substates)
            adata_substates.obs[
                f"{target_cell_type} substates"
            ] = f"{target_cell_type} " + adata_substates.obs.louvain.astype(str)
            adata_substates.obs[f"{target_cell_type} substates"] = adata_substates.obs[
                f"{target_cell_type} substates"
            ].astype("category")

            one_hot = pd.get_dummies(adata_substates.obs.louvain, dtype=np.bool)
            # Join the encoded df
            df = adata_substates.obs.join(one_hot)

            distinct_louvain = len(np.unique(adata_substates.obs.louvain))
            pval_source_type = []
            for st in titles:
                pval_cluster = []
                for j in range(distinct_louvain):
                    crosstab = np.array(pd.crosstab(df[f"source type {st}"], df[str(j)]))
                    if crosstab.shape[0] < 2:
                        crosstab = np.vstack([crosstab, [0, 0]])
                    oddsratio, pvalue = stats.fisher_exact(crosstab)
                    pvalue = correct(np.array([pvalue]))
                    pval_cluster.append(pvalue)
                pval_source_type.append(pval_cluster)
                pbar.update(1)

        print("n cells: ", adata_substates.shape[0])
        substate_counts = adata_substates.obs[f"{target_cell_type} substates"].value_counts()
        print(substate_counts)

        columns = [f"{target_cell_type} {x}" for x in np.unique(adata_substates.obs.louvain)]
        pval = pd.DataFrame(
            np.array(pval_source_type).squeeze(), index=[x.replace("_", " ") for x in titles], columns=columns
        )
        log_pval = np.log10(pval)

        if filter_titles:
            log_pval = log_pval.sort_values(columns, ascending=True).filter(items=filter_titles, axis=0)
        if clip_pvalues:
            log_pval[log_pval < clip_pvalues] = clip_pvalues
        fold_change_df = adata_substates.obs[[cluster_col, f"{target_cell_type} substates"] + sorce_type_names]
        counts = pd.pivot_table(
            fold_change_df.replace({"in neighbourhood": 1, "not in neighbourhood": 0}),
            index=[f"{target_cell_type} substates"],
            aggfunc=np.sum,
            margins=True,
        ).T
        counts["new_index"] = [x.replace("source type ", "") for x in counts.index]
        counts = counts.set_index("new_index")

        fold_change = counts.loc[:, columns].div(np.array(substate_counts), axis=1)
        fold_change = fold_change.subtract(np.array(counts["All"] / adata_substates.shape[0]), axis=0)

        if filter_titles:
            fold_change = fold_change.fillna(0).filter(items=filter_titles, axis=0)
        return adata.copy(), adata_substates.copy(), log_pval, fold_change

    def cluster_enrichment(
        self,
        pvalues,
        fold_change,
        figsize: Tuple[float, float] = (4.0, 10.0),
        fontsize: Optional[int] = None,
        pad: float = 0.15,
        pvalues_cmap=None,
        linspace: Optional[Tuple[float, float, int]] = None,
        save: Union[str, None] = None,
        suffix: str = "_cluster_enrichment.pdf",
        show: bool = True,
    ):
        """Plot cluster enrichment (uses the p-values and fold change computed by `compute_cluster_enrichment()`).

        Parameters
        ----------
        pvalues
            P-values.
        fold_change
            Fold change.
        fontsize : int, optional
           Font size.
        figsize : tuple
           Figure size.
        pad : float
            Pad.
        pvalues_cmap : tuple, optional
            Cmap of p-values.
        linspace : tuple, optional
            Linspace.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        show : bool
            Whether to display plot.
        """

        class MidpointNormalize(colors.Normalize):
            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                colors.Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))

        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        m = pvalues.shape[1]
        n = pvalues.shape[0]
        y = np.arange(n + 1)
        x = np.arange(m + 1)
        xs, ys = np.meshgrid(x, y)

        triangles1 = [(i + j * (m + 1), i + 1 + j * (m + 1), i + (j + 1) * (m + 1)) for j in range(n) for i in range(m)]
        triangles2 = [
            (i + 1 + j * (m + 1), i + 1 + (j + 1) * (m + 1), i + (j + 1) * (m + 1)) for j in range(n) for i in range(m)
        ]
        triang1 = Triangulation(xs.ravel() - 0.5, ys.ravel() - 0.5, triangles1)
        triang2 = Triangulation(xs.ravel() - 0.5, ys.ravel() - 0.5, triangles2)
        if not pvalues_cmap:
            pvalues_cmap = plt.get_cmap("Greys_r")
        img1 = plt.tripcolor(
            triang1,
            np.array(pvalues).ravel(),
            cmap=pvalues_cmap,
        )
        img2 = plt.tripcolor(
            triang2, np.array(fold_change).ravel(), cmap=plt.get_cmap("seismic"), norm=MidpointNormalize(midpoint=0.0)
        )

        if linspace:
            ticks = np.linspace(linspace[0], linspace[1], linspace[2], endpoint=True)
            plt.colorbar(
                img2,
                ticks=ticks,
                pad=pad,
                orientation="horizontal",
            ).set_label("fold change")
        else:
            plt.colorbar(
                img2,
                pad=pad,
                orientation="horizontal",
            ).set_label("fold change")
        plt.colorbar(
            img1,
        ).set_label("$log_{10}$ FDR-corrected pvalues")
        plt.xlim(x[0] - 0.5, x[-1] - 0.5)
        plt.ylim(y[0] - 0.5, y[-1] - 0.5)
        plt.yticks(y[:-1])
        plt.xticks(x[:-1])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_yticklabels(list(pvalues.index))
        ax.set_xticklabels(list(pvalues.columns), rotation=90)

        # Save, show and return figure.
        if save is not None:
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

    @staticmethod
    def umaps_cluster_enrichment(
        adata: AnnData,
        filter_titles: list,
        nrows: int = 4,
        ncols: int = 5,
        size: Optional[int] = None,
        figsize: Tuple[float, float] = (18, 12),
        fontsize: Optional[int] = None,
        save: Union[str, None] = None,
        suffix: str = "_cluster_enrichment_umaps.pdf",
        show: bool = True,
    ):
        """Plot cluster enrichment.

        Uses the AnnData object from `compute_cluster_enrichment()`.

        Parameters
        ----------
        adata : AnnData
           Annotated data object.
        filter_titles : list
            Filter certain titles.
        nrows : int
            Number of rows in grid.
        ncols : int
           Number of columns in grid.
        figsize : tuple
           Figure size.
        fontsize : int, optional
           Font size.
        size : int, optional
            Point size. If `None`, is automatically computed as 120000 / n_cells.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        show : bool
            Whether to display plot.
        """
        for x in filter_titles:
            adata.uns[f"source type {x}_colors"] = ["darkgreen", "lightgrey"]
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        plt.ioff()
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
        )
        n = len(filter_titles)
        axs = axs.flat
        for ax in axs[n:]:
            ax.remove()
        ax = axs[:n]

        for i, x in enumerate(filter_titles[:-1]):
            sc.pl.umap(adata, color=f"source type {x}", title=x, show=False, size=size, legend_loc="None", ax=ax[i])
            ax[i].set_xlabel("")
            ax[i].set_ylabel("")
        sc.pl.umap(
            adata,
            color=f"source type {filter_titles[-1]}",
            title=filter_titles[-1],
            size=size,
            show=False,
            ax=ax[n - 1],
        )
        ax[n - 1].set_xlabel("")
        ax[n - 1].set_ylabel("")
        # Save, show and return figure.
        # plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

    def spatial_substates(
        self,
        adata_substates: AnnData,
        image_key: str,
        target_cell_type: str,
        clean_view: bool = False,
        figsize: Tuple[float, float] = (7.0, 7.0),
        spot_size: int = 40,
        fontsize: Optional[int] = None,
        legend_loc: str = "right margin",
        palette: Union[str, list] = "tab10",
        save: Union[str, None] = None,
        suffix: str = "_spatial_substates.pdf",
        show: bool = True,
        copy: bool = False,
    ):
        """Plot spatial allocation of cells.

        Parameters
        ----------
        adata_substates : AnnData
            AnnData substates object.
        image_key : str
            Image key.
        target_cell_type : str
            Target cell type.
        fontsize : int, optional
           Font size.
        figsize : tuple
           Figure size.
        spot_size : int
            Diameter of spot (in coordinate space) for each point. Diameter in pixels of the spots will be
            `size * spot_size * scale_factor`. This argument is required if it cannot be resolved from library info.
        legend_loc : str
            Location of legend, either `'on data'`, `'right margin'` or a valid keyword for the loc parameter of Legend.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        palette : str, optional
            Colors to use for plotting categorical annotation groups. The palette can be a valid `ListedColormap`
            name (`'Set2'`, `'tab20'`, …). If `None`, `mpl.rcParams["axes.prop_cycle"]` is used unless the categorical
            variable already has colors stored in `adata.uns["{var}_colors"]`. If provided, values of
            `adata.uns["{var}_colors"]` will be set.
        suffix : str
            Suffix of file name to save to.
        show : bool
            Whether to display plot.
        clean_view : bool
            Whether to show cleaned view.
        copy : bool
            Whether to return a copy of the AnnaData object.

        Returns
        -------
        AnnData if `copy` is True.
        """
        temp_adata = self.img_celldata[image_key].copy()
        cluster_id = temp_adata.uns["metadata"]["cluster_col_preprocessed"]
        if clean_view:
            temp_adata = temp_adata[np.argwhere(np.array(temp_adata.obsm["spatial"])[:, 1] < 0).squeeze()]
            adata_substates = adata_substates[
                np.argwhere(np.array(adata_substates.obsm["spatial"])[:, 1] < 0).squeeze()
            ]
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=figsize,
        )
        sc.pl.spatial(
            # adata,
            temp_adata[temp_adata.obs[cluster_id] != target_cell_type],
            spot_size=spot_size,
            ax=ax,
            show=False,
            na_color="whitesmoke",
            title="",
        )
        sc.pl.spatial(
            adata_substates,
            color=f"{target_cell_type} substates",
            spot_size=spot_size,
            ax=ax,
            show=False,
            legend_loc=legend_loc,
            title="",
            palette=palette,
        )
        ax.invert_yaxis()
        ax.set_xlabel("")
        ax.set_ylabel("")
        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + image_key + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if copy:
            return temp_adata

    def ligrec(
        self,
        image_key: Optional[str] = None,
        source_groups: Optional[Union[str, Sequence[str]]] = None,
        undefined_type: Optional[str] = None,
        hgnc_names: Optional[List[str]] = None,
        fraction: Optional[float] = None,
        pvalue_threshold: float = 0.3,
        width: float = 3.0,
        seed: int = 10,
        random_state: int = 0,
        fontsize: Optional[int] = None,
        save: Union[str, None] = None,
        suffix: str = "_ligrec.pdf",
        show: bool = True,
        copy: bool = True,
    ):
        """Plot spatial allocation of cells.

        Parameters
        ----------
        image_key : str, optional
            Image key.
        source_groups : str, optional
            Source interaction clusters. If `None`, select all clusters.
        undefined_type : str
            Undefined cell type.
        hgnc_names : list, optional
            List of HGNC names.
        fraction : float, optional
            Subsample to this `fraction` of the number of observations.
        pvalue_threshold : float
            Only show interactions with p-value <= `pvalue_threshold`.
        width : float
            Width.
        seed : int
            Random seed for reproducibility.
        random_state : int
            Random seed to change subsampling.
        fontsize : int, optional
           Font size.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        show : bool
            Whether to display plot.
        copy : bool
            Whether to return a copy of the AnnaData object.

        Returns
        -------
        AnnData if `copy` is True.
        """
        interactions = import_intercell_network(
            transmitter_params={"categories": "ligand"}, receiver_params={"categories": "receptor"}
        )
        if "source" in interactions.columns:
            interactions.pop("source")
        if "target" in interactions.columns:
            interactions.pop("target")
        interactions.rename(
            columns={"genesymbol_intercell_source": "source", "genesymbol_intercell_target": "target"}, inplace=True
        )
        if image_key:
            temp_adata = self.img_celldata[image_key]
        else:
            if fraction:
                temp_adata = sc.pp.subsample(self.celldata, fraction=fraction, copy=True, random_state=random_state)
            else:
                temp_adata = self.celldata.copy()

        cluster_id = temp_adata.uns["metadata"]["cluster_col_preprocessed"]
        if undefined_type:
            temp_adata = temp_adata[temp_adata.obs[cluster_id] != undefined_type]

        print("n cells:", temp_adata.shape[0])
        temp_adata = temp_adata.copy()

        if hgnc_names:
            hgcn_x = pd.DataFrame(temp_adata.X, columns=hgnc_names)
            temp_adata = AnnData(
                X=hgcn_x,
                obs=temp_adata.obs.astype("category"),
                obsm=temp_adata.obsm,
                obsp=temp_adata.obsp,
                uns=temp_adata.uns,
            )

        sq.gr.ligrec(temp_adata, interactions=interactions, cluster_key=cluster_id, use_raw=False, seed=seed)
        if save is not None:
            save = save + image_key + suffix

        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        sq.pl.ligrec(
            temp_adata,
            cluster_key=cluster_id,
            title="",
            source_groups=source_groups,
            pvalue_threshold=pvalue_threshold,
            width=width,
            save=save,
        )
        if show:
            plt.show()

        plt.close()
        plt.ion()

        if copy:
            return temp_adata.copy()

    @staticmethod
    def ligrec_barplot(
        adata: AnnData,
        source_group: str,
        figsize: Tuple[float, float] = (5.0, 4.0),
        fontsize: Optional[int] = None,
        pvalue_threshold: float = 0.05,
        save: Union[str, None] = None,
        suffix: str = "_ligrec_barplot.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        """Plot spatial allocation of cells.

        Parameters
        ----------
        adata : AnnData
            AnnData object.
        source_group : str
            Source interaction cluster.
        figsize : tuple
           Figure size.
        pvalue_threshold : float
            Only show interactions with p-value <= `pvalue_threshold`.
        fontsize : int, optional
           Font size.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        show : bool
            Whether to display plot.
        return_axs : bool
            Whether to return axis objects.

        Returns
        -------
        axis if `return_axs` is True.
        """
        cluster_id = adata.uns["metadata"]["cluster_col_preprocessed"]
        pvals = adata.uns[f"{cluster_id}_ligrec"]["pvalues"].xs(source_group, axis=1)
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.barplot(
            x=list(np.sum(pvals < pvalue_threshold, axis=0).index),
            y=list(np.sum(pvals < pvalue_threshold, axis=0)),
            ax=ax,
            color="steelblue",
        )
        ax.grid(False)
        ax.tick_params(axis="x", labelrotation=90)

        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None

    def compute_variance_decomposition(
        self,
        undefined_type: Optional[str] = None,
    ):
        """Compute variance decomposition.

        Parameters
        ----------
        undefined_type : str
            Undefined cell type.

        Returns
        -------
        var_decomposition
        """
        temp_adata = self.celldata.copy()
        cluster_id = temp_adata.uns["metadata"]["cluster_col_preprocessed"]
        img_col = temp_adata.uns["metadata"]["image_col"]
        if undefined_type:
            temp_adata = temp_adata[temp_adata.obs[cluster_id] != undefined_type]

        df = pd.DataFrame(temp_adata.X, columns=temp_adata.var_names)
        df["image_col"] = pd.Series(list(temp_adata.obs[img_col]), dtype="category")
        df["cluster_col_preprocessed"] = pd.Series(list(temp_adata.obs[cluster_id]), dtype="category")
        images = np.unique(df["image_col"])
        variance_decomposition = []
        with tqdm(total=len(images)) as pbar:
            for img in images:
                mean_img_genes = np.mean(df[df["image_col"] == img], axis=0)
                mean_img_global = np.mean(mean_img_genes)

                intra_ct_var = []
                inter_ct_var = []
                gene_var = []
                for ct in np.unique(df["cluster_col_preprocessed"]):
                    img_celltype = np.array(df[(df["image_col"] == img) & (df["cluster_col_preprocessed"] == ct)])[
                        :, :-2
                    ]
                    if img_celltype.shape[0] == 0:
                        continue
                    mean_image_celltype = np.mean(img_celltype, axis=0)

                    for i in range(img_celltype.shape[0]):
                        intra_ct_var.append((img_celltype[i, :] - mean_image_celltype) ** 2)
                        inter_ct_var.append((mean_image_celltype - mean_img_genes) ** 2)
                        gene_var.append((mean_img_genes - mean_img_global) ** 2)

                intra_ct_var = np.sum(intra_ct_var)
                inter_ct_var = np.sum(inter_ct_var)
                gene_var = np.sum(gene_var)
                variance_decomposition.append(np.array([img, intra_ct_var, inter_ct_var, gene_var]))
                pbar.update(1)
        df = (
            pd.DataFrame(
                variance_decomposition, columns=["image_col", "intra_celltype_var", "inter_celltype_var", "gene_var"]
            )
            .astype(
                {
                    "image_col": str,
                    "intra_celltype_var": "float32",
                    "inter_celltype_var": "float32",
                    "gene_var": "float32",
                }
            )
            .set_index("image_col")
        )

        df["total"] = df.intra_celltype_var + df.inter_celltype_var + df.gene_var
        df["intra cell type variance"] = df.intra_celltype_var / df.total
        df["inter cell type variance"] = df.inter_celltype_var / df.total
        df["gene variance"] = df.gene_var / df.total
        return df

    @staticmethod
    def variance_decomposition(
        df,
        figsize: Tuple[float, float] = (16.0, 3.5),
        fontsize: Optional[int] = None,
        multiindex: bool = False,
        save: Union[str, None] = None,
        suffix: str = "_variance_decomposition.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        """Plot spatial allocation of cells.

        Parameters
        ----------
        df
            Variance decomposition dataframe.
        figsize : tuple,
            Figure size.
        fontsize : int, optional
           Font size.
        multiindex : bool
            Multiindex.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        show : bool
            Whether to display plot.
        return_axs : bool
            Whether to return axis objects.

        Returns
        -------
        axis
            If `return_axs` is True.
        """
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        df.plot(
            y=["intra cell type variance", "inter cell type variance", "gene variance"],
            kind="bar",
            stacked=True,
            figsize=figsize,
            ax=ax,
            colormap="Blues_r",
        )
        if multiindex:

            def process_index(k):
                return tuple(k.split("_"))

            df["index1"], df["index2"] = zip(*map(process_index, df.index))
            df = df.set_index(["index1", "index2"])

            ax.set_xlabel("")
            xlabel_mapping = OrderedDict()
            for index1, index2 in df.index:
                xlabel_mapping.setdefault(index1, [])
                xlabel_mapping[index1].append(index2)

            hline = []
            new_xlabels = []
            for _index1, index2_list in xlabel_mapping.items():
                # slice_list[0] = "{} - {}".format(mouse, slice_list[0])
                index2_list[0] = "{}".format(index2_list[0])
                new_xlabels.extend(index2_list)

                if hline:
                    hline.append(len(index2_list) + hline[-1])
                else:
                    hline.append(len(index2_list))
            ax.set_xticklabels(new_xlabels)
        ax.set_xlabel("")
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None

    def cell_radius(
        self,
        area_key: Optional[str] = None,
        volume_key: Optional[str] = None,
        figsize: Tuple[float, float] = (16.0, 3.5),
        fontsize: Optional[int] = None,
        text_pos: Tuple[float, float] = (1.1, 0.9),
        save: Union[str, None] = None,
        suffix: str = "_distribution_cellradius.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        """Plots the cell radius distribution.

        Parameters
        ----------
        area_key : str, optional
            Key for cell area in obs.
        volume_key : str, optional
            Key for cell volume in obs.
        figsize : tuple,
            Figure size.
        fontsize : int, optional
           Font size.
        text_pos : tuple
            Relative text position.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        show : bool
            Whether to display plot.
        return_axs : bool
            Whether to return axis objects.

        Returns
        -------
        axis
            If `return_axs` is True.
        """
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)

        if area_key:
            x = np.sqrt(self.celldata.obs[area_key])

        if volume_key:
            x = np.cbrt(self.celldata.obs[volume_key])

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.histplot(x, ax=ax)
        plt.axvline(np.mean(x), color='Red', linewidth=2, ax=ax)
        min_ylim, max_ylim = plt.ylim()
        plt.text(np.mean(x) * text_pos[0], max_ylim * text_pos[1], 'mean: {:.2f} $\mu$m'.format(np.mean(x)), ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")

        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None

    def minimal_cell_distance(
        self,
        figsize: Tuple[float, float] = (16.0, 3.5),
        fontsize: Optional[int] = None,
        text_pos: Tuple[float, float] = (1.1, 0.9),
        save: Union[str, None] = None,
        suffix: str = "_distribution_min_celldistance.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        """Plots the minimal cell distance distribution.

        Parameters
        ----------
        figsize : tuple,
            Figure size.
        fontsize : int, optional
           Font size.
        text_pos : tuple
            Relative text position.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        show : bool
            Whether to display plot.
        return_axs : bool
            Whether to return axis objects.

        Returns
        -------
        axis
            If `return_axs` is True.
        """
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)

        x = []
        with tqdm(total=len(self.img_celldata.keys())) as pbar:
            for adata in self.img_celldata.values():
                dist = adata.obsp['adjacency_matrix_distances'].todense()
                for i in range(dist.shape[0]):
                    vec = dist[i, :]
                    vec = vec[vec != 0]
                    if vec.shape[1] == 0:
                        continue
                    x.append(np.min(vec))
                pbar.update(1)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.histplot(x, ax=ax)
        plt.axvline(np.mean(x), color='Red', linewidth=2, ax=ax)
        min_ylim, max_ylim = plt.ylim()
        plt.text(np.mean(x) * text_pos[0], max_ylim * text_pos[1], 'mean: {:.2f} $\mu$m'.format(np.mean(x)), ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")

        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None


class DataLoader(GraphTools, PlottingTools):
    """DataLoader class. Inherits all functions from GraphTools and PlottingTools."""

    def __init__(
        self,
        data_path: str,
        radius: Optional[int] = None,
        coord_type: str = 'generic',
        n_rings: int = 1,
        label_selection: Optional[List[str]] = None,
        n_top_genes: Optional[int] = None,
        cell_type_coarseness: str = 'fine',
    ):
        """Initialize DataLoader.

        Parameters
        ----------
        data_path : str
            Data path.
        radius : int
            Radius.
        label_selection : list, optional
            label selection.
        """
        self.data_path = data_path
        self.cell_type_coarseness = cell_type_coarseness

        print("Loading data from raw files")
        self.register_celldata(n_top_genes=n_top_genes)
        self.register_img_celldata()
        self.register_graph_features(label_selection=label_selection)
        self.compute_adjacency_matrices(radius=radius, coord_type=coord_type, n_rings=n_rings)
        self.radius = radius

        print(
            "Loaded %i images with complete data from %i patients "
            "over %i cells with %i cell features and %i distinct celltypes."
            % (
                len(self.img_celldata),
                len(self.patients),
                self.celldata.shape[0],
                self.celldata.shape[1],
                len(self.celldata.uns["node_type_names"]),
            )
        )

    @property
    def patients(self):
        """Return number of patients in celldata.

        Returns
        -------
        patients
        """
        return np.unique(np.asarray(list(self.celldata.uns["img_to_patient_dict"].values())))

    def register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        print("registering celldata")
        self._register_celldata(n_top_genes=n_top_genes)
        assert self.celldata is not None, "celldata was not loaded"

    def register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        print("collecting image-wise celldata")
        self._register_img_celldata()
        assert self.img_celldata is not None, "image-wise celldata was not loaded"

    def register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        print("adding graph-level covariates")
        self._register_graph_features(label_selection=label_selection)

    @abc.abstractmethod
    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        pass

    @abc.abstractmethod
    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        pass

    @abc.abstractmethod
    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        pass

    def size_factors(self):
        """Get size factors (Only makes sense with positive input).

        Returns
        -------
        sf_dict
        """
        # Check if irregular sums are encountered:
        for i, adata in self.img_celldata.items():
            if np.any(np.sum(adata.X, axis=1) <= 0):
                print("WARNING: found irregular node sizes in image %s" % str(i))
        # Get global mean of feature intensity across all features:
        global_mean_per_node = self.celldata.X.sum(axis=1).mean(axis=0)
        return {i: global_mean_per_node / np.sum(adata.X, axis=1) for i, adata in self.img_celldata.items()}

    @property
    def var_names(self):
        return self.celldata.var_names


class DataLoaderZhang(DataLoader):
    """DataLoaderZhang class. Inherits all functions from DataLoader."""

    cell_type_merge_dict = {
        'fine': {
            "Astrocytes": "Astrocytes",
            "Endothelial": "Endothelial",
            "L23_IT": "L2/3 IT",
            "L45_IT": "L4/5 IT",
            "L5_IT": "L5 IT",
            "L5_PT": "L5 PT",
            "L56_NP": "L5/6 NP",
            "L6_CT": "L6 CT",
            "L6_IT": "L6 IT",
            "L6_IT_Car3": "L6 IT Car3",
            "L6b": "L6b",
            "Lamp5": "Lamp5",
            "Microglia": "Microglia",
            "OPC": "OPC",
            "Oligodendrocytes": "Oligodendrocytes",
            "PVM": "PVM",
            "Pericytes": "Pericytes",
            "Pvalb": "Pvalb",
            "SMC": "SMC",
            "Sncg": "Sncg",
            "Sst": "Sst",
            "Sst_Chodl": "Sst Chodl",
            "VLMC": "VLMC",
            "Vip": "Vip",
            "other": "other",
        }
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 0.109,
            "fn": "preprocessed_zhang.h5ad",
            "image_col": "slice_id",
            "pos_cols": ["center_x", "center_y"],
            "cluster_col": "subclass",
            "cluster_col_preprocessed": "subclass_preprocessed",
            "patient_col": "mouse",
            "cell_type_coarseness": self.cell_type_coarseness,
        }

        celldata = read_h5ad(os.path.join(self.data_path, metadata["fn"])).copy()
        celldata.uns["metadata"] = metadata
        celldata.uns["img_keys"] = list(np.unique(celldata.obs[metadata["image_col"]]))

        img_to_patient_dict = {
            str(x): celldata.obs[metadata["patient_col"]].values[i].split("_")[0]
            for i, x in enumerate(celldata.obs[metadata["image_col"]].values)
        }

        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = celldata.obs[metadata["pos_cols"]]

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="str").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[metadata["cluster_col_preprocessed"]].astype(
            "str"
        )

        # register node type names
        node_type_names = list(np.unique(celldata.obs[metadata["cluster_col_preprocessed"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Save processed data to attributes.
        for adata in self.img_celldata.values():
            graph_covariates = {
                "label_names": {},
                "label_tensors": {},
                "label_selection": [],
                "continuous_mean": {},
                "continuous_std": {},
                "label_data_types": {},
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": {},
            "label_selection": [],
            "continuous_mean": {},
            "continuous_std": {},
            "label_data_types": {},
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DataLoaderJarosch(DataLoader):
    """DataLoaderJarosch class. Inherits all functions from DataLoader."""

    cell_type_merge_dict = {
        'fine': {
            "B cells": "B cells",
            "CD4 T cells": "CD4 T cells",
            "CD8 T cells": "CD8 T cells",
            "GATA3+ epithelial": "GATA3+ epithelial",
            "Ki67 high epithelial": "Ki67 epithelial",
            "Ki67 low epithelial": "Ki67 epithelial",
            "Lamina propria cells": "Lamina propria cells",
            "Macrophages": "Macrophages",
            "Monocytes": "Monocytes",
            "PD-L1+ cells": "PD-L1+ cells",
            "intraepithelial Lymphocytes": "intraepithelial Lymphocytes",
            "muscular cells": "muscular cells",
            "other Lymphocytes": "other Lymphocytes",
        }
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 0.5,
            "fn": "raw_inflamed_colon_1.h5ad",
            "image_col": "Annotation",
            "pos_cols": ["X", "Y"],
            "cluster_col": "celltype_Level_2",
            "cluster_col_preprocessed": "celltype_Level_2_preprocessed",
            "patient_col": None,
            "cell_type_coarseness": self.cell_type_coarseness,
        }

        celldata = read_h5ad(os.path.join(self.data_path, metadata["fn"]))
        feature_cols_hgnc_names = [
            'CD14',
            'MS4A1',
            'IL2RA',
            'CD3G',
            'CD4',
            'PTPRC',
            'PTPRC',
            'PTPRC',
            'CD68',
            'CD8A',
            'KRT5',  # 'KRT1', 'KRT14'
            'FOXP3',
            'GATA3',
            'MKI67',
            'Nuclei',
            'PDCD1',
            'CD274',
            'SMN1',
            'VIM'
        ]
        X = DataFrame(celldata.X, columns=feature_cols_hgnc_names)
        celldata = AnnData(
            X=X, obs=celldata.obs, uns=celldata.uns, obsm=celldata.obsm,
            varm=celldata.varm, obsp=celldata.obsp
        )
        celldata.var_names_make_unique()
        celldata = celldata[celldata.obs[metadata["image_col"]] != "Dirt"].copy()
        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata.obs[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = celldata.obs[metadata["pos_cols"]]

        img_to_patient_dict = {k: "p_1" for k in img_keys}
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="category").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[metadata["cluster_col_preprocessed"]].astype(
            "category"
        )

        # register node type names
        node_type_names = list(np.unique(celldata.obs[metadata["cluster_col_preprocessed"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Save processed data to attributes.
        for adata in self.img_celldata.values():
            graph_covariates = {
                "label_names": {},
                "label_tensors": {},
                "label_selection": [],
                "continuous_mean": {},
                "continuous_std": {},
                "label_data_types": {},
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": {},
            "label_selection": [],
            "continuous_mean": {},
            "continuous_std": {},
            "label_data_types": {},
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DataLoaderHartmann(DataLoader):
    """DataLoaderHartmann class. Inherits all functions from DataLoader."""

    cell_type_merge_dict = {
        'fine': {
            "Imm_other": "Other immune cells",
            "Epithelial": "Epithelial",
            "Tcell_CD4": "CD4 T cells",
            "Myeloid_CD68": "CD68 Myeloid",
            "Fibroblast": "Fibroblast",
            "Tcell_CD8": "CD8 T cells",
            "Endothelial": "Endothelial",
            "Myeloid_CD11c": "CD11c Myeloid",
        }
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 400 / 1024,
            "fn": ["scMEP_MIBI_singlecell/scMEP_MIBI_singlecell.csv", "scMEP_sample_description.xlsx"],
            "image_col": "point",
            "pos_cols": ["center_colcoord", "center_rowcoord"],
            "cluster_col": "Cluster",
            "cluster_col_preprocessed": "Cluster_preprocessed",
            "patient_col": "donor",
            "cell_type_coarseness": self.cell_type_coarseness,
        }
        celldata_df = read_csv(os.path.join(self.data_path, metadata["fn"][0]))
        celldata_df["point"] = [f"scMEP_point_{str(x)}" for x in celldata_df["point"]]
        celldata_df = celldata_df.fillna(0)
        # celldata_df = celldata_df.dropna(inplace=False).reset_index()
        feature_cols = [
            "H3",
            "vimentin",
            "SMA",
            "CD98",
            "NRF2p",
            "CD4",
            "CD14",
            "CD45",
            "PD1",
            "CD31",
            "SDHA",
            "Ki67",
            "CS",
            "S6p",
            "CD11c",
            "CD68",
            "CD36",
            "ATP5A",
            "CD3",
            "CD39",
            "VDAC1",
            "G6PD",
            "XBP1",
            "PKM2",
            "ASCT2",
            "GLUT1",
            "CD8",
            "CD57",
            "LDHA",
            "IDH2",
            "HK1",
            "Ecad",
            "CPT1A",
            "CK",
            "NaKATPase",
            "HIF1A",
            # "X1",
            # "cell_size",
            # "category",
            # "donor",
            # "Cluster",
        ]

        celldata = AnnData(
            X=celldata_df[feature_cols], obs=celldata_df[
                ["point", "cell_id", "cell_size", "donor", "Cluster"]
            ].astype("category")
        )

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): celldata_df[metadata["patient_col"]].values[i]
            for i, x in enumerate(celldata_df[metadata["image_col"]].values)
        }
        # img_to_patient_dict = {k: "p_1" for k in img_keys}
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="category").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[metadata["cluster_col_preprocessed"]].astype(
            "category"
        )

        # register node type names
        node_type_names = list(np.unique(celldata.obs[metadata["cluster_col_preprocessed"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # DEFINE COLUMN NAMES FOR TABULAR DATA.
        # Define column names to extract from patient-wise tabular data:
        patient_col = "ID"
        # These are required to assign the image to dieased and non-diseased:
        disease_features = {"Diagnosis": "categorical"}
        patient_features = {"ID": "categorical", "Age": "continuous", "Sex": "categorical"}
        label_cols = {}
        label_cols.update(disease_features)
        label_cols.update(patient_features)

        if label_selection is None:
            label_selection = set(label_cols.keys())
        else:
            label_selection = set(label_selection)
        label_cols_toread = list(label_selection.intersection(set(list(label_cols.keys()))))
        usecols = label_cols_toread + [patient_col]

        tissue_meta_data = read_excel(os.path.join(self.data_path, "scMEP_sample_description.xlsx"), usecols=usecols)
        # BUILD LABEL VECTORS FROM LABEL COLUMNS
        # The columns contain unprocessed numeric and categorical entries that are now processed to prediction-ready
        # numeric tensors. Here we first generate a dictionary of tensors for each label (label_tensors). We then
        # transform this to have as output of this section dictionary by image with a dictionary by labels as values
        # which can be easily queried by image in a data generator.
        # Subset labels and label types:
        label_cols = {label: nt for label, nt in label_cols.items() if label in label_selection}
        label_tensors = {}
        label_names = {}  # Names of individual variables in each label vector (eg. categories in onehot-encoding).
        # 1. Standardize continuous labels to z-scores:
        continuous_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        continuous_std = {
            feature: tissue_meta_data[feature].std(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "continuous":
                label_tensors[feature] = (tissue_meta_data[feature].values - continuous_mean[feature]) / continuous_std[
                    feature
                ]
                label_names[feature] = [feature]
        # 2. One-hot encode categorical columns
        # Force all entries in categorical columns to be string so that GLM-like formula processing can be performed.
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "categorical":
                tissue_meta_data[feature] = tissue_meta_data[feature].astype("str")
        # One-hot encode each string label vector:
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "categorical":
                oh = pd.get_dummies(tissue_meta_data[feature], prefix=feature, prefix_sep=">", drop_first=False)
                # Change all entries of corresponding observation to np.nan instead.
                idx_nan_col = np.array([i for i, x in enumerate(oh.columns) if x.endswith(">nan")])
                if len(idx_nan_col) > 0:
                    assert len(idx_nan_col) == 1, "fatal processing error"
                    nan_rows = np.where(oh.iloc[:, idx_nan_col[0]].values == 1.0)[0]
                    oh.loc[nan_rows, :] = np.nan
                # Drop nan element column.
                oh = oh.loc[:, [x for x in oh.columns if not x.endswith(">nan")]]
                label_tensors[feature] = oh.values
                label_names[feature] = oh.columns
        # Make sure all tensors are 2D for indexing:
        for feature in list(label_tensors.keys()):
            if len(label_tensors[feature].shape) == 1:
                label_tensors[feature] = np.expand_dims(label_tensors[feature], axis=1)
        # The dictionary of tensor is nested in slices in a dictionary by image which is easier to query with a
        # generator.
        tissue_meta_data_patients = tissue_meta_data[patient_col].values.tolist()
        label_tensors = {
            img: {
                feature_name: np.array(features[tissue_meta_data_patients.index(patient), :], ndmin=1)
                for feature_name, features in label_tensors.items()
            }
            if patient in tissue_meta_data_patients
            else None
            for img, patient in self.celldata.uns["img_to_patient_dict"].items()
        }
        # Reduce to observed patients:
        label_tensors = dict([(k, v) for k, v in label_tensors.items() if v is not None])

        # Save processed data to attributes.
        for k, adata in self.img_celldata.items():
            graph_covariates = {
                "label_names": label_names,
                "label_tensors": label_tensors[k],
                "label_selection": list(label_cols.keys()),
                "continuous_mean": continuous_mean,
                "continuous_std": continuous_std,
                "label_data_types": label_cols,
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": label_names,
            "label_selection": list(label_cols.keys()),
            "continuous_mean": continuous_mean,
            "continuous_std": continuous_std,
            "label_data_types": label_cols,
        }
        self.celldata.uns["graph_covariates"] = graph_covariates

        # self.ref_img_keys = {k: [] for k, v in self.nodes_by_image.items()}


class DataLoaderPascualReguant(DataLoader):
    """DataLoaderPascualReguant class. Inherits all functions from DataLoader."""

    cell_type_merge_dict = {
        'fine': {
            "B cell": "B cells",
            "Endothelial cells": "Endothelial cells",
            "ILC": "ILC",
            "Monocyte/Macrohage/DC": "Monocyte/Macrohage/DC",
            "NK cell": "NK cells",
            "Plasma cell": "Plasma cells CD8",
            "T cytotoxic cell": "T cytotoxic cells",
            "T helper cell": "T helper cells",
            "other": "other",
        }
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 0.325,
            "fn": ["TONSIL_MFI_nuclei_data_table.xlsx", "TONSIL_MFI_membranes_data_table.xlsx"],
            "image_col": "img_keys",
            "pos_cols": ["Location_Center_X", "Location_Center_Y"],
            "cluster_col": "cell_class",
            "cluster_col_preprocessed": "cell_class_preprocessed",
            "patient_col": None,
            "cell_type_coarseness": self.cell_type_coarseness,
        }
        nuclei_df = read_excel(os.path.join(self.data_path, metadata["fn"][0]))
        membranes_df = read_excel(os.path.join(self.data_path, metadata["fn"][1]))

        celldata_df = nuclei_df.join(membranes_df.set_index("ObjectNumber"), on="ObjectNumber")

        feature_cols = [
            "Bcl6",
            "Foxp3",
            "Helios",
            "IRF4",
            "Ki67",
            "Pax5",
            "CCR6",
            "CD103",
            "CD11c",
            "CD123",
            "CD127",
            "CD138",
            "CD14",
            "CD141",
            "CD16",
            "CD161",
            "CD19",
            "CD20",
            "CD21",
            "CD23",
            "CD3",
            "CD31",
            "CD34",
            "CD38",
            "CD4",
            "CD45",
            "CD45RA",
            "CD45RO",
            "CD49a",
            "CD56",
            "CD69",
            "CD7",
            "CD8",
            "CD94",
            "CXCR3",
            "FcER1a",
            "GranzymeA",
            "HLADR",
            "ICOS",
            "IgA",
            "IgG",
            "IgM",
            "Langerin",
            "NKp44",
            "RANKL",
            "SMA",
            "TCRVa72",
            "TCRgd",
            "VCAM",
            "Vimentin",
            "cKit",
        ]
        celldata = AnnData(X=celldata_df[feature_cols], obs=celldata_df[["ObjectNumber", "cell_class"]])

        celldata.uns["metadata"] = metadata
        celldata.obs["img_keys"] = np.repeat("tonsil_image", repeats=celldata.shape[0])
        celldata.uns["img_keys"] = ["tonsil_image"]
        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        celldata.uns["img_to_patient_dict"] = {"tonsil_image": "tonsil_patient"}

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="category").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[metadata["cluster_col_preprocessed"]].astype(
            "category"
        )
        # register node type names
        node_type_names = list(np.unique(celldata.obs[metadata["cluster_col_preprocessed"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Save processed data to attributes.
        for adata in self.img_celldata.values():
            graph_covariates = {
                "label_names": {},
                "label_tensors": {},
                "label_selection": [],
                "continuous_mean": {},
                "continuous_std": {},
                "label_data_types": {},
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": {},
            "label_selection": [],
            "continuous_mean": {},
            "continuous_std": {},
            "label_data_types": {},
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DataLoaderSchuerch(DataLoader):
    """DataLoaderSchuerch class. Inherits all functions from DataLoader."""

    cell_type_merge_dict = {
        'fine': {
            "B cells": "B cells",
            "CD11b+ monocytes": "monocytes",
            "CD11b+CD68+ macrophages": "macrophages",
            "CD11c+ DCs": "dendritic cells",
            "CD163+ macrophages": "macrophages",
            "CD3+ T cells": "CD3+ T cells",
            "CD4+ T cells": "CD4+ T cells",
            "CD4+ T cells CD45RO+": "CD4+ T cells",
            "CD4+ T cells GATA3+": "CD4+ T cells",
            "CD68+ macrophages": "macrophages",
            "CD68+ macrophages GzmB+": "macrophages",
            "CD68+CD163+ macrophages": "macrophages",
            "CD8+ T cells": "CD8+ T cells",
            "NK cells": "NK cells",
            "Tregs": "Tregs",
            "adipocytes": "adipocytes",
            "dirt": "dirt",
            "granulocytes": "granulocytes",
            "immune cells": "immune cells",
            "immune cells / vasculature": "immune cells",
            "lymphatics": "lymphatics",
            "nerves": "nerves",
            "plasma cells": "plasma cells",
            "smooth muscle": "smooth muscle",
            "stroma": "stroma",
            "tumor cells": "tumor cells",
            "tumor cells / immune cells": "immune cells",
            "undefined": "undefined",
            "vasculature": "vasculature",
        },
        'binary': {
            'B cells': 'immune cells',
            'CD11b+ monocytes': 'immune cells',
            'CD11b+CD68+ macrophages': 'immune cells',
            'CD11c+ DCs': 'immune cells',
            'CD163+ macrophages': 'immune cells',
            'CD3+ T cells': 'immune cells',
            'CD4+ T cells': 'immune cells',
            'CD4+ T cells CD45RO+': 'immune cells',
            'CD4+ T cells GATA3+': 'immune cells',
            'CD68+ macrophages': 'immune cells',
            'CD68+ macrophages GzmB+': 'immune cells',
            'CD68+CD163+ macrophages': 'immune cells',
            'CD8+ T cells': 'immune cells',
            'NK cells': 'immune cells',
            'Tregs': 'immune cells',
            'adipocytes': 'other',
            'dirt': 'other',
            'granulocytes': 'immune cells',
            'immune cells': 'immune cells',
            'immune cells / vasculature': 'immune cells',
            'lymphatics': 'immune cells',
            'nerves': 'other',
            'plasma cells': 'other',
            'smooth muscle': 'other',
            'stroma': 'other',
            'tumor cells': 'other',
            'tumor cells / immune cells': 'immune cells',
            'undefined': 'other',
            'vasculature': 'other'
        },
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 0.377442,
            "fn": "CRC_clusters_neighborhoods_markers_NEW.csv",
            "image_col": "File Name",
            "pos_cols": ["X:X", "Y:Y"],
            "cluster_col": "ClusterName",
            "cluster_col_preprocessed": "ClusterName_preprocessed",
            "patient_col": "patients",
            "cell_type_coarseness": self.cell_type_coarseness,
        }
        celldata_df = read_csv(os.path.join(self.data_path, metadata["fn"]))

        feature_cols = [
            "CD44 - stroma:Cyc_2_ch_2",
            "FOXP3 - regulatory T cells:Cyc_2_ch_3",
            "CD8 - cytotoxic T cells:Cyc_3_ch_2",
            "p53 - tumor suppressor:Cyc_3_ch_3",
            "GATA3 - Th2 helper T cells:Cyc_3_ch_4",
            "CD45 - hematopoietic cells:Cyc_4_ch_2",
            "T-bet - Th1 cells:Cyc_4_ch_3",
            "beta-catenin - Wnt signaling:Cyc_4_ch_4",
            "HLA-DR - MHC-II:Cyc_5_ch_2",
            "PD-L1 - checkpoint:Cyc_5_ch_3",
            "Ki67 - proliferation:Cyc_5_ch_4",
            "CD45RA - naive T cells:Cyc_6_ch_2",
            "CD4 - T helper cells:Cyc_6_ch_3",
            "CD21 - DCs:Cyc_6_ch_4",
            "MUC-1 - epithelia:Cyc_7_ch_2",
            "CD30 - costimulator:Cyc_7_ch_3",
            "CD2 - T cells:Cyc_7_ch_4",
            "Vimentin - cytoplasm:Cyc_8_ch_2",
            "CD20 - B cells:Cyc_8_ch_3",
            "LAG-3 - checkpoint:Cyc_8_ch_4",
            "Na-K-ATPase - membranes:Cyc_9_ch_2",
            "CD5 - T cells:Cyc_9_ch_3",
            "IDO-1 - metabolism:Cyc_9_ch_4",
            "Cytokeratin - epithelia:Cyc_10_ch_2",
            "CD11b - macrophages:Cyc_10_ch_3",
            "CD56 - NK cells:Cyc_10_ch_4",
            "aSMA - smooth muscle:Cyc_11_ch_2",
            "BCL-2 - apoptosis:Cyc_11_ch_3",
            "CD25 - IL-2 Ra:Cyc_11_ch_4",
            "CD11c - DCs:Cyc_12_ch_3",
            "PD-1 - checkpoint:Cyc_12_ch_4",
            "Granzyme B - cytotoxicity:Cyc_13_ch_2",
            "EGFR - signaling:Cyc_13_ch_3",
            "VISTA - costimulator:Cyc_13_ch_4",
            "CD15 - granulocytes:Cyc_14_ch_2",
            "ICOS - costimulator:Cyc_14_ch_4",
            "Synaptophysin - neuroendocrine:Cyc_15_ch_3",
            "GFAP - nerves:Cyc_16_ch_2",
            "CD7 - T cells:Cyc_16_ch_3",
            "CD3 - T cells:Cyc_16_ch_4",
            "Chromogranin A - neuroendocrine:Cyc_17_ch_2",
            "CD163 - macrophages:Cyc_17_ch_3",
            "CD45RO - memory cells:Cyc_18_ch_3",
            "CD68 - macrophages:Cyc_18_ch_4",
            "CD31 - vasculature:Cyc_19_ch_3",
            "Podoplanin - lymphatics:Cyc_19_ch_4",
            "CD34 - vasculature:Cyc_20_ch_3",
            "CD38 - multifunctional:Cyc_20_ch_4",
            "CD138 - plasma cells:Cyc_21_ch_3",
            "HOECHST1:Cyc_1_ch_1",
            "CDX2 - intestinal epithelia:Cyc_2_ch_4",
            "Collagen IV - bas. memb.:Cyc_12_ch_2",
            "CD194 - CCR4 chemokine R:Cyc_14_ch_3",
            "MMP9 - matrix metalloproteinase:Cyc_15_ch_2",
            "CD71 - transferrin R:Cyc_15_ch_4",
            "CD57 - NK cells:Cyc_17_ch_4",
            "MMP12 - matrix metalloproteinase:Cyc_21_ch_4",
        ]
        feature_cols_hgnc_names = [
            'CD44',
            'FOXP3',
            'CD8A',
            'TP53',
            'GATA3',
            'PTPRC',
            'TBX21',
            'CTNNB1',
            'HLA-DR',
            'CD274',
            'MKI67',
            'PTPRC',
            'CD4',
            'CR2',
            'MUC1',
            'TNFRSF8',
            'CD2',
            'VIM',
            'MS4A1',
            'LAG3',
            'ATP1A1',
            'CD5',
            'IDO1',
            'KRT1',
            'ITGAM',
            'NCAM1',
            'ACTA1',
            'BCL2',
            'IL2RA',
            'ITGAX',
            'PDCD1',
            'GZMB',
            'EGFR',
            'VISTA',
            'FUT4',
            'ICOS',
            'SYP',
            'GFAP',
            'CD7',
            'CD247',
            'CHGA',
            'CD163',
            'PTPRC',
            'CD68',
            'PECAM1',
            'PDPN',
            'CD34',
            'CD38',
            'SDC1',
            'HOECHST1:Cyc_1_ch_1',  ##
            'CDX2',
            'COL6A1',
            'CCR4',
            'MMP9',
            'TFRC',
            'B3GAT1',
            'MMP12'
        ]
        X = DataFrame(np.array(celldata_df[feature_cols]), columns=feature_cols_hgnc_names)
        celldata = AnnData(X=X, obs=celldata_df[["File Name", "patients", "ClusterName"]])
        celldata.var_names_make_unique()

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): celldata_df[metadata["patient_col"]].values[i]
            for i, x in enumerate(celldata_df[metadata["image_col"]].values)
        }
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="category").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[metadata["cluster_col_preprocessed"]].astype(
            "category"
        )

        # register node type names
        node_type_names = list(np.unique(celldata.obs[metadata["cluster_col_preprocessed"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Graph features are based on TMA spot and not patient, thus patient_col is technically wrong.
        # For aspects where patients are needed (e.g. train-val-test split) the correct patients that are
        # loaded in _register_images() are used
        patient_col = "TMA spot / region"
        disease_features = {}
        patient_features = {"Sex": "categorical", "Age": "continuous"}
        survival_features = {"DFS": "survival"}
        tumor_features = {
            # not sure where these features belong
            "Group": "categorical",
            "LA": "percentage",
            "Diffuse": "percentage",
            "Klintrup_Makinen": "categorical",
            "CLR_Graham_Appelman": "categorical",
        }
        treatment_features = {}
        col_renaming = {}

        label_cols = {}
        label_cols.update(disease_features)
        label_cols.update(patient_features)
        label_cols.update(survival_features)
        label_cols.update(tumor_features)
        label_cols.update(treatment_features)

        if label_selection is None:
            label_selection = set(label_cols.keys())
        else:
            label_selection = set(label_selection)
        label_cols_toread = list(label_selection.intersection(set(list(label_cols.keys()))))
        if "DFS" in label_selection:
            censor_col = "DFS_Censor"
            label_cols_toread = label_cols_toread + [censor_col]
        # there are two LA and Diffuse columns for the two cores that are represented by one patient row
        if "LA" in label_cols_toread:
            label_cols_toread = label_cols_toread + ["LA.1"]
        if "Diffuse" in label_cols_toread:
            label_cols_toread = label_cols_toread + ["Diffuse.1"]
        label_cols_toread_csv = [
            col_renaming[col] if col in list(col_renaming.keys()) else col for col in label_cols_toread
        ]

        usecols = label_cols_toread_csv + [patient_col]
        tissue_meta_data = read_csv(
            os.path.join(self.data_path, "CRC_TMAs_patient_annotations.csv"),
            # sep='\t',
            usecols=usecols,
        )[usecols]
        tissue_meta_data.columns = label_cols_toread + [patient_col]

        # preprocess the loaded csv data:
        # the rows after the first 35 are just descriptions that were included in the excel file
        # for easier work with the data, we expand the data to have two columns per patient representing the two cores
        # that have different LA and Diffuse labels
        patient_data = tissue_meta_data[:35]
        long_patient_data = pd.DataFrame(np.repeat(patient_data.values, 2, axis=0))
        long_patient_data.columns = patient_data.columns
        long_patient_data["copy"] = ["A", "B"] * 35
        if "Diffuse" in label_cols_toread:
            long_patient_data = long_patient_data.rename(columns={"Diffuse": "DiffuseA", "Diffuse.1": "DiffuseB"})
            long_patient_data["Diffuse"] = np.zeros((70,))
            long_patient_data.loc[long_patient_data["copy"] == "A", "Diffuse"] = long_patient_data[
                long_patient_data["copy"] == "A"
            ]["DiffuseA"]
            long_patient_data.loc[long_patient_data["copy"] == "B", "Diffuse"] = long_patient_data[
                long_patient_data["copy"] == "B"
            ]["DiffuseB"]
            long_patient_data.loc[long_patient_data["Diffuse"].isnull(), "Diffuse"] = 0
            # use the proportion of diffuse cores within this spot as probability of being diffuse
            long_patient_data["Diffuse"] = long_patient_data["Diffuse"].astype(float) / 2
            long_patient_data = long_patient_data.drop("DiffuseA", axis=1)
            long_patient_data = long_patient_data.drop("DiffuseB", axis=1)
        if "LA" in label_cols_toread:
            long_patient_data = long_patient_data.rename(columns={"LA": "LAA", "LA.1": "LAB"})
            long_patient_data["LA"] = np.zeros((70,))
            long_patient_data.loc[long_patient_data["copy"] == "A", "LA"] = long_patient_data[
                long_patient_data["copy"] == "A"
            ]["LAA"]
            long_patient_data.loc[long_patient_data["copy"] == "B", "LA"] = long_patient_data[
                long_patient_data["copy"] == "B"
            ]["LAB"]
            long_patient_data.loc[long_patient_data["LA"].isnull(), "LA"] = 0
            # use the proportion of LA cores within this spot as probability of being LA
            long_patient_data["LA"] = long_patient_data["LA"].astype(float) / 2
            long_patient_data = long_patient_data.drop("LAA", axis=1)
            long_patient_data = long_patient_data.drop("LAB", axis=1)
        tissue_meta_data = long_patient_data

        # BUILD LABEL VECTORS FROM LABEL COLUMNS
        # The columns contain unprocessed numeric and categorical entries that are now processed to prediction-ready
        # numeric tensors. Here we first generate a dictionary of tensors for each label (label_tensors). We then
        # transform this to have as output of this section dictionary by image with a dictionary by labels as values
        # which can be easily queried by image in a data generator.
        # Subset labels and label types:
        label_cols = {label: type for label, type in label_cols.items() if label in label_selection}
        label_tensors = {}
        label_names = {}  # Names of individual variables in each label vector (eg. categories in onehot-encoding).
        # 1. Standardize continuous labels to z-scores:
        continuous_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        continuous_std = {
            feature: tissue_meta_data[feature].std(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "continuous":
                label_tensors[feature] = (tissue_meta_data[feature].values - continuous_mean[feature]) / continuous_std[
                    feature
                ]
                label_names[feature] = [feature]
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "percentage":
                label_tensors[feature] = tissue_meta_data[feature]
                label_names[feature] = [feature]
        # 2. One-hot encode categorical columns
        # Force all entries in categorical columns to be string so that GLM-like formula processing can be performed.
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "categorical":
                tissue_meta_data[feature] = tissue_meta_data[feature].astype("str")
        # One-hot encode each string label vector:
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "categorical":
                oh = pd.get_dummies(tissue_meta_data[feature], prefix=feature, prefix_sep=">", drop_first=False)
                # Change all entries of corresponding observation to np.nan instead.
                idx_nan_col = np.array([i for i, x in enumerate(oh.columns) if x.endswith(">nan")])
                if len(idx_nan_col) > 0:
                    assert len(idx_nan_col) == 1, "fatal processing error"
                    nan_rows = np.where(oh.iloc[:, idx_nan_col[0]].values == 1.0)[0]
                    oh.loc[nan_rows, :] = np.nan
                # Drop nan element column.
                oh = oh.loc[:, [x for x in oh.columns if not x.endswith(">nan")]]
                label_tensors[feature] = oh.values
                label_names[feature] = oh.columns
        # 3. Add censoring information to survival
        survival_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "survival"
        }
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "survival":
                label_tensors[feature] = np.concatenate(
                    [
                        np.expand_dims(tissue_meta_data[feature].values / survival_mean[feature], axis=1),
                        np.expand_dims(tissue_meta_data[censor_col].values, axis=1),
                    ],
                    axis=1,
                )
                label_names[feature] = [feature]
        # Make sure all tensors are 2D for indexing:
        for feature in list(label_tensors.keys()):
            if len(label_tensors[feature].shape) == 1:
                label_tensors[feature] = np.expand_dims(label_tensors[feature], axis=1)
        # The dictionary of tensor is nested in slices in a dictionary by image which is easier to query with a
        # generator.
        # tissue_meta_data_patients = tissue_meta_data[patient_col].values.tolist()
        # image keys are of the form reg0xx_A or reg0xx_B with xx going from 01 to 70
        # label tensors have entries (1+2)_A, (1+2)_B, (2+3)_A, (2+3)_B, ...
        img_to_index = {
            img: 2 * ((int(img[4:6]) - 1) // 2) if img[7] == "A" else 2 * ((int(img[4:6]) - 1) // 2) + 1
            for img in self.img_to_patient_dict.keys()
        }
        label_tensors = {
            img: {
                feature_name: np.array(features[index, :], ndmin=1) for feature_name, features in label_tensors.items()
            }
            for img, index in img_to_index.items()
        }

        # Save processed data to attributes.
        for k, adata in self.img_celldata.items():
            graph_covariates = {
                "label_names": label_names,
                "label_tensors": label_tensors[k],
                "label_selection": list(label_cols.keys()),
                "continuous_mean": continuous_mean,
                "continuous_std": continuous_std,
                "label_data_types": label_cols,
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": label_names,
            "label_selection": list(label_cols.keys()),
            "continuous_mean": continuous_mean,
            "continuous_std": continuous_std,
            "label_data_types": label_cols,
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DataLoaderLohoff(DataLoader):
    """DataLoaderLohoff class. Inherits all functions from DataLoader."""

    cell_type_merge_dict = {
        'fine': {
            'Allantois': 'Allantois',
            'Anterior somitic tissues': 'Anterior somitic tissues',
            'Blood progenitors': 'Blood progenitors',
            'Cardiomyocytes': 'Cardiomyocytes',
            'Cranial mesoderm': 'Cranial mesoderm',
            'Definitive endoderm': 'Definitive endoderm',
            'Dermomyotome': 'Dermomyotome',
            'Endothelium': 'Endothelium',
            'Erythroid': 'Erythroid',
            'ExE endoderm': 'ExE endoderm',
            'Forebrain/Midbrain/Hindbrain': 'Forebrain/Midbrain/Hindbrain',
            'Gut tube': 'Gut tube',
            'Haematoendothelial progenitors': 'Haematoendothelial progenitors',
            'Intermediate mesoderm': 'Intermediate mesoderm',
            'Lateral plate mesoderm': 'Lateral plate mesoderm',
            'Low quality': 'Low quality',
            'Mixed mesenchymal mesoderm': 'Mixed mesenchymal mesoderm',
            'NMP': 'NMP',
            'Neural crest': 'Neural crest',
            'Presomitic mesoderm': 'Presomitic mesoderm',
            'Sclerotome': 'Sclerotome',
            'Spinal cord': 'Spinal cord',
            'Splanchnic mesoderm': 'Splanchnic mesoderm',
            'Surface ectoderm': 'Surface ectoderm'
        }
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 1.,
            "fn": "preprocessed_lohoff.h5ad",
            "image_col": "embryo",
            "pos_cols": ["x_global", "y_global"],
            "cluster_col": "celltype_mapped_refined",
            "cluster_col_preprocessed": "celltype_mapped_refined",
            "patient_col": "embryo",
            "cell_type_coarseness": self.cell_type_coarseness,
        }

        celldata = read_h5ad(os.path.join(self.data_path, metadata["fn"])).copy()
        celldata.uns["metadata"] = metadata
        celldata.uns["img_keys"] = list(np.unique(celldata.obs[metadata["image_col"]]))

        img_to_patient_dict = {
            str(x): celldata.obs[metadata["patient_col"]].values[i].split("_")[0]
            for i, x in enumerate(celldata.obs[metadata["image_col"]].values)
        }
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = celldata.obs[metadata["pos_cols"]]

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="category").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[metadata["cluster_col_preprocessed"]].astype(
            "category"
        )
        # register node type names
        node_type_names = list(np.unique(celldata.obs[metadata["cluster_col_preprocessed"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Save processed data to attributes.
        for adata in self.img_celldata.values():
            graph_covariates = {
                "label_names": {},
                "label_tensors": {},
                "label_selection": [],
                "continuous_mean": {},
                "continuous_std": {},
                "label_data_types": {},
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": {},
            "label_selection": [],
            "continuous_mean": {},
            "continuous_std": {},
            "label_data_types": {},
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DataLoaderLuWT(DataLoader):
    """DataLoaderLuWT class. Inherits all functions from DataLoader."""

    cell_type_merge_dict = {
        'fine': {
            "1": "AEC",
            "2": "SEC",
            "3": "MK",
            "4": "Hepatocyte",
            "5": "Macrophage",
            "6": "Myeloid",
            "7": "Erythroid progenitor",
            "8": "Erythroid cell",
            "9": "Unknown",
        }
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 0.1079,
            "fn": "FinalClusteringResults 190517 WT.csv",
            "image_col": "FOV",
            "pos_cols": ["Center_x", "Center_y"],
            "cluster_col": "CellTypeID_new",
            "cluster_col_preprocessed": "CellTypeID_new_preprocessed",
            "cell_type_coarseness": self.cell_type_coarseness,
        }
        celldata_df = read_csv(os.path.join(self.data_path, metadata["fn"]))

        feature_cols = [
            "Abcb4",
            "Abcc3",
            "Adgre1",
            "Ammecr1",
            "Angpt1",
            "Angptl2",
            "Arsb",
            "Axin2",
            "B4galt6",
            "Bmp2",
            "Bmp5",
            "Bmp7",
            "Cd34",
            "Cd48",
            "Cd93",
            "Cdh11",
            "Cdh5",
            "Celsr2",
            "Clec14a",
            "Col4a1",
            "Cspg4",
            "Ctnnal1",
            "Cxadr",
            "Cxcl12",
            "Dkk2",
            "Dkk3",
            "Dll1",
            "Dll4",
            "E2f2",
            "Efnb2",
            "Egfr",
            "Egr1",
            "Eif3a",
            "Elk3",
            "Eng",
            "Ep300",
            "Epcam",
            "Ephb4",
            "Fam46c",
            "Fbxw7",
            "Fgf1",
            "Fgf2",
            "Flt3",
            "Flt4",
            "Fstl1",
            "Fzd1",
            "Fzd2",
            "Fzd3",
            "Fzd4",
            "Fzd5",
            "Fzd7",
            "Fzd8",
            "Gca",
            "Gfap",
            "Gnaz",
            "Gpd1",
            "Hc",
            "Hgf",
            "Hoxb4",
            "Icam1",
            "Igf1",
            "Il6",
            "Il7r",
            "Itga2b",
            "Itgam",
            "Jag1",
            "Jag2",
            "Kdr",
            "Kit",
            "Kitl",
            "Lef1",
            "Lepr",
            "Lox",
            "Lyve1",
            "Maml1",
            "Mecom",
            "Meis1",
            "Meis2",
            "Mertk",
            "Mki67",
            "Mmrn1",
            "Mpl",
            "Mpp1",
            "Mrc1",
            "Mrvi1",
            "Myh10",
            "Ndn",
            "Nes",
            "Nkd2",
            "Notch1",
            "Notch2",
            "Notch3",
            "Notch4",
            "Nrp1",
            "Olr1",
            "Pdgfra",
            "Pdpn",
            "Pecam1",
            "Podxl",
            "Pou2af1",
            "Prickle2",
            "Procr",
            "Proz",
            "Pzp",
            "Rassf4",
            "Rbpj",
            "Runx1",
            "Sardh",
            "Satb1",
            "Sdc3",
            "Sfrp1",
            "Sfrp2",
            "Sgms2",
            "Slamf1",
            "Slc25a37",
            "Stab2",
            "Tcf7",
            "Tcf7l1",
            "Tcf7l2",
            "Tek",
            "Tet1",
            "Tet2",
            "Tfrc",
            "Tgfb2",
            "Timp3",
            "Tmem56",
            "Tmod1",
            "Tox",
            "Vangl2",
            "Vav1",
            "Vcam1",
            "Vwf",
        ]

        celldata = AnnData(
            X=celldata_df[feature_cols],
            obs=celldata_df[["CellID", "FOV", "CellTypeID_new", "Center_x", "Center_y"]]
        )

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): "patient"
            for x in celldata_df[metadata["image_col"]].values
        }
        # img_to_patient_dict = {k: "p_1" for k in img_keys}
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="str").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[metadata["cluster_col_preprocessed"]].astype(
            "str"
        )
        celldata = celldata[celldata.obs[metadata["cluster_col_preprocessed"]] != 'Unknown']

        # register node type names
        node_type_names = list(np.unique(celldata.obs[metadata["cluster_col_preprocessed"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Save processed data to attributes.
        for adata in self.img_celldata.values():
            graph_covariates = {
                "label_names": {},
                "label_tensors": {},
                "label_selection": [],
                "continuous_mean": {},
                "continuous_std": {},
                "label_data_types": {},
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": {},
            "label_selection": [],
            "continuous_mean": {},
            "continuous_std": {},
            "label_data_types": {},
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DataLoaderLuWTimputed(DataLoader):
    """DataLoaderLuWTimputed class. Inherits all functions from DataLoader."""

    cell_type_merge_dict = {
        'fine': {
            1: "AEC",
            2: "SEC",
            3: "MK",
            4: "Hepatocyte",
            5: "Macrophage",
            6: "Myeloid",
            7: "Erythroid progenitor",
            8: "Erythroid cell",
        }
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 0.1079,
            "fn": "merfish_wt_imputed_fetal_liver.h5ad",
            "image_col": "FOV",
            "pos_cols": ["Center_x", "Center_y"],
            "cluster_col": "CellTypeID_new",
            "cluster_col_preprocessed": "CellTypeID_new_preprocessed",
            "n_top_genes": n_top_genes,
            "cell_type_coarseness": self.cell_type_coarseness,
        }
        celldata = read_h5ad(self.data_path + metadata["fn"])
        celldata.uns["metadata"] = metadata
        if n_top_genes:
            sc.pp.highly_variable_genes(celldata, n_top_genes=n_top_genes)
            celldata = celldata[:, celldata.var.highly_variable].copy()
        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Save processed data to attributes.
        for adata in self.img_celldata.values():
            graph_covariates = {
                "label_names": {},
                "label_tensors": {},
                "label_selection": [],
                "continuous_mean": {},
                "continuous_std": {},
                "label_data_types": {},
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": {},
            "label_selection": [],
            "continuous_mean": {},
            "continuous_std": {},
            "label_data_types": {},
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DataLoaderLuTET2(DataLoader):
    """DataLoaderLuTET2 class. Inherits all functions from DataLoader."""

    cell_type_merge_dict = {
        'fine': {
            "1": "AEC",
            "2": "SEC",
            "3": "MK",
            "4": "Hepatocyte",
            "5": "Macrophage",
            "6": "Myeloid",
            "7": "Erythroid progenitor",
            "8": "Erythroid cell",
            "9": "Unknown",
        }
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 0.1079,
            "fn": "FinalClusteringResults 190727 TET2.csv",
            "image_col": "FOV",
            "pos_cols": ["Center_x", "Center_y"],
            "cluster_col": "CellTypeID_new",
            "cluster_col_preprocessed": "CellTypeID_new_preprocessed",
            "cell_type_coarseness": self.cell_type_coarseness,
        }
        celldata_df = read_csv(os.path.join(self.data_path, metadata["fn"]))

        feature_cols = [
            "Abcb4",
            "Abcc3",
            "Adgre1",
            "Ammecr1",
            "Angpt1",
            "Angptl2",
            "Arsb",
            "Axin2",
            "B4galt6",
            "Bmp2",
            "Bmp5",
            "Bmp7",
            "Cd34",
            "Cd48",
            "Cd93",
            "Cdh11",
            "Cdh5",
            "Celsr2",
            "Clec14a",
            "Col4a1",
            "Cspg4",
            "Ctnnal1",
            "Cxadr",
            "Cxcl12",
            "Dkk2",
            "Dkk3",
            "Dll1",
            "Dll4",
            "E2f2",
            "Efnb2",
            "Egfr",
            "Egr1",
            "Eif3a",
            "Elk3",
            "Eng",
            "Ep300",
            "Epcam",
            "Ephb4",
            "Fam46c",
            "Fbxw7",
            "Fgf1",
            "Fgf2",
            "Flt3",
            "Flt4",
            "Fstl1",
            "Fzd1",
            "Fzd2",
            "Fzd3",
            "Fzd4",
            "Fzd5",
            "Fzd7",
            "Fzd8",
            "Gca",
            "Gfap",
            "Gnaz",
            "Gpd1",
            "Hc",
            "Hgf",
            "Hoxb4",
            "Icam1",
            "Igf1",
            "Il6",
            "Il7r",
            "Itga2b",
            "Itgam",
            "Jag1",
            "Jag2",
            "Kdr",
            "Kit",
            "Kitl",
            "Lef1",
            "Lepr",
            "Lox",
            "Lyve1",
            "Maml1",
            "Mecom",
            "Meis1",
            "Meis2",
            "Mertk",
            "Mki67",
            "Mmrn1",
            "Mpl",
            "Mpp1",
            "Mrc1",
            "Mrvi1",
            "Myh10",
            "Ndn",
            "Nes",
            "Nkd2",
            "Notch1",
            "Notch2",
            "Notch3",
            "Notch4",
            "Nrp1",
            "Olr1",
            "Pdgfra",
            "Pdpn",
            "Pecam1",
            "Podxl",
            "Pou2af1",
            "Prickle2",
            "Procr",
            "Proz",
            "Pzp",
            "Rassf4",
            "Rbpj",
            "Runx1",
            "Sardh",
            "Satb1",
            "Sdc3",
            "Sfrp1",
            "Sfrp2",
            "Sgms2",
            "Slamf1",
            "Slc25a37",
            "Stab2",
            "Tcf7",
            "Tcf7l1",
            "Tcf7l2",
            "Tek",
            "Tet1",
            "Tet2",
            "Tfrc",
            "Tgfb2",
            "Timp3",
            "Tmem56",
            "Tmod1",
            "Tox",
            "Vangl2",
            "Vav1",
            "Vcam1",
            "Vwf",
        ]

        celldata = AnnData(
            X=celldata_df[feature_cols],
            obs=celldata_df[["CellID", "FOV", "CellTypeID_new", "Center_x", "Center_y"]]
        )

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): "patient"
            for x in celldata_df[metadata["image_col"]].values
        }
        # img_to_patient_dict = {k: "p_1" for k in img_keys}
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="str").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[metadata["cluster_col_preprocessed"]].astype(
            "str"
        )
        # register node type names
        node_type_names = list(np.unique(celldata.obs[metadata["cluster_col_preprocessed"]]))
        print(node_type_names)
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Save processed data to attributes.
        for adata in self.img_celldata.values():
            graph_covariates = {
                "label_names": {},
                "label_tensors": {},
                "label_selection": [],
                "continuous_mean": {},
                "continuous_std": {},
                "label_data_types": {},
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": {},
            "label_selection": [],
            "continuous_mean": {},
            "continuous_std": {},
            "label_data_types": {},
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DataLoader10xVisiumMouseBrain(DataLoader):
    """DataLoader10xVisiumMouseBrain class. Inherits all functions from DataLoader."""

    cell_type_merge_dict = {
        'fine': {
            'Cortex_1': 'Cortex 1',
            'Cortex_2': 'Cortex 2',
            'Cortex_3': 'Cortex 3',
            'Cortex_4': 'Cortex 4',
            'Cortex_5': 'Cortex 5',
            'Fiber_tract': 'Fiber tract',
            'Hippocampus': 'Hippocampus',
            'Hypothalamus_1': 'Hypothalamus 1',
            'Hypothalamus_2': 'Hypothalamus 2',
            'Lateral_ventricle': 'Lateral ventricle',
            'Pyramidal_layer': 'Pyramidal layer',
            'Pyramidal_layer_dentate_gyrus': 'Pyramidal layer dentate gyrus',
            'Striatum': 'Striatum',
            'Thalamus_1': 'Thalamus 1',
            'Thalamus_2': 'Thalamus 2'
        }
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 1.,
            "fn": "visium_hne_adata.h5ad",
            "image_col": "in_tissue",
            "cluster_col": "cluster",
            "cluster_col_preprocessed": "cluster_preprocessed",
            "patient_col": "in_tissue",
            "n_top_genes": n_top_genes,
            "cell_type_coarseness": self.cell_type_coarseness,
        }

        celldata = read_h5ad(self.data_path + metadata["fn"]).copy()
        if n_top_genes:
            sc.pp.highly_variable_genes(celldata, n_top_genes=n_top_genes)
            celldata = celldata[:, celldata.var.highly_variable].copy()

        celldata.X = celldata.X.toarray()
        celldata.uns["metadata"] = metadata
        celldata.uns["img_keys"] = list(np.unique(celldata.obs[metadata["image_col"]]))

        celldata.uns["img_to_patient_dict"] = {"1": "1"}
        self.img_to_patient_dict = {"1": "1"}

        celldata.obs[metadata["cluster_col"]] = celldata.obs[metadata["cluster_col"]].astype(
            "str"
        )
        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="str").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[metadata["cluster_col_preprocessed"]].astype(
            "str"
        )
        # register node type names
        node_type_names = list(np.unique(celldata.obs[metadata["cluster_col_preprocessed"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Save processed data to attributes.
        for adata in self.img_celldata.values():
            graph_covariates = {
                "label_names": {},
                "label_tensors": {},
                "label_selection": [],
                "continuous_mean": {},
                "continuous_std": {},
                "label_data_types": {},
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": {},
            "label_selection": [],
            "continuous_mean": {},
            "continuous_std": {},
            "label_data_types": {},
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DataLoaderMetabric(DataLoader):
    """DataLoaderMetabric class. Inherits all functions from DataLoader."""

    cell_type_merge_dict = {
        'fine': {
            'B cells': 'B cells',
            'Basal CKlow': 'Tumor cells',
            'Endothelial': 'Endothelial',
            'Fibroblasts': 'Fibroblasts',
            'Fibroblasts CD68+': 'Fibroblasts',
            'HER2+': 'Tumor cells',
            'HR+ CK7-': 'Tumor cells',
            'HR+ CK7- Ki67+': 'Tumor cells',
            'HR+ CK7- Slug+': 'Tumor cells',
            'HR- CK7+': 'Tumor cells',
            'HR- CK7-': 'Tumor cells',
            'HR- CKlow CK5+': 'Tumor cells',
            'HR- Ki67+': 'Tumor cells',
            'HRlow CKlow': 'Tumor cells',
            'Hypoxia': 'Tumor cells',
            'Macrophages Vim+ CD45low': 'Macrophages',
            'Macrophages Vim+ Slug+': 'Macrophages',
            'Macrophages Vim+ Slug-': 'Macrophages',
            'Myoepithelial': 'Myoepithelial',
            'Myofibroblasts': 'Myofibroblasts',
            'T cells': 'T cells',
            'Vascular SMA+': 'Vascular SMA+'
        }
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""

        metadata = {
            "lateral_resolution": None,
            "fn": "single_cell_data/single_cell_data.csv",
            "image_col": "ImageNumber",
            "pos_cols": ['Location_Center_X', 'Location_Center_Y'],
            "cluster_col": "description",
            "cluster_col_preprocessed": "description_preprocessed",
            "patient_col": "metabricId",
            "cell_type_coarseness": self.cell_type_coarseness,
        }
        celldata_df = read_csv(os.path.join(self.data_path, metadata["fn"]))

        cell_count = pd.DataFrame({'count': celldata_df.groupby(metadata['image_col']).size()}).reset_index()
        img_ids_pass = set([
            x for x, y in zip(cell_count[metadata['image_col']].values, cell_count["count"].values) if y >= 100
        ])
        celldata_df = celldata_df.iloc[np.where([x in img_ids_pass for x in celldata_df[metadata['image_col']].values])[0], :]

        feature_cols = [
            'HH3_total',
            'CK19',
            'CK8_18',
            'Twist',
            'CD68',
            'CK14',
            'SMA',
            'Vimentin',
            'c_Myc',
            'HER2',
            'CD3',
            'HH3_ph',
            'Erk1_2',
            'Slug',
            'ER',
            'PR',
            'p53',
            'CD44',
            'EpCAM',
            'CD45',
            'GATA3',
            'CD20',
            'Beta_catenin',
            'CAIX',
            'E_cadherin',
            'Ki67',
            'EGFR',
            'pS6',
            'Sox9',
            'vWF_CD31',
            'pmTOR',
            'CK7',
            'panCK',
            'c_PARP_c_Casp3',
            'DNA1',
            'DNA2',
            'H3K27me3',
            'CK5',
            'Fibronectin'
        ]
        X = DataFrame(np.array(celldata_df[feature_cols]), columns=feature_cols)
        celldata = AnnData(
            X=X,
            obs=celldata_df[[metadata['image_col'], metadata['patient_col'], metadata['cluster_col']]]
        )
        celldata.var_names_make_unique()

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): celldata_df[metadata["patient_col"]].values[i]
            for i, x in enumerate(celldata_df[metadata["image_col"]].values)
        }
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="category").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[metadata["cluster_col_preprocessed"]].astype("category")

        # register node type names
        node_type_names = list(np.unique(celldata.obs[metadata["cluster_col_preprocessed"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {image key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.
        Parameters
        ----------
        label_selection
            Label selection.
        """
        patient_col = 'METABRIC.ID'
        disease_features = {
            'grade': 'categorical',
            'grade_collapsed': 'categorical',
            'tumor_size': 'continuous',
            'hist_type': 'categorical',
            'stage': 'categorical'
        }
        patient_features = {
            'age': 'continuous',
            'menopausal': 'categorical'
        }
        survival_features = {
            'time_last_seen': 'survival'
        }
        tumor_features = {
            'ERstatus': 'categorical',
            'lymph_pos': 'continuous'

        }
        treatment_features = {
            'CT': 'categorical',
            'HT': 'categorical',
            'RT': 'categorical',
            'surgery': 'categorical',
            'NPI': 'categorical'
        }
        col_renaming = {  # column aliases for convenience within this function
            'grade': 'Grade',
            'tumor_size': 'Size',
            'hist_type': 'Histological.Type',
            'stage': 'Stage',
            'age': 'Age.At.Diagnosis',
            'menopausal': 'Inferred.Menopausal.State',
            'death_breast': 'DeathBreast',
            'time_last_seen': 'T',
            'ERstatus': 'ER.Status',
            'lymph_pos': 'Lymph.Nodes.Positive',
            'surgery': 'Breast.Surgery'
        }

        label_cols = {}
        label_cols.update(disease_features)
        label_cols.update(patient_features)
        label_cols.update(survival_features)
        label_cols.update(tumor_features)
        label_cols.update(treatment_features)

        if label_selection is None:
            label_selection = set(label_cols.keys())
        else:
            label_selection = set(label_selection)
        label_cols_toread = list(label_selection.intersection(set(list(label_cols.keys()))))
        label_cols_toread = [label for label in label_cols_toread if label != 'grade_collapsed']
        if 'time_last_seen' in label_selection:
            censor_col = 'death_breast'
            label_cols_toread = label_cols_toread + [censor_col]
        if 'grade_collapsed' in label_selection and 'grade' not in label_selection:
            label_cols_toread.append('grade')
        label_cols_toread_csv = [
            col_renaming[col] if col in list(col_renaming.keys()) else col
            for col in label_cols_toread
        ]

        usecols = label_cols_toread_csv + [patient_col] + ['Cohort', 'Date.Of.Diagnosis']
        tissue_meta_data = read_csv(
            os.path.join(self.data_path + 'single_cell_data/41586_2019_1007_MOESM7_ESM.csv'),
            sep='\t',
            usecols=usecols
        )[usecols]
        tissue_meta_data.columns = label_cols_toread + [patient_col] + ['cohort', 'date_of_diagnosis']

        if "grade_collapsed" in label_selection:
            tissue_meta_data['grade_collapsed'] = ['3' if grade == '3' else '1&2' for grade in
                                                   tissue_meta_data['grade']]
        if 'grade_collapsed' in label_selection and 'grade' not in label_selection:
            tissue_meta_data.drop('grade', 1, inplace=True)

        # BUILD LABEL VECTORS FROM LABEL COLUMNS
        # The columns contain unprocessed numeric and categorical entries that are now processed to prediction-ready
        # numeric tensors. Here we first generate a dictionary of tensors for each label (label_tensors). We then
        # transform this to have as output of this section dictionary by image with a dictionary by labels as values
        # which can be easily queried by image in a data generator.
        # Subset labels and label types:
        label_cols = {label: type for label, type in label_cols.items() if label in label_selection}
        label_tensors = {}
        label_names = {}  # Names of individual variables in each label vector (eg. categories in onehot-encoding).
        # 1. Standardize continuous labels to z-scores:
        continuous_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == 'continuous'
        }
        continuous_std = {
            feature: tissue_meta_data[feature].std(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == 'continuous'
        }
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == 'continuous':
                label_tensors[feature] = (tissue_meta_data[feature].values - continuous_mean[feature]) \
                                         / continuous_std[feature]
                label_names[feature] = [feature]
        # 2. One-hot encode categorical columns
        # Force all entries in categorical columns to be string so that GLM-like formula processing can be performed.
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == 'categorical':
                tissue_meta_data[feature] = tissue_meta_data[feature].astype('str')
        # One-hot encode each string label vector:
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == 'categorical':
                oh = pd.get_dummies(
                    tissue_meta_data[feature],
                    prefix=feature,
                    prefix_sep='>',
                    drop_first=False
                )
                # Change all entries of corresponding observation to np.nan instead.
                idx_nan_col = np.array([i for i, x in enumerate(oh.columns) if x.endswith('>nan')])
                if len(idx_nan_col) > 0:
                    assert len(idx_nan_col) == 1, "fatal processing error"
                    nan_rows = np.where(oh.iloc[:, idx_nan_col[0]].values == 1.)[0]
                    oh.loc[nan_rows, :] = np.nan
                # Drop nan element column.
                oh = oh.loc[:, [x for x in oh.columns if not x.endswith('>nan')]]
                label_tensors[feature] = oh.values
                label_names[feature] = oh.columns
        # 3. Add censoring information to survival
        survival_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == 'survival'
        }
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == 'survival':
                label_tensors[feature] = np.concatenate([
                    np.expand_dims(tissue_meta_data[feature].values / survival_mean[feature], axis=1),
                    np.expand_dims(tissue_meta_data[censor_col].values, axis=1),
                ], axis=1)
                label_names[feature] = [feature]
        # Make sure all tensors are 2D for indexing:
        for feature in list(label_tensors.keys()):
            if len(label_tensors[feature].shape) == 1:
                label_tensors[feature] = np.expand_dims(label_tensors[feature], axis=1)
        # The dictionary of tensor is nested in slices in a dictionary by image which is easier to query with a
        # generator.
        tissue_meta_data_patients = tissue_meta_data[patient_col].values.tolist()
        label_tensors = {
            img: {
                feature_name: np.array(features[tissue_meta_data_patients.index(patient), :], ndmin=1)
                for feature_name, features in label_tensors.items()
            } if patient in tissue_meta_data_patients else None
            for img, patient in self.img_to_patient_dict.items()
        }
        # Reduce data to patients with graph-level labels:
        label_tensors = {k: v for k, v in label_tensors.items() if v is not None}
        img_keys = label_tensors.keys()
        self.img_celldata = {k: adata for k, adata in self.img_celldata.items() if k in img_keys}
        im_col = self.celldata.uns['metadata']['image_col']
        self.celldata = self.celldata[[str(im) in img_keys for im in self.celldata.obs[im_col]]]

        img_to_patient_dict = {im: pat for im, pat in self.img_to_patient_dict.items() if im in img_keys}
        self.celldata.uns['img_to_patient_dict'] = img_to_patient_dict
        for adata in self.img_celldata.values():
            adata.uns['img_to_patient_dict'] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # Save processed data to attributes.
        for k, adata in self.img_celldata.items():
            graph_covariates = {
                "label_names": label_names,
                "label_tensors": label_tensors[k],
                "label_selection": list(label_cols.keys()),
                "continuous_mean": continuous_mean,
                "continuous_std": continuous_std,
                "label_data_types": label_cols,
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": label_names,
            "label_selection": list(label_cols.keys()),
            "continuous_mean": continuous_mean,
            "continuous_std": continuous_std,
            "label_data_types": label_cols,
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DataLoaderBaselZurichZenodo(DataLoader):
    """DataLoaderBaselZurichZenodo class. Inherits all functions from DataLoader."""

    cell_type_merge_dict = {
        'fine': {
            "1": "B cells",
            "2": "T and B cells",
            "3": "T cells",
            "4": "macrophages",
            "5": "T cells",
            "6": "macrophages",
            "7": "endothelial",
            "8": "stromal cells", # "vimentin hi stromal cell",
            "9": "stromal cells", # "small circular stromal cell",
            "10": "stromal cells", # "small elongated stromal cell",
            "11": "stromal cells", # "fibronectin hi stromal cell",
            "12": "stromal cells", # "large elongated stromal cell",
            "13": "stromal cells", # "SMA hi vimentin hi stromal cell",
            "14": "tumor cells", #"hypoxic tumor cell",
            "15": "tumor cells", #"apoptotic tumor cell",
            "16": "tumor cells", #"proliferative tumor cell",
            "17": "tumor cells", #"p53+ EGFR+ tumor cell",
            "18": "tumor cells", #"basal CK tumor cell",
            "19": "tumor cells", #"CK7+ CK hi cadherin hi tumor cell",
            "20": "tumor cells", #"CK7+ CK+ tumor cell",
            "21": "tumor cells", #"epithelial low tumor cell",
            "22": "tumor cells", #"CK low HR low tumor cell",
            "23": "tumor cells", #"CK+ HR hi tumor cell",
            "24": "tumor cells", #"CK+ HR+ tumor cell",
            "25": "tumor cells", #"CK+ HR low tumor cell",
            "26": "tumor cells", #"CK low HR hi p53+ tumor cell",
            "27": "tumor cells", #"myoepithelial tumor cell"
        }
    }

    def _register_images(self):
        """
        Creates mapping of full image names to shorter identifiers.
        """

        # Define mapping of image identifiers to numeric identifiers:
        img_tab_basel = read_csv(
            self.data_path + 'Data_publication/BaselTMA/Basel_PatientMetadata.csv',
            usecols=['core', 'FileName_FullStack', 'PID', 'diseasestatus'],
            dtype={'core': str, 'FileName_FullStack': str, 'PID': str, 'diseasestatus': str}
        )
        img_tab_basel['PID'] = ["b" + str(p) for p in img_tab_basel['PID'].values]
        img_tab_zurich = read_csv(
            self.data_path + 'Data_publication/ZurichTMA/Zuri_PatientMetadata.csv',
            usecols=['core', 'FileName_FullStack', 'grade', 'PID', 'location'],
            dtype={'core': str, 'FileName_FullStack': str, 'grade': str, 'PID': str, 'location': str}
        )
        img_tab_zurich['PID'] = ["z" + str(p) for p in img_tab_zurich['PID'].values]
        img_tab_zurich['diseasestatus'] = [
            'tumor' if a else 'non-tumor' for a in img_tab_zurich['location'] != '[]'
        ]
        img_tab_zurich = img_tab_zurich.drop('location', axis=1)
        # drop Metastasis images
        img_tab_zurich = img_tab_zurich[img_tab_zurich['grade'] != 'METASTASIS'].drop('grade', axis=1)
        img_tab_bz = pd.concat([img_tab_basel, img_tab_zurich], axis=0, sort=True, ignore_index=True)
        img_tab_bz = img_tab_bz[img_tab_bz['diseasestatus'] == 'tumor']

        self.img_key_to_fn = dict(img_tab_bz[['core', 'FileName_FullStack']].values)
        self.img_to_patient_dict = dict(img_tab_bz[['core', 'PID']].values)

    def _load_node_positions(self):
        from PIL import Image

        position_matrix = []
        for k, fn in self.img_key_to_fn.items():
            fn = self.data_path + "OMEnMasks/Basel_Zuri_masks/" + fn
            # Mask file have slightly different file name, extended either by _mask or _maks:
            if os.path.exists(".".join(fn.split(".")[:-1]) + "_maks.tiff"):
                fn = ".".join(fn.split(".")[:-1]) + "_maks.tiff"
            elif os.path.exists(".".join(fn.split(".")[:-1]) + "_mask.tiff"):
                fn = ".".join(fn.split(".")[:-1]) + "_mask.tiff"
            else:
                raise ValueError("file %s not found" % fn)

            # Load image from tiff:
            img_array = np.array(Image.open(fn))
            # Throughout all files, nodes are refered to via the string core_id+"_"+str(i) where i is the integer
            # encoding the object in the segmentation mask.
            node_ids_img = np.sort(np.unique(img_array))
            # 0 encodes background:
            node_ids_img = node_ids_img[node_ids_img != 0]
            # Only ranks of objects encoded in masks are used!  # TODO check
            node_ids_rank = np.arange(1, len(node_ids_img) + 1)
            # Drop images with fewer than 100 nodes
            if len(node_ids_rank) < 100:
                continue
            # Find centre of object mask of each node:  # TODO check, rank used
            center_array = [np.where(img_array == node_ids_img[i - 1]) for i in node_ids_rank]
            pm = np.array([[f'{k}_{i+1}', x[0].mean(), x[1].mean()] for i, x in enumerate(center_array)])
            position_matrix.append(pm)
        position_matrix = np.concatenate(position_matrix, axis=0)
        position_matrix = pd.DataFrame(position_matrix, columns=['id', 'x', 'y'])
        return position_matrix

    def _load_node_features(self):
        full_cell_key_col = "id"  # column with full cell identifier (including image identifier)
        feature_col = "channel"
        signal_col = "mc_counts"

        features_basel = read_csv(
            self.data_path + 'Data_publication/BaselTMA/SC_dat.csv',
            usecols=[full_cell_key_col, feature_col, signal_col],
            dtype={full_cell_key_col: str, feature_col: str, signal_col: float}
        )
        features_zurich = read_csv(
            self.data_path + 'Data_publication/ZurichTMA/SC_dat.csv',
            usecols=[full_cell_key_col, feature_col, signal_col],
            dtype={full_cell_key_col: str, feature_col: str, signal_col: float}
        )
        features_zb = pd.concat([
            features_basel,
            features_zurich
        ], axis=0, ignore_index=True)
        node_features = features_zb.pivot_table(index='id', columns='channel', values='mc_counts')

        return node_features

    def _load_node_types(self):
        """
        Loads the cell types.
        """
        # Direct meta cluster annotation from main text Fig 1b
        # Also hard coded maps from cluster numbers to annotation as in https://github.com/BodenmillerGroup/SCPathology_publication/blob/4e99e10c2bc6d0f1dd168d534df39870d1ecb549/R/BaselTMA_pipeline.Rmd#L471

        full_cell_key_col = "id"  # column with full cell identifier (including image identifier)
        cluster_col = "cluster"
        node_cluster_basel = read_csv(
            self.data_path + 'Cluster_labels/Basel_metaclusters.csv',
            usecols=[full_cell_key_col, cluster_col],
            dtype={full_cell_key_col: str, cluster_col: str}
        )
        node_cluster_zurich = read_csv(
            self.data_path + 'Cluster_labels/Zurich_matched_metaclusters.csv',
            usecols=[full_cell_key_col, cluster_col],
            dtype={full_cell_key_col: str, cluster_col: str}
        )
        node_cluster = pd.concat([
            node_cluster_basel,
            node_cluster_zurich
        ], axis=0, ignore_index=True)

        return node_cluster

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        self._register_images()
        position_matrix = self._load_node_positions()
        node_features = self._load_node_features()
        node_types = self._load_node_types()

        node_types.set_index('id', inplace=True)
        position_matrix.set_index('id', inplace=True)
        celldata_df = pd.concat([position_matrix, node_features, node_types], axis=1, ignore_index=False, join='outer')
        celldata_df = celldata_df[celldata_df['x'] == celldata_df['x']]

        celldata_df['core'] = ['_'.join(a.split('_')[:-1]) for a in celldata_df.index]
        celldata_df['PID'] = [self.img_to_patient_dict[c] for c in celldata_df['core']]

        metadata = {
            "lateral_resolution": None,
            "fn": None,
            "image_col": "core",
            "pos_cols": ["x", "y"],
            "cluster_col": "cluster",
            "cluster_col_preprocessed": "cluster_preprocessed",
            "patient_col": "PID",
            "cell_type_coarseness": self.cell_type_coarseness,
        }

        feature_cols = [
            '1021522Tm169Di EGFR',
            '1031747Er167Di ECadhe',
            '112475Gd156Di Estroge',
            '117792Dy163Di GATA3',
            '1261726In113Di Histone',
            '1441101Er168Di Ki67',
            '174864Nd148Di SMA',
            '1921755Sm149Di Vimenti',
            '198883Yb176Di cleaved',
            '201487Eu151Di cerbB',
            '207736Tb159Di p53',
            '234832Lu175Di panCyto',
            '3111576Nd143Di Cytoker',
            'Nd145Di Twist',
            '312878Gd158Di Progest',
            '322787Nd150Di cMyc',
            '3281668Nd142Di Fibrone',
            '346876Sm147Di Keratin',
            '3521227Gd155Di Slug',
            '361077Dy164Di CD20',
            '378871Yb172Di vWF',
            '473968La139Di Histone',
            '651779Pr141Di Cytoker',
            '6967Gd160Di CD44',
            '71790Dy162Di CD45',
            '77877Nd146Di CD68',
            '8001752Sm152Di CD3epsi',
            '92964Er166Di Carboni',
            '971099Nd144Di Cytoker',
            '98922Yb174Di Cytoker',
            'phospho Histone',
            'phospho S6',
            'phospho mTOR',
            'Area'
        ]

        X = DataFrame(np.array(celldata_df[feature_cols]), columns=feature_cols)
        celldata = AnnData(X=X, obs=celldata_df[[metadata['image_col'], metadata['patient_col'], metadata['cluster_col']]])
        celldata.var_names_make_unique()

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): celldata_df[metadata["patient_col"]].values[i]
            for i, x in enumerate(celldata_df[metadata["image_col"]].values)
        }
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="category").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[
            metadata["cluster_col_preprocessed"]].astype(
            "category"
        )

        # register node type names
        types = celldata.obs[metadata["cluster_col_preprocessed"]]

        node_type_names = list(np.unique(types[types == types]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) if x == x else 0
                for x in types
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = types == types
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.
        Parameters
        ----------
        label_selection
            Label selection.
        """
        # DEFINE COLUMN NAMES FOR TABULAR DATA.
        # Define column names to extract from patient-wise tabular data:
        patient_col = 'PID'
        img_key_col = 'core'
        # These are required to assign the image to dieased and non-diseased:
        image_status_cols = ['location', 'diseasestatus']
        # Labels are defined as a column name and a label type:
        disease_features = {
            'grade': 'categorical',
            'grade_collapsed': 'categorical',
            'tumor_size': 'continuous',
            'diseasestatus': 'categorical',
            'location': 'categorical',
            'tumor_type': 'categorical'
        }
        patient_features = {
            'age': 'continuous'
        }
        survival_features = {
            'Patientstatus': 'categorical',
            'DFSmonth': 'survival',
            'OSmonth': 'survival'
        }
        tumor_featues = {
            'clinical_type': 'categorical',
            'Subtype': 'categorical',
            'PTNM_M': 'categorical',
            'PTNM_T': 'categorical',
            'PTNM_N': 'categorical',
            'PTNM_Radicality': 'categorical',
            'Lymphaticinvasion': 'categorical',
            'Venousinvasion': 'categorical',
            'ERStatus': 'categorical',
            'PRStatus': 'categorical',
            'HER2Status': 'categorical',
            # 'ER+DuctalCa': 'categorical',
            'TripleNegDuctal': 'categorical',
            # 'hormonesensitive': 'categorical',
            # 'hormoneresistantaftersenstive': 'categorical',
            'microinvasion': 'categorical',
            'I_plus_neg': 'categorical',
            'SN': 'categorical',
            # 'MIC': 'categorical'
        }
        treatment_feature = {
            'Pre-surgeryTx': 'categorical',
            'Post-surgeryTx': 'categorical'
        }
        batch_features = {
            'TMABlocklabel': 'categorical',
            'Yearofsamplecollection': 'continuous'
        }
        ncell_features = {  # not used right now
            '%tumorcells': 'percentage',
            '%normalepithelialcells': 'percentage',
            '%stroma': 'percentage',
            '%inflammatorycells': 'percentage',
            'Count_Cells': 'continuous'
        }
        label_cols = {}
        label_cols.update(disease_features)
        label_cols.update(patient_features)
        label_cols.update(survival_features)
        label_cols.update(tumor_featues)
        label_cols.update(treatment_feature)
        label_cols.update(batch_features)
        label_cols.update(ncell_features)
        # Clean selected labels based on defined labels:
        if label_selection is None:
            label_selection = set(label_cols.keys())
        else:
            label_selection = set(label_selection)
        label_cols_toread = list(label_selection.intersection(set(list(label_cols.keys()))))
        # Make sure censoring information is read if surival is predicted:
        if 'DFSmonth' in label_cols_toread:
            if 'OSmonth' not in label_cols_toread:
                label_cols_toread.append('OSmonth')
        if 'OSmonth' in label_cols_toread:
            if 'Patientstatus' not in label_cols_toread:
                label_cols_toread.append('Patientstatus')
        if 'grade_collapsed' in label_selection and 'grade' not in label_selection:
            label_cols_toread.append('grade')

        # READ RAW LABELS, COVARIATES AND IDENTIFIERS FROM TABLES.
        # Read all labels and image- and patient-identifiers from table. This full set is overlapped to the existing
        # columns of file so that files with different column spaces can be read.
        # The patients are renamed for each patient set with a prefix to guarantee uniqueness.
        # The output of this workflow is (1) a single table with rows for each image and with all columns modified
        # so that the can further be processed to tensors of labels and covariates through GLM formula-like commands and
        # (2) indices of diseased and non-diseased images in this table.
        cols_toread = [patient_col, img_key_col] + image_status_cols + label_cols_toread  # full list of columns to read
        # Read Basel data.
        cols_found_basel = read_csv(self.data_path + "Data_publication/BaselTMA/Basel_PatientMetadata.csv", nrows=0)
        cols_toread_basel = set(cols_found_basel.columns) & set(cols_toread)
        tissue_meta_data_basel = read_csv(
            self.data_path + "Data_publication/BaselTMA/Basel_PatientMetadata.csv",
            usecols=cols_toread_basel
        )
        tissue_meta_data_basel[patient_col] = ["b" + str(x) for x in tissue_meta_data_basel[patient_col].values]
        # Read Zuri data.
        cols_found_zuri = read_csv(self.data_path + "Data_publication/ZurichTMA/Zuri_PatientMetadata.csv", nrows=0)
        cols_toread_zuri = set(cols_found_zuri.columns) & set(cols_toread)
        tissue_meta_data_zuri = read_csv(
            self.data_path + "Data_publication/ZurichTMA/Zuri_PatientMetadata.csv",
            usecols=cols_toread_zuri
        )
        tissue_meta_data_zuri[patient_col] = ["z" + str(x) for x in tissue_meta_data_zuri[patient_col].values]

        # Modify specific columns:
        # The diseasestatus is not given in the Zuri data but can be inferred from the location column.
        tissue_meta_data_zuri['diseasestatus'] = [
            'tumor' if a else 'non-tumor' for a in tissue_meta_data_zuri['location'] != '[]'
        ]
        # Tumor size is masked if the image does not contain a tumor:
        if 'tumor_size' in label_selection:
            no_tumor = list(tissue_meta_data_basel['diseasestatus'] == 'non-tumor')
            tissue_meta_data_basel.loc[no_tumor, 'tumor_size'] = np.nan
        # Add missing Patientstatus and survival labels in Zuri data that are only given in Basel data set:
        if 'Patientstatus' in label_selection:
            tissue_meta_data_zuri['Patientstatus'] = np.nan
        if 'OSmonth' in label_selection:
            tissue_meta_data_zuri['OSmonth'] = np.nan
        if 'DFSmonth' in label_selection:
            tissue_meta_data_zuri['DFSmonth'] = np.nan
        # Add censoring column if survival is given:
        # All states recorded: alive, alive w metastases, death, death by primary disease
        # Also densor non-disease caused death.
        if 'OSmonth' in label_selection:
            tissue_meta_data_basel['censor_OS'] = [
                0 if x in ["alive", "alive w metastases"] else 1  # penalty-scale for over-estimation
                for x in tissue_meta_data_basel['Patientstatus'].values
            ]
            tissue_meta_data_zuri["censor_OS"] = np.nan

        if 'DFSmonth' in label_selection:
            tissue_meta_data_basel['censor_DFS'] = [
                0 if tissue_meta_data_basel['OSmonth'][idx] == tissue_meta_data_basel['DFSmonth'][idx] else 1
                for idx in tissue_meta_data_basel['OSmonth'].index
            ]
            tissue_meta_data_zuri["censor_DFS"] = np.nan

        # Replace missing observations labeled as "[]" for PTNM_N, PTNM_M, PTNM_T
        if 'PTNM_N' in label_selection:
            tissue_meta_data_zuri['PTNM_N'] = [a[1:] for a in tissue_meta_data_zuri['PTNM_N']]
            tissue_meta_data_zuri['PTNM_N'].replace(']', 'nan', inplace=True)
        if 'PTNM_M' in label_selection:
            tissue_meta_data_zuri['PTNM_M'] = [a[1:] for a in tissue_meta_data_zuri['PTNM_M']]
            tissue_meta_data_zuri['PTNM_M'].replace(']', 'nan', inplace=True)
        if 'PTNM_T' in label_selection:
            tissue_meta_data_zuri['PTNM_T'] = [a[1:] for a in tissue_meta_data_zuri['PTNM_T']]
            tissue_meta_data_zuri['PTNM_T'].replace(']', 'nan', inplace=True)

        # Merge Basel and Zuri data.
        tissue_meta_data = pd.concat([
            tissue_meta_data_basel,
            tissue_meta_data_zuri
        ], axis=0, sort=True, ignore_index=True)

        if "grade_collapsed" in label_selection:
            tissue_meta_data['grade_collapsed'] = ['3' if grade == '3' else '1&2' for grade in
                                                   tissue_meta_data['grade']]

        # Drop already excluded images (e.g. METASTASIS or to few nodes)
        tissue_meta_data = tissue_meta_data[tissue_meta_data[img_key_col].isin(list(self.img_to_patient_dict.keys()))].reset_index()

        # Final processing:
        # Remove columns that are only used to infer missing entries in other columns:
        if 'location' not in label_selection:
            tissue_meta_data.drop('location', 1, inplace=True)
        if 'grade_collapsed' in label_selection and 'grade' not in label_selection:
            tissue_meta_data.drop('grade', 1, inplace=True)
        # Some non-label columns remain in the table as these are used to build objects that subset images into groups,
        # these columns are removed below once their information is processed. These columns are, if they are not
        # among the chosen labels:
        # ["diseasestatus", patient_col]

        # Remove diseasestatus column that is only used to assign diseased and non-diseased index vectors from meta
        # data table:
        if 'diseasestatus' not in label_selection:
            tissue_meta_data.drop('diseasestatus', 1, inplace=True)

        # BUILD LABEL VECTORS FROM LABEL COLUMNS
        # The columns contain unprocessed numeric and categorical entries that are now processed to prediction-ready
        # numeric tensors. Here we first generate a dictionary of tensors for each label (label_tensors). We then
        # transform this to have as output of this section dictionary by image with a dictionary by labels as values
        # which can be easily queried by image in a data generator.
        # Subset labels and label types:
        label_cols = {label: type for label, type in label_cols.items() if label in label_selection}
        label_tensors = {}
        label_names = {}  # Names of individual variables in each label vector (eg. categories in onehot-encoding).
        # 1. Standardize continuous labels to z-scores:
        continuous_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == 'continuous'
        }
        continuous_std = {
            feature: tissue_meta_data[feature].std(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == 'continuous'
        }
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == 'continuous':
                label_tensors[feature] = (tissue_meta_data[feature].values - continuous_mean[feature]) / continuous_std[
                    feature]
                label_names[feature] = [feature]
        # 2. Scale percentages into [0, 1]
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == 'percentage':
                # Take "%" out of name if present
                feature_renamed = feature.replace('%', 'percentage_')
                label_cols = dict([(k, v) if k != feature else (feature_renamed, v) for k, v in label_cols.items()])
                label_tensors[feature_renamed] = tissue_meta_data[feature].values / 100.
                label_names[feature_renamed] = [feature_renamed]
        # 3. One-hot encode categorical columns
        # Force all entries in categorical columns to be string so that GLM-like formula processing can be performed.
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == 'categorical':
                tissue_meta_data[feature] = tissue_meta_data[feature].astype('str')
        # One-hot encode each string label vector:
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == 'categorical':
                oh = pd.get_dummies(
                    tissue_meta_data[feature],
                    prefix=feature,
                    prefix_sep='>',
                    drop_first=False
                )
                # Change all entries of corresponding observation to np.nan instead.
                idx_nan_col = np.array([i for i, x in enumerate(oh.columns) if x.endswith('>nan')])
                if len(idx_nan_col) > 0:
                    assert len(idx_nan_col) == 1, "fatal processing error"
                    nan_rows = np.where(oh.iloc[:, idx_nan_col[0]].values == 1.)[0]
                    oh.loc[nan_rows, :] = np.nan
                # Drop nan element column.
                oh = oh.loc[:, [x for x in oh.columns if not x.endswith('>nan')]]
                label_tensors[feature] = oh.values
                label_names[feature] = oh.columns
        # 4. Add censoring information to survival
        survival_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == 'survival'
        }
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == 'survival':
                if feature == 'DFSmonth':
                    censor_col = 'censor_DFS'
                if feature == 'OSmonth':
                    censor_col = 'censor_OS'
                label_tensors[feature] = np.concatenate([
                    np.expand_dims(tissue_meta_data[feature].values / survival_mean[feature], axis=1),
                    np.expand_dims(tissue_meta_data[censor_col].values, axis=1),
                ], axis=1)
                label_names[feature] = [feature]
        # Make sure all tensors are 2D for indexing:
        for feature in list(label_tensors.keys()):
            if len(label_tensors[feature].shape) == 1:
                label_tensors[feature] = np.expand_dims(label_tensors[feature], axis=1)
        # The dictionary of tensor is nested in slices in a dictionary by image which is easier to query with a
        # generator.
        images = tissue_meta_data[img_key_col].values.tolist()
        assert np.all([len(list(self.img_to_patient_dict.keys())) == x.shape[0] for x in list(label_tensors.values())]), \
            "fatal processing error"
        label_tensors = {
            img: {
                kk: np.array(vv[images.index(img), :], ndmin=1)
                for kk, vv in label_tensors.items()  # iterate over labels
            } for img in self.img_to_patient_dict.keys()  # iterate over images
        }

        # Save processed data to attributes.
        for k, adata in self.img_celldata.items():
            graph_covariates = {
                "label_names": label_names,
                "label_tensors": label_tensors[k],
                "label_selection": list(label_cols.keys()),
                "continuous_mean": continuous_mean,
                "continuous_std": continuous_std,
                "label_data_types": label_cols,
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": label_names,
            "label_selection": list(label_cols.keys()),
            "continuous_mean": continuous_mean,
            "continuous_std": continuous_std,
            "label_data_types": label_cols,
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DataLoaderIonpath(DataLoader):
    """DataLoaderIonpath class. Inherits all functions from DataLoader."""

    cell_type_merge_dict = {  # from shareCellData/readme.rtf
        'fine': {
            "Group_1": "Unidentified",
            "Group_3": "Endothelial",
            "Group_4": "Mesenchymal",
            "Group_5": "Tumor",
            "Group_6": "Keratin-positive tumor",
            "immuneGroup_1": "Tregs",
            "immuneGroup_2": "CD4 T",
            "immuneGroup_3": "CD8 T",
            "immuneGroup_4": "CD3 T",
            "immuneGroup_5": "NK",
            "immuneGroup_6": "B",
            "immuneGroup_7": "Neutrophils",
            "immuneGroup_8": "Macrophages",
            "immuneGroup_9": "DC",
            "immuneGroup_10": "DC-Mono",
            "immuneGroup_11": "Mono-Neu",
            "immuneGroup_12": "Other immune"
        }
    }

    def _register_images(self):
        sample_col = "SampleID"
        usecols = [sample_col]
        # Define mapping of image identifiers to file names:
        node_tab = read_csv(
            self.data_path + 'shareCellData/cellData.csv',
            usecols=usecols, dtype={sample_col: str}
        ).drop_duplicates([sample_col])
        node_tab['filename'] = node_tab['SampleID'].apply(
            lambda x: 'p' + str(x).replace(',', '') + '_labeledcellData.tiff'
        )
        image_tab = node_tab[[sample_col, 'filename']].drop_duplicates([sample_col, 'filename'])
        self.img_key_to_fn = dict(image_tab[[sample_col, 'filename']].values)

    def _load_node_positions(self):
        from PIL import Image

        position_matrix = []
        for k, fn in self.img_key_to_fn.items():
            fn = self.data_path + "shareCellData/" + fn
            if not os.path.exists(fn):
                raise ValueError("file %s not found" % fn)
            # Load image from tiff:
            img_array = np.array(Image.open(fn))
            # Throughout all files, nodes are referred to via the string k+"_"+str(i) where i is the integer
            # encoding the object in the segmentation mask.
            node_ids = np.sort(np.unique(img_array))
            # 0 encodes background:
            node_ids = node_ids[node_ids != 0]
            # Assert all registered node identifiers occur in image:
            if np.any([x not in node_ids for x in node_ids]):
                raise ValueError(
                    "%i out of %i registered nodes for image %s were not found in mask" %
                    (np.sum([x not in node_ids for x in node_ids]), len(node_ids), k)
                )
            # Find centre of object mask of each node:
            center_array = [np.where(img_array == i) for i in node_ids]
            pm = np.array([[f'{k}_{i}', x[0].mean(), x[1].mean()] for i, x in zip(node_ids, center_array)])
            position_matrix.append(pm)
        position_matrix = np.concatenate(position_matrix, axis=0)
        position_matrix = pd.DataFrame(position_matrix, columns=['CellID', 'x', 'y'])
        return position_matrix

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": None,
            "fn": "shareCellData/cellData.csv",
            "image_col": "SampleID",
            "pos_cols": ["x", "y"],
            "cluster_col": "cluster",
            "cluster_col_preprocessed": "cluster_preprocessed",
            "patient_col": "patient",
            "cell_type_coarseness": self.cell_type_coarseness,
        }

        self._register_images()
        position_matrix = self._load_node_positions()
        position_matrix.set_index('CellID', inplace=True)

        celldata_df = read_csv(os.path.join(self.data_path, metadata["fn"]))
        celldata_df['CellID'] = celldata_df['SampleID'].astype(str) + '_' + celldata_df['cellLabelInImage'].astype(str)
        celldata_df[metadata['image_col']] = celldata_df[metadata['image_col']].astype(str)
        celldata_df['patient'] = celldata_df[metadata['image_col']]
        celldata_df.set_index('CellID', inplace=True)

        cluster_cols = ["immuneGroup", "Group"]
        celldata_df["cluster"] = [
            cluster_cols[0] + "_" + str(celldata_df[cluster_cols[0]].values[i]) if celldata_df[cluster_cols[0]].values[i] != "0"
            else cluster_cols[1] + "_" + str(celldata_df[cluster_cols[1]].values[i])
            for i in range(celldata_df.shape[0])
        ]

        celldata_df = pd.concat([celldata_df, position_matrix], axis=1, ignore_index=False, join='inner')

        feature_cols = [
            "cellSize",
            "Vimentin",
            "SMA",
            "Background",
            "B7H3",
            "FoxP3",
            "Lag3",
            "CD4",
            "CD16",
            "CD56",
            "OX40",
            "PD1",
            "CD31",
            "PD-L1",
            "EGFR",
            "Ki67",
            "CD209",
            "CD11c",
            "CD138",
            "CD163",
            "CD68",
            "CSF-1R",
            "CD8",
            "CD3",
            "IDO",
            "Keratin17",
            "CD63",
            "CD45RO",
            "CD20",
            "p53",
            "Beta catenin",
            "HLA-DR",
            "CD11b",
            "CD45",
            "H3K9ac",
            "Pan-Keratin",
            "H3K27me3",
            "phospho-S6",
            "MPO",
            "Keratin6",
            "HLA_Class_1"
        ]
        X = DataFrame(np.array(celldata_df[feature_cols]), columns=feature_cols)
        celldata = AnnData(X=X, obs=celldata_df[[metadata['image_col'], metadata['patient_col'], metadata['cluster_col']]])
        celldata.var_names_make_unique()

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): celldata_df[metadata["patient_col"]].values[i]
            for i, x in enumerate(celldata_df[metadata["image_col"]].values)
        }
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="category").map(self.cell_type_merge_dict[self.cell_type_coarseness])
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[metadata["cluster_col_preprocessed"]].astype("category")

        # register node type names
        types = celldata.obs[metadata["cluster_col_preprocessed"]]

        node_type_names = list(np.unique(types[types == types]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x) if x == x else 0
                for x in types
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = types == types
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.
        Parameters
        ----------
        label_selection
            Label selection.
        """
        # DEFINE COLUMN NAMES FOR TABULAR DATA.
        # Define column names to extract from patient-wise tabular data:
        img_key_col = 'InternalId'
        patient_col = "patient"  # is added
        censor_col = "survival_censor"  # will be added if survival is handled
        # Labels are defined as a column name and a label type:
        disease_features = {
            'STAGE': 'categorical',
            'GRADE': 'categorical'
        }
        patient_features = {
            'age': 'continuous'
        }
        survival_features = {
            'Censored': 'categorical',
            'Survival_days_capped*': 'survival'
        }
        tumor_featues = {
            'ER': 'categorical',
            'PR': 'categorical',
            'HER2NEU': 'categorical',
            'CS_TUM_SIZE': 'categorical',
            'RECURRENCE_LABEL': 'categorical'
        }
        batch_features = {
            'YEAR': 'categorical'
        }
        label_cols = {}
        label_cols.update(disease_features)
        label_cols.update(patient_features)
        label_cols.update(survival_features)
        label_cols.update(tumor_featues)
        label_cols.update(batch_features)
        # Clean selected labels based on defined labels:
        if label_selection is None:
            label_selection = set(label_cols.keys())
        else:
            label_selection = set(label_selection)
        label_cols_toread = list(label_selection.intersection(set(list(label_cols.keys()))))
        # Make sure censoring information is read if surival is predicted:
        if 'Survival_days_capped*' in label_cols_toread or 'Survival_days_capped*' in label_cols_toread:
            if 'Censored' not in label_cols_toread:
                label_cols_toread.append('Censored')

        # READ RAW LABELS, COVARIATES AND IDENTIFIERS FROM TABLES.
        # Read all labels and image- and patient-identifiers from table. This full set is overlapped to the existing
        # columns of file so that files with different column spaces can be read.
        # The patients are renamed for each patient set with a prefix to guarantee uniqueness.
        # The output of this workflow is (1) a single table with rows for each image and with all columns modified
        # so that the can further be processed to tensors of labels and covariates through GLM formula-like commands and
        # (2) indices of diseased and non-diseased images in this table.
        cols_toread = [img_key_col] + label_cols_toread  # full list of columns to read
        tissue_meta_data = read_csv(
            self.data_path + "1-s2.0-S0092867418311000-mmc2.csv",
            usecols=cols_toread, sep=";"
        )
        tissue_meta_data = tissue_meta_data.iloc[:-3, :].copy()  # drop last rows which are empty

        # Modify specific columns:
        # Add censoring column if survival is given:
        # All states recorded: alive, alive w metastases, death, death by primary disease
        # Also densor non-disease caused death.
        if 'Survival_days_capped*' in label_selection:
            tissue_meta_data[censor_col] = [
                0 if x == 1 else 1  # penalty-scale for over-estimation
                for x in tissue_meta_data['Censored'].values
            ]

        # GROUP IMAGES BY PATIENTS
        # The output of this section is (1) a map from numeric image identifiers to patients (img_to_patient_dict) and
        # (2) a map of diseased images to the set of corresponding healthy images (nondiseased_ref_dict).
        img_to_patient_dict = {
            x: x for x in tissue_meta_data[img_key_col].values
        }

        # BUILD LABEL VECTORS FROM LABEL COLUMNS
        # The columns contain unprocessed numeric and categorical entries that are now processed to prediction-ready
        # numeric tensors. Here we first generate a dictionary of tensors for each label (label_tensors). We then
        # transform this to have as output of this section dictionary by image with a dictionary by labels as values
        # which can be easily queried by image in a data generator.
        # Subset labels and label types:
        label_cols = {label: type for label, type in label_cols.items() if label in label_selection}
        label_tensors = {}
        label_names = {}  # Names of individual variables in each label vector (eg. categories in onehot-encoding).
        # 1. Standardize continuous labels to z-scores:
        continuous_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == 'continuous'
        }
        continuous_std = {
            feature: tissue_meta_data[feature].std(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == 'continuous'
        }
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == 'continuous':
                label_tensors[feature] = (tissue_meta_data[feature].values - continuous_mean[feature]) / continuous_std[
                    feature]
                label_names[feature] = [feature]
        # 2. Scale percentages into [0, 1]
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == 'percentage':
                # Take "%" out of name if present
                feature_renamed = feature.replace('%', 'percentage_')
                label_cols = dict([(k, v) if k != feature else (feature_renamed, v) for k, v in label_cols.items()])
                label_tensors[feature_renamed] = tissue_meta_data[feature].values / 100.
                label_names[feature_renamed] = [feature_renamed]
        # 3. One-hot encode categorical columns
        # Force all entries in categorical columns to be string so that GLM-like formula processing can be performed.
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == 'categorical':
                tissue_meta_data[feature] = tissue_meta_data[feature].astype('str')
        # One-hot encode each string label vector:
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == 'categorical':
                oh = pd.get_dummies(
                    tissue_meta_data[feature],
                    prefix=feature,
                    prefix_sep='>',
                    drop_first=False
                )
                # Change all entries of corresponding observation to np.nan instead.
                idx_nan_col = np.array([i for i, x in enumerate(oh.columns) if x.endswith('>nan')])
                if len(idx_nan_col) > 0:
                    assert len(idx_nan_col) == 1, "fatal processing error"
                    nan_rows = np.where(oh.iloc[:, idx_nan_col[0]].values == 1.)[0]
                    oh.loc[nan_rows, :] = np.nan
                # Drop nan element column.
                oh = oh.loc[:, [x for x in oh.columns if not x.endswith('>nan')]]
                label_tensors[feature] = oh.values
                label_names[feature] = oh.columns
        # 4. Add censoring information to survival
        survival_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == 'survival'
        }
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == 'survival':
                label_tensors[feature] = np.concatenate([
                    np.expand_dims(tissue_meta_data[feature].values / survival_mean[feature], axis=1),
                    np.expand_dims(tissue_meta_data[censor_col].values, axis=1),
                ], axis=1)
                label_names[feature] = [feature]
        # Make sure all tensors are 2D for indexing:
        for feature in list(label_tensors.keys()):
            if len(label_tensors[feature].shape) == 1:
                label_tensors[feature] = np.expand_dims(label_tensors[feature], axis=1)
        # The dictionary of tensor is nested in slices in a dictionary by image which is easier to query with a
        # generator.
        assert np.all([len(list(img_to_patient_dict.keys())) == x.shape[0] for x in list(label_tensors.values())]), \
            "fatal processing error"
        label_tensors = {
            k: {
                kk: np.array(vv[i, :], ndmin=1) for kk, vv in label_tensors.items()  # iterate over labels
            } for i, k in enumerate(list(img_to_patient_dict.keys()))  # iterate over images
        }

        # Save processed data to attributes.
        for k, adata in self.img_celldata.items():
            graph_covariates = {
                "label_names": label_names,
                "label_tensors": label_tensors[k],
                "label_selection": list(label_cols.keys()),
                "continuous_mean": continuous_mean,
                "continuous_std": continuous_std,
                "label_data_types": label_cols,
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": label_names,
            "label_selection": list(label_cols.keys()),
            "continuous_mean": continuous_mean,
            "continuous_std": continuous_std,
            "label_data_types": label_cols,
        }
        self.celldata.uns["graph_covariates"] = graph_covariates
