import abc
from typing import Dict, List, Tuple, Union, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from anndata import AnnData, read_h5ad
from matplotlib.ticker import FormatStrFormatter
from matplotlib.tri import Triangulation
from pandas import read_csv, read_excel

# ToDo add graph covariates

class GraphTools:
    celldata: AnnData
    img_celldata: Dict[str, AnnData]

    def compute_adjacency_matrices(self, radius: int, transform: str = None):
        """

        :param radius:
        :param transform:
        :return:
        """
        for k, adata in self.img_celldata.items():
            sq.gr.spatial_neighbors(adata=adata, radius=radius, transform=transform, key_added="adjacency_matrix")

    def _get_degrees(self, max_distances: list):
        # ToDo this is not efficient with squidpy distances. Keep old implementation from tissue package
        degs = {}
        degrees = {}
        for k, adata in self.img_celldata.items():
            dist_matrix = adata.obsp["adjacency_matrix_distances"]
            degs[k] = {dist: np.sum(dist_matrix < dist * dist, axis=0) for dist in max_distances}
        for dist in max_distances:
            degrees[dist] = [deg[dist] for deg in degs.values()]
        return degrees

    def plot_degree_vs_dist(
        self,
        degree_matrices=None,
        max_distances=None,
        lateral_resolution: float = 1.0,
        save: Union[str, None] = None,
        suffix: str = "_degree_vs_dist.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        """

        :param degree_matrices:
        :param max_distances:
        :param lateral_resolution:
        :param save: Whether (if not None) and where (path as string given as save) to save plot.
        :param suffix: Suffix of file name to save to.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :return:

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
    def plot_noise_structure(
        self,
        undefined_type: Optional[str] = None,
        merge_types: Optional[Tuple[list, list]] = None,
        min_x: Optional[float] = None,
        max_x: Optional[float] = None,
        panel_width: float = 2.0,
        panel_height: float = 2.7,
        save: Optional[str] = None,
        suffix: str = "_noise_structure.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        feature_mat = pd.concat(
            [
                pd.concat(
                    [
                        pd.DataFrame(
                            {
                                "image": [k for i in range(adata.shape[0])],
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
        if undefined_type is not None:
            feature_mat = feature_mat[feature_mat["cell_type"] != undefined_type]

        if merge_types is not None:
            for mt in merge_types[0]:
                feature_mat = feature_mat.replace(mt, merge_types[-1])

        plt.ioff()
        ct = np.unique(feature_mat["cell_type"].values)
        nrows = len(ct) // 12 + int(len(ct) % 12 > 0)
        fig, ax = plt.subplots(
            ncols=12, nrows=nrows, figsize=(12 * panel_width, nrows * panel_height), sharex="all", sharey="all"
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
            ax[i].set_title(ci.replace("_", "\n").replace(" ", "\n").replace("/", "\n"), fontsize=14)
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

    def plot_umap(
        self,
        image_key: str,
        target_cell_type: Optional[str] = None,
        n_neighbors: Optional[int] = 10,
        n_pcs: Optional[int] = None,
        panel_width: float = 5.,
        panel_height: float = 5.,
        front_size: int = 20,
        size: Optional[int] = None,
        palette: Optional[str] = None,
        save: Union[str, None] = None,
        suffix: str = "_umap.pdf",
        show: bool = True,
        copy: bool = True,
    ):
        temp_adata = self.img_celldata[image_key].copy()
        cluster_id = temp_adata.uns['metadata']['cluster_col']
        if target_cell_type:
            temp_adata = temp_adata[temp_adata.obs[cluster_id] == target_cell_type]
        sc.pp.neighbors(temp_adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.louvain(temp_adata)
        sc.tl.umap(temp_adata)
        print('n cells: ', temp_adata.shape[0])
        if target_cell_type:
            temp_adata.obs[f"{target_cell_type}_sub-states"] = target_cell_type + temp_adata.obs.louvain.astype(str)
            print(temp_adata.obs[f"{target_cell_type}_sub-states"].value_counts())
            color = [f"{target_cell_type}_sub-states"]
        else:
            color = [cluster_id]

        plt.ioff()
        sc.set_figure_params(scanpy=True, fontsize=front_size)
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(panel_width, panel_height), )
        sc.pl.umap(
            temp_adata,
            color=color,
            ax=ax,
            show=False,
            size=size,
            palette=palette,
            title='')
        # Save, show and return figure.
        if save is not None:
            plt.savefig(save + image_key + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if copy:
            return temp_adata

    def plot_spatial(
        self,
        image_key: str,
        undefined_type: Optional[str] = None,
        panel_width: float = 7.,
        panel_height: float = 7.,
        front_size: int = 18,
        spot_size: int = 30,
        legend_loc: str = 'left',
        save: Union[str, None] = None,
        suffix: str = "_spatial.pdf",
        clean_view: bool = False,
        show: bool = True,
        copy: bool = True,
    ):
        temp_adata = self.img_celldata[image_key].copy()
        cluster_id = temp_adata.uns['metadata']['cluster_col']
        if undefined_type:
            temp_adata = temp_adata[temp_adata.obs[cluster_id] != undefined_type]

        if clean_view:
            temp_adata = temp_adata[np.argwhere(np.array(temp_adata.obsm['spatial'])[:, 1] < 0).squeeze()]

        plt.ioff()
        sc.set_figure_params(scanpy=True, fontsize=front_size)
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(panel_width, panel_height))
        sc.pl.spatial(
            temp_adata,
            color=cluster_id,
            spot_size=spot_size,
            legend_loc=legend_loc,
            ax=ax,
            show=False,
            title=''
        )
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

    def cluster_enrichment(
        self,
        image_key: str,
        target_cell_type: str,
        undefined_type: Optional[str] = None,
        filter_titles: Optional[List[str]] = None,
        n_neighbors: Optional[int] = 40,
        n_pcs: Optional[int] = 40,
        clip_pvalues: Optional[int] = -5,
    ):
        sorce_type_names = [f"source type {x.replace('_', ' ')}" for x in list(self.celldata.uns['node_type_names'].values())]
        h_1 = []
        h_0 = []
        fov = []
        a = []
        pm = []
        img_size = []
        for k, adata in self.img_celldata.items():
            img_size.append(adata.shape[0])
            h_1.append(adata.X)
            fov.append(np.expand_dims(np.repeat(k, adata.shape[0]), axis=1))
            h_0.append(adata.obsm['node_types'])
            pm.append(adata.obsm['spatial'])
            a.append(np.array(adata.obsp['adjacency_matrix_connectivities'].todense()))

        source_type = []
        for i, adj in enumerate(a):
            adj = np.asarray(adj > 0, dtype="int")
            source_type.append(
                np.matmul(adj, h_0[i].astype(int))
            )

        h_1 = pd.DataFrame(np.concatenate(h_1, axis=0), columns=self.celldata.var_names)
        fov = pd.DataFrame(np.concatenate(fov, axis=0), columns=['fov'])
        source_type = pd.DataFrame(
            (np.concatenate(source_type, axis=0) > 0).astype(str), columns=sorce_type_names
        ).replace({'True': 'in neighbourhood', 'False': 'not in neighbourhood'}, regex=True)
        h_0 = pd.DataFrame(
            np.concatenate(h_0, axis=0),
            columns=[x.replace('_', ' ') for x in list(self.celldata.uns['node_type_names'].values())]
        )
        position_matrix = np.concatenate(pm, axis=0)
        target_type = pd.DataFrame(np.array(h_0.idxmax(axis=1)), columns=['target_cell'], dtype="category")

        metadata = pd.concat(
            [fov, target_type, source_type],
            axis=1
        )

        adata = AnnData(X=h_1, obs=metadata)
        adata.obsm["spatial"] = position_matrix
        if undefined_type:
            adata = adata[adata.obs['target_cell'] != undefined_type]

        adata_substates = adata[
            (adata.obs['target_cell'] == target_cell_type) & (adata.obs['fov'] == image_key)
            ]
        sc.pp.neighbors(adata_substates, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.louvain(adata_substates)
        sc.tl.umap(adata_substates)

        print('n cells: ', adata_substates.shape[0])
        adata_substates.obs[
            f"{target_cell_type}_sub-states"] = f"{target_cell_type} " + adata_substates.obs.louvain.astype(
            str)
        print(adata_substates.obs[f"{target_cell_type}_sub-states"].value_counts())

        one_hot = pd.get_dummies(adata_substates.obs.louvain, dtype=np.bool)
        # Join the encoded df
        df = adata_substates.obs.join(one_hot)

        import scipy.stats as stats
        from diffxpy.testing.correction import correct

        titles = list(self.celldata.uns['node_type_names'].values())
        distinct_louvain = len(np.unique(adata_substates.obs.louvain))
        pval_source_type = []
        for i, st in enumerate(titles):
            pval_cluster = []
            for j in range(distinct_louvain):
                crosstab = np.array(pd.crosstab(df[f"source type {st}"], df[str(j)]))
                if crosstab.shape[0] < 2:
                    crosstab = np.vstack([crosstab, [0, 0]])
                oddsratio, pvalue = stats.fisher_exact(crosstab)
                pvalue = correct(np.array([pvalue]))
                pval_cluster.append(pvalue)
            pval_source_type.append(pval_cluster)

        columns = [f"{target_cell_type} {x}" for x in np.unique(adata_substates.obs.louvain)]
        pval = pd.DataFrame(
            np.array(pval_source_type).squeeze(),
            index=[x.replace('_', ' ') for x in titles],
            columns=columns
        )
        log_pval = np.log10(pval)

        if filter_titles:
            log_pval = log_pval.sort_values(columns, ascending=True).filter(
                items=filter_titles,
                axis=0
            )
        if clip_pvalues:
            log_pval[log_pval < clip_pvalues] = clip_pvalues
        fold_change_df = adata_substates.obs[
            ['target_cell', f"{target_cell_type}_sub-states"] + sorce_type_names
            ]
        counts = pd.pivot_table(
            fold_change_df.replace({'in neighbourhood': 1, 'not in neighbourhood': 0}),
            index=[f"{target_cell_type}_sub-states"],
            aggfunc=np.sum,
            margins=True
        ).T
        counts['new_index'] = [x.replace('source type ', '') for x in counts.index]
        counts = counts.set_index('new_index')

        fold_change = counts.loc[:, columns].div(counts["All"], axis=0)
        fold_change = fold_change.subtract(np.array(counts['All'] / np.sum(counts['All'])), axis=0)

        if filter_titles:
            fold_change = fold_change.fillna(0).filter(
                items=filter_titles,
                axis=0
            )
        return adata_substates, log_pval, fold_change

    def plot_cluster_enrichment(
        self,
        pvalues,
        fold_change,
        panel_width: float = 2.5,
        panel_height: float = 10.,
        frontsize: int = 14,
        save: Union[str, None] = None,
        suffix: str = "_cluster_enrichment.pdf",
        show: bool = True,
    ):

        class MidpointNormalize(colors.Normalize):
            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                colors.Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))

        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(panel_width, panel_height), )
        sc.set_figure_params(scanpy=True, fontsize=frontsize)
        M = pvalues.shape[1]
        N = pvalues.shape[0]
        y = np.arange(N + 1)
        x = np.arange(M + 1)
        xs, ys = np.meshgrid(x, y)

        triangles1 = [(i + j * (M + 1), i + 1 + j * (M + 1), i + (j + 1) * (M + 1)) for j in range(N) for i in range(M)]
        triangles2 = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1)) for j in range(N) for i in
                      range(M)]
        triang1 = Triangulation(xs.ravel() - 0.5, ys.ravel() - 0.5, triangles1)
        triang2 = Triangulation(xs.ravel() - 0.5, ys.ravel() - 0.5, triangles2)
        img1 = plt.tripcolor(
            triang1,
            np.array(pvalues).ravel(),
            cmap=plt.get_cmap('Greys_r'),
        )
        img2 = plt.tripcolor(
            triang2,
            np.array(fold_change).ravel(),
            cmap=plt.get_cmap('seismic'),
            norm=MidpointNormalize(midpoint=0.)
        )

        # v1 = np.linspace(0.08, 0.22, 3, endpoint=True)
        cbar = plt.colorbar(
            img2,
            # ticks=v1,
            pad=0.2, orientation="horizontal",
        ).set_label(f"fold change")
        # cbar.set_yticklabels(["{:4.2f}".format(i) for i in v1]) # add the labels
        plt.colorbar(img1, ).set_label("$\log_{10}$ FDR-corrected pvalues")
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

    def plot_cluster_enrichment_umaps(
        self,
        adata,
        filter_titles,
        save: Union[str, None] = None,
        suffix: str = "_cluster_enrichment_umaps.pdf",
        show: bool = True,
    ):
        for i, x in enumerate(filter_titles):
            adata.uns[f"source type {x}_colors"] = ['darkgreen', 'lightgrey']
        sc.set_figure_params(scanpy=True, fontsize=24)
        plt.ioff()
        fig, axs = plt.subplots(
            nrows=4, ncols=5, figsize=(6 * 3, 3 * 4), )
        fig.supxlabel("UMAP2")
        fig.supylabel("UMAP1")
        N = len(filter_titles)
        #ax = trim_axs(ax, 25)
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        ax = axs[:N]

        for i, x in enumerate(filter_titles[:-1]):
            sc.pl.umap(
                adata,
                color=f"source type {x}",
                title=x,
                show=False,
                legend_loc='None',
                ax=ax[i]
            )
            ax[i].set_xlabel('')
            ax[i].set_ylabel('')
        sc.pl.umap(
            adata,
            color=f"source type {filter_titles[-1]}",
            title=filter_titles[-1],
            show=False,
            ax=ax[N-1]
        )
        ax[N-1].set_xlabel('')
        ax[N-1].set_ylabel('')
        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

    def plot_spatial_substates(
        self,
        adata_substates: AnnData,
        image_key: str,
        target_cell_type: str,
        clean_view: bool = False,
        panel_width: float = 7.,
        panel_height: float = 7.,
        spot_size: Optional[int] = 40,
        save: Union[str, None] = None,
        suffix: str = "_spatial_substates.pdf",
        show: bool = True,
        copy: bool = False
    ):
        temp_adata = self.img_celldata[image_key].copy()
        cluster_id = temp_adata.uns['metadata']['cluster_col']
        if clean_view:
            temp_adata = temp_adata[np.argwhere(np.array(temp_adata.obsm['spatial'])[:, 1] < 0).squeeze()]
            adata_substates = adata_substates[np.argwhere(np.array(adata_substates.obsm['spatial'])[:, 1] < 0).squeeze()]
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(panel_width, panel_height), )
        sc.set_figure_params(scanpy=True, fontsize=18)
        sc.pl.spatial(
            # adata,
            temp_adata[temp_adata.obs[cluster_id] != target_cell_type],
            spot_size=spot_size,
            ax=ax,
            show=False,
            na_color='whitesmoke',
            title=''
        )
        sc.pl.spatial(
            adata_substates,
            color=f"{target_cell_type}sub-states",
            spot_size=spot_size,
            ax=ax,
            show=False,
            legend_loc='left',
            title='',
            palette='tab10'
        )
        ax.invert_yaxis()
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

    def plot_ligrec(
        self,
        image_key: Optional[str] = None,
        source_groups: Optional[Union[str, Sequence[str]]] = None,
        undefined_type: Optional[str] = None,
        fraction: Optional[float] = None,
        pvalue_threshold: float = 0.3,
        width: float = 3.,
        seed: int = 10,
        random_state: int = 0,
        front_size: int = 18,
        save: Union[str, None] = None,
        suffix: str = "_ligrec.pdf",
        show: bool = True,
        copy: bool = True
    ):
        from omnipath.interactions import import_intercell_network
        interactions = import_intercell_network(
            transmitter_params={"categories": "ligand"},
            receiver_params={"categories": "receptor"}
        )
        if 'source' in interactions.columns:
            interactions.pop('source')
        if 'target' in interactions.columns:
            interactions.pop('target')
        interactions.rename(
            columns={"genesymbol_intercell_source": 'source', "genesymbol_intercell_target": 'target'}, inplace=True
        )

        if image_key:
            temp_adata = self.img_celldata[image_key]
        else:
            temp_adata = sc.pp.subsample(self.celldata, fraction=fraction, copy=True, random_state=random_state)

        cluster_id = temp_adata.uns['metadata']['cluster_col']
        if undefined_type:
            temp_adata = temp_adata[temp_adata.obs[cluster_id] != undefined_type]

        temp_adata = temp_adata.copy()
        print(temp_adata)

        sq.gr.ligrec(
            temp_adata,
            interactions=interactions,
            cluster_key=cluster_id,
            use_raw=False,
            seed=seed
        )
        if save is not None:
            save = save + image_key + suffix

        sc.set_figure_params(scanpy=True, fontsize=front_size)
        sq.pl.ligrec(
            temp_adata,
            cluster_key=cluster_id,
            title='',
            source_groups=source_groups,
            pvalue_threshold=pvalue_threshold,
            width=width,
            save=save
        )
        if show:
            plt.show()

        plt.close()
        plt.ion()

        if copy:
            return temp_adata

    def plot_ligrec_barplot(
        self,
        adata: AnnData,
        source_group: str,
        panel_width: float = 8.,
        panel_height: float = 4.,
        front_size: int = 18,
        pvalue_threshold: float = 0.05,
        save: Union[str, None] = None,
        suffix: str = "_ligrec_barplot.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        cluster_id = adata.uns['metadata']['cluster_col']
        pvals = adata.uns[f"{cluster_id}_ligrec"]['pvalues'].xs((source_group), axis=1)

        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(panel_width, panel_height), )
        sc.set_figure_params(scanpy=True, fontsize=front_size)
        sns.barplot(
            x=list(np.sum(pvals < pvalue_threshold, axis=0).index),
            y=list(np.sum(pvals < pvalue_threshold, axis=0)),
            ax=ax,
            color='steelblue'
        )
        ax.grid(False)
        ax.tick_params(axis='x', labelrotation=90)

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


class DataLoader(GraphTools, PlottingTools):
    def __init__(
        self,
        data_path: str,
        radius: int,
        label_selection: Union[List[str], None] = None,
    ):
        self.data_path = data_path

        print("Loading data from raw files")
        self.register_celldata()
        self.register_img_celldata()
        self.register_graph_features(label_selection=label_selection)
        self.compute_adjacency_matrices(radius=radius)
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
        return np.unique(np.asarray(list(self.celldata.uns["img_to_patient_dict"].values())))

    def register_celldata(self):
        """
        Loads anndata object of complete dataset.
        :return:
        """
        print("registering celldata")
        self._register_celldata()
        assert self.celldata is not None, "celldata was not loaded"

    def register_img_celldata(self):
        """
        Loads dictionary of of image-wise celldata objects with {imgage key : anndata object of image}.
        :return:
        """
        print("collecting image-wise celldata")
        self._register_img_celldata()
        assert self.img_celldata is not None, "image-wise celldata was not loaded"

    def register_graph_features(self, label_selection):
        print("adding graph-level covariates")
        self._register_graph_features(label_selection=label_selection)

    @abc.abstractmethod
    def _register_celldata(self):
        pass

    @abc.abstractmethod
    def _register_img_celldata(self):
        pass

    @abc.abstractmethod
    def _register_graph_features(self, label_selection):
        pass

    def merge_types(self, cell_type_mapping_dict: Dict[str, str]):
        """

        :param cell_type_mapping_dict: dictionary specifying cell type merge logic
        :return:
        """
        key_to_pos = {key: pos for pos, key in enumerate(np.sort(list(self.celldata.uns["node_type_names"].keys())))}
        new_types = np.sort(np.unique(list(cell_type_mapping_dict.values())))
        positions = {}
        for t in new_types:
            keys = [
                idx for idx, name in self.celldata.uns["node_type_names"].items() if cell_type_mapping_dict[idx] == t
            ]
            positions[t] = [key_to_pos[key] for key in keys]
        new_node_types = np.concatenate(
            [np.sum(self.celldata.obsm["node_types"][:, positions[t]], axis=1, keepdims=True) for t in new_types],
            axis=1,
        )
        self.celldata.obsm["node_types"] = new_node_types
        self.celldata.uns["node_type_names"] = {name: name for i, name in enumerate(new_types)}

        for key, adata in self.img_celldata.items():
            new_node_types = np.concatenate(
                [np.sum(adata.obsm["node_types"][:, positions[t]], axis=1, keepdims=True) for t in new_types], axis=1
            )

            adata.obsm["node_types"] = new_node_types
            adata.uns["node_type_names"] = {name: name for i, name in enumerate(new_types)}

    def size_factors(self):
        """
        Get size factors. Only makes sense with positive input.

        :return: Dict[str, np.ndarray] dictionary of size factors
        """
        # Check if irregular sums are encountered:
        for i, adata in self.img_celldata.items():
            if np.any(np.sum(adata.X, axis=1) <= 0):
                print("WARNING: found irregular node sizes in image %s" % str(i))
        # Get global mean of feature intensity across all features:
        global_mean_per_node = self.celldata.X.sum(axis=1).mean(axis=0)
        return {i: global_mean_per_node / np.sum(adata.X, axis=1) for i, adata in self.img_celldata.items()}


class DataLoaderZhang(DataLoader):
    def _register_celldata(self):
        """
        Registers an Anndata object over all images and collects all necessary information.
        :return:
        """
        metadata = {
            "lateral_resolution": 0.105,
            "fn": "preprocessed_zhang.h5ad",
            "image_col": "slice_id",
            "pos_cols": ["center_x", "center_y"],
            "cluster_col": "subclass",
            "patient_col": "mouse",
        }

        celldata = read_h5ad(self.data_path + metadata["fn"])
        celldata = celldata[celldata.obs[metadata["image_col"]] != "Dirt"]
        celldata.uns["metadata"] = metadata
        celldata.uns["img_keys"] = list(np.unique(celldata.obs[metadata["image_col"]]))

        img_to_patient_dict = {
            str(x): celldata.obs[metadata["patient_col"]].values[i].split("_")[0]
            for i, x in enumerate(celldata.obs[metadata["image_col"]].values)
        }
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict

        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = celldata.obs[metadata["pos_cols"]]

        # register node type names
        node_type_names = list(np.unique(celldata.obs[metadata["cluster_col"]]))
        celldata.uns["node_type_names"] = {x: x.replace('_', ' ') for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [node_type_names.index(x) for x in celldata.obs[metadata["cluster_col"]].values]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def merge_types_predefined(self):
        pass

    def _register_img_celldata(self):
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        # Save processed data to attributes.
        for k, adata in self.img_celldata.items():
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
    def _register_celldata(self):
        """
        Registers an Anndata object over all images and collects all necessary information.
        :return:
        """
        metadata = {
            "lateral_resolution": 0.5,
            "fn": "raw_inflamed_colon_1.h5ad",
            "image_col": "Annotation",
            "pos_cols": ["X", "Y"],
            "cluster_col": "celltype_Level_2",
            "patient_col": None,
        }

        celldata = read_h5ad(self.data_path + metadata["fn"])
        celldata = celldata[celldata.obs[metadata["image_col"]] != "Dirt"]
        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata.obs[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = celldata.obs[metadata["pos_cols"]]

        img_to_patient_dict = {k: "p_1" for k in img_keys}
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict

        # register node type names
        node_type_names = list(np.unique(celldata.obs[metadata["cluster_col"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [node_type_names.index(x) for x in celldata.obs[metadata["cluster_col"]].values]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def merge_types_predefined(self):
        cell_type_tumor_dict = {
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

        self.merge_types(cell_type_tumor_dict)

    def _register_img_celldata(self):
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        # Save processed data to attributes.
        for k, adata in self.img_celldata.items():
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
    def _register_celldata(self):
        """
        Registers an Anndata object over all images and collects all necessary information.
        :return:
        """
        metadata = {
            "lateral_resolution": 400 / 1024,
            "fn": ["scMEP_MIBI_singlecell/scMEP_MIBI_singlecell.csv", "scMEP_sample_description.xlsx"],
            "image_col": "point",
            "pos_cols": ["center_colcoord", "center_rowcoord"],
            "cluster_col": "Cluster",
            "patient_col": "donor",
        }
        celldata_df = read_csv(self.data_path + metadata["fn"][0])
        celldata_df = celldata_df.dropna(inplace=False).reset_index()
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

        celldata = AnnData(X=celldata_df[feature_cols], obs=celldata_df[["point", "cell_id", "donor", "Cluster"]])

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

        # register node type names
        node_type_names = list(np.unique(celldata_df[metadata["cluster_col"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [node_type_names.index(x) for x in celldata_df[metadata["cluster_col"]].values]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def merge_types_predefined(self):
        pass

    def _register_img_celldata(self):
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection: Union[List[str], None] = None):
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

        tissue_meta_data = read_excel(self.data_path + "scMEP_sample_description.xlsx", usecols=usecols)
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
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "continuous":
                label_tensors[feature] = (tissue_meta_data[feature].values - continuous_mean[feature]) / continuous_std[
                    feature
                ]
                label_names[feature] = [feature]
        # 2. One-hot encode categorical columns
        # Force all entries in categorical columns to be string so that GLM-like formula processing can be performed.
        for i, feature in enumerate(list(label_cols.keys())):
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
    def _register_celldata(self):
        """
        Registers an Anndata object over all images and collects all necessary information.
        :return:
        """
        metadata = {
            "lateral_resolution": 0.325,
            "fn": ["TONSIL_MFI_nuclei_data_table.xlsx", "TONSIL_MFI_membranes_data_table.xlsx"],
            "image_col": "img_keys",
            "pos_cols": ["Location_Center_X", "Location_Center_Y"],
            "cluster_col": "cell_class",
            "patient_col": None,
        }
        nuclei_df = read_excel(self.data_path + metadata["fn"][0])
        membranes_df = read_excel(self.data_path + metadata["fn"][1])

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

        # register node type names
        node_type_names = list(np.unique(celldata_df[metadata["cluster_col"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [node_type_names.index(x) for x in celldata_df[metadata["cluster_col"]].values]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def merge_types_predefined(self):
        pass

    def _register_img_celldata(self):
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        # Save processed data to attributes.
        for k, adata in self.img_celldata.items():
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
    def _register_celldata(self):
        """
        Registers an Anndata object over all images and collects all necessary information.
        :return:
        """
        metadata = {
            "lateral_resolution": 0.377442,
            "fn": "CRC_clusters_neighborhoods_markers_NEW.csv",
            "image_col": "File Name",
            "pos_cols": ["X:X", "Y:Y"],
            "cluster_col": "ClusterName",
            "patient_col": "patients",
        }
        celldata_df = read_csv(self.data_path + metadata["fn"])

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

        celldata = AnnData(X=celldata_df[feature_cols], obs=celldata_df[["File Name", "patients", "ClusterName"]])

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): celldata_df[metadata["patient_col"]].values[i]
            for i, x in enumerate(celldata_df[metadata["image_col"]].values)
        }
        # img_to_patient_dict = {k: "p_1" for k in img_keys}
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict

        # register node type names
        node_type_names = list(np.unique(celldata_df[metadata["cluster_col"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [node_type_names.index(x) for x in celldata_df[metadata["cluster_col"]].values]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def merge_types_predefined(self):
        """
        Merges loaded cell types based on defined dictionary.
        :return:
        """
        cell_type_tumor_dict = {
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
        }
        self.merge_types(cell_type_tumor_dict)

    def _register_img_celldata(self):
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[self.celldata.obs[image_col] == k].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        # Save processed data to attributes.
        for k, adata in self.img_celldata.items():
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
