import abc
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import squidpy as sq
from anndata import AnnData, read_h5ad
from pandas import read_csv

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

    def _get_degrees(
        self,
        max_distances: list
    ):
        degs = {}
        degrees = {}
        for k, adata in self.img_celldata.items():
            dist_matrix = adata.obsp['adjacency_matrix_distances']
            degs[k] = {dist: np.sum(dist_matrix < dist * dist, axis=0) for dist in max_distances}
        for dist in max_distances:
            degrees[dist] = [deg[dist] for deg in degs.values()]
        return degrees

    def plot_degree_vs_dist(
            self,
            degree_matrices=None,
            max_distances=None,
            lateral_resolution: float = 1.,
            save: Union[str, None] = None,
            suffix: str = "_degree_vs_dist.pdf",
            show: bool = True,
            return_axs: bool = False
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
                raise ValueError('Provide either distance matrices or distance values!')
            else:
                degree_matrices = self._get_degrees(
                    max_distances
                )

        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.ioff()
        fig = plt.figure(figsize=(4, 3))

        mean_degree = []
        distances = []

        for dist, degrees in degree_matrices.items():
            mean_d = [np.mean(degree) for degree in degrees]
            print(np.mean(mean_d))
            mean_degree += mean_d
            distances += [np.int(dist*lateral_resolution)] * len(mean_d)

        sns_data = pd.DataFrame({
            "dist": distances,
            "mean_degree": mean_degree,
        })
        ax = fig.add_subplot(111)
        sns.boxplot(
            data=sns_data,
            x="dist",
            color='steelblue',
            y="mean_degree",
            ax=ax
        )
        ax.set_yscale('log', basey=10)
        plt.ylabel('')
        plt.xlabel('')
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


class DataLoader(GraphTools):
    def __init__(
        self,
        data_path: str,
        radius: int,
    ):
        self.data_path = data_path

        print("Loading data from raw files")
        self.register_celldata()
        self.register_img_celldata()
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

    @abc.abstractmethod
    def _register_celldata(self):
        pass

    @abc.abstractmethod
    def _register_img_celldata(self):
        pass

    def plot_noise_structure(
            self,
            undefined_type: Union[str, None] = None,
            merge_types: Union[None, Tuple[list, list]] = None,
            min_x: Union[None, float] = None,
            max_x: Union[None, float] = None,
            panel_width: float = 2.,
            panel_height: float = 2.7,
            save: Union[str, None] = None,
            suffix: str = "_noise_structure.pdf",
            show: bool = True,
            return_axs: bool = False
    ):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
        import seaborn as sns

        feature_mat = pd.concat([
            pd.concat([
                pd.DataFrame({
                    "image": [k for i in range(adata.shape[0])],
                }),
                pd.DataFrame(
                    adata.X,
                    columns=list(adata.var_names)
                ),
                pd.DataFrame(
                    np.asarray(list(adata.uns["node_type_names"].values()))[
                        np.argmax(adata.obsm["node_types"][k], axis=1)
                    ],
                    columns=["cell_type"]
                )
            ], axis=1).melt(value_name="expression", var_name="gene", id_vars=["cell_type", "image"])
            for k, adata in self.img_celldata.items()
        ])
        feature_mat["log_expression"] = np.log(feature_mat["expression"].values + 1)
        if undefined_type is not None:
            feature_mat = feature_mat[feature_mat['cell_type'] != undefined_type]

        if merge_types is not None:
            for mt in merge_types[0]:
                feature_mat = feature_mat.replace(mt, merge_types[-1])

        plt.ioff()
        ct = np.unique(feature_mat["cell_type"].values)
        nrows = len(ct) // 12 + int(len(ct) % 12 > 0)
        fig, ax = plt.subplots(
            ncols=12, nrows=nrows, figsize=(12 * panel_width, nrows * panel_height), sharex='all', sharey='all'
        )
        ax = ax.flat
        for axis in ax[len(ct):]:
            axis.remove()
        for i, ci in enumerate(ct):
            tab = feature_mat.loc[feature_mat["cell_type"].values == ci, :]
            x = np.log(tab.groupby(["gene"])["expression"].mean() + 1)
            y = np.log(tab.groupby(["gene"])["expression"].var() + 1)
            sns.scatterplot(
                x=x,
                y=y,
                ax=ax[i]
            )
            min_x = np.min(x) if min_x is None else min_x
            max_x = np.max(x) if max_x is None else max_x
            sns.lineplot(
                x=[min_x, max_x],
                y=[2 * min_x, 2 * max_x],
                color="black",
                ax=ax[i]
            )
            ax[i].grid(False)
            ax[i].set_title(ci.replace('_', '\n').replace(' ', '\n').replace('/', '\n'), fontsize=14)
            ax[i].set_xlabel("")
            ax[i].set_ylabel("")
            ax[i].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        # ax[0].set_ylabel("log var")
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
            [np.sum(
                self.celldata.obsm["node_types"][:, positions[t]], axis=1, keepdims=True
            ) for t in new_types],
            axis=1
        )
        self.celldata.obsm["node_types"] = new_node_types
        self.celldata.uns["node_type_names"] = {name: name for i, name in enumerate(new_types)}

        for key, adata in self.img_celldata.items():
            new_node_types = np.concatenate([
                np.sum(adata.obsm["node_types"][:, positions[t]], axis=1, keepdims=True) for t in new_types
            ], axis=1)

            adata.obsm["node_types"] = new_node_types
            adata.uns["node_type_names"] = {name: name for i, name in enumerate(new_types)}


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
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [node_type_names.index(x) for x in celldata.obs[metadata["cluster_col"]].values]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def merge_types_predefined(self, coarseness=None):
        pass

    def _register_img_celldata(self):
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[k] = self.celldata[self.celldata.obs[image_col] == k]
        self.img_celldata = img_celldata


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
            img_celldata[k] = self.celldata[self.celldata.obs[image_col] == k]
        self.img_celldata = img_celldata


class DataLoaderHartmann(DataLoader):
    def _register_celldata(self):
        """
        Registers an Anndata object over all images and collects all necessary information.
        :return:
        """
        metadata = {
            "lateral_resolution": 400 / 1024,
            "fn": "scMEP_MIBI_singlecell/scMEP_MIBI_singlecell.csv",
            "image_col": "point",
            "pos_cols": ["center_colcoord", "center_rowcoord"],
            "cluster_col": "Cluster",
            "patient_col": "donor",
        }

        celldata_df = read_csv(self.data_path + metadata["fn"])
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

    def _register_img_celldata(self):
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[k] = self.celldata[self.celldata.obs[image_col] == k]
        self.img_celldata = img_celldata


class DataLoaderPascualReguant(DataLoader):
    pass


class DataLoaderSchuerch(DataLoader):
    pass
