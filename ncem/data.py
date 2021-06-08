import abc
import numpy as np
import scanpy as sc
import squidpy as sq

from anndata import AnnData, read_h5ad
from pandas import read_csv
from typing import Dict, List, Union, Tuple

# ToDo add graph covariates


class GraphTools:
    def compute_adjacency_matrices(self, radius: int, transform: str = None):
        """

        :param max_dist:
        :param transform:
        :return:
        """
        for k, adata in self.img_celldata.items():
            sq.gr.spatial_neighbors(
                adata=adata,
                radius=radius,
                transform=transform,
                key_added='adjacency_matrix'
            )
            print(adata)


class DataLoader(GraphTools):
    def __init__(
            self,
            data_path: str,
            radius: int,
    ):
        self.data_path = data_path

        print('Loading data from raw files')
        self.register_celldata()
        self.register_img_celldata()
        self.compute_adjacency_matrices(radius=radius)
        self.radius = radius

        print(
            "Loaded %i images with complete data from %i patients "
            "over %i cells with %i cell features and %i distinct celltypes." % (
                len(self.img_celldata),
                len(self.patients),
                self.celldata.shape[0],
                self.celldata.shape[1],
                len(self.celldata.uns['node_type_names'])
            )
        )

    @property
    def patients(self):
        return np.unique(np.asarray(list(self.celldata.uns['img_to_patient_dict'].values())))

    def register_celldata(self):
        """
        Loads anndata object of complete dataset.
        :return:
        """
        print('registering celldata')
        self._register_celldata()
        assert self.celldata is not None, "celldata was not loaded"

    def register_img_celldata(self):
        """
        Loads dictionary of of image-wise celldata objects with {imgage key : anndata object of image}.
        :return:
        """
        print('collecting image-wise celldata')
        self._register_img_celldata()
        assert self.img_celldata is not None, "image-wise celldata was not loaded"

    @abc.abstractmethod
    def _register_celldata(self):
        pass

    @abc.abstractmethod
    def _register_img_celldata(self):
        pass


class DataLoaderZhang(DataLoader):

    def _register_celldata(self):
        """
        Registers an Anndata object over all images and collects all necessary information.
        :param data_path:
        :return:
        """
        metadata = {
            "lateral_resolution": 0.105,
            "fn": 'preprocessed_zhang.h5ad',
            "image_col": 'slice_id',
            "pos_cols": ["center_x", "center_y"],
            "cluster_col": "subclass",
            "patient_col": "mouse"
        }

        celldata = read_h5ad(self.data_path + metadata["fn"])
        celldata = celldata[celldata.obs[metadata["image_col"]] != 'Dirt']
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
        node_type_idx = np.array([  # index in encoding vector
            node_type_names.index(x) for x in celldata.obs[metadata["cluster_col"]].values
        ])
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types
        # ToDo merge nodes

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
        :param data_path:
        :return:
        """
        metadata = {
            "lateral_resolution": 0.5,
            "fn": 'raw_inflamed_colon_1.h5ad',
            "image_col": 'Annotation',
            "pos_cols": ["X", "Y"],
            "cluster_col": "celltype_Level_2",
            "patient_col": None
        }

        celldata = read_h5ad(self.data_path + metadata["fn"])
        celldata = celldata[celldata.obs[metadata["image_col"]] != 'Dirt']
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
        node_type_idx = np.array([  # index in encoding vector
            node_type_names.index(x) for x in celldata.obs[metadata["cluster_col"]].values
        ])
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types
        # ToDo merge nodes

        self.celldata = celldata

    def merge_types_predefined(self, coarseness=None):
        cell_type_tumor_dict = {
            'B cells': 'B cells',
            'CD4 T cells': 'CD4 T cells',
            'CD8 T cells': 'CD8 T cells',
            'GATA3+ epithelial': 'GATA3+ epithelial',
            'Ki67 high epithelial': 'Ki67 epithelial',
            'Ki67 low epithelial': 'Ki67 epithelial',
            'Lamina propria cells': 'Lamina propria cells',
            'Macrophages': 'Macrophages',
            'Monocytes': 'Monocytes',
            'PD-L1+ cells': 'PD-L1+ cells',
            'intraepithelial Lymphocytes': 'intraepithelial Lymphocytes',
            'muscular cells': 'muscular cells',
            'other Lymphocytes': 'other Lymphocytes',
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
        :param data_path:
        :return:
        """
        metadata = {
            "lateral_resolution": 400/1024,
            "fn": 'scMEP_MIBI_singlecell/scMEP_MIBI_singlecell.csv',
            "image_col": 'point',
            "pos_cols": ["center_colcoord", "center_rowcoord"],
            "cluster_col": "Cluster",
            "patient_col": "donor"
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

        celldata = AnnData(
            X=celldata_df[feature_cols],
            obs=celldata_df[['point', 'cell_id', 'donor', 'Cluster']]
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
        #img_to_patient_dict = {k: "p_1" for k in img_keys}
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict

        # register node type names
        node_type_names = list(np.unique(celldata_df[metadata["cluster_col"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array([  # index in encoding vector
            node_type_names.index(x) for x in celldata_df[metadata["cluster_col"]].values
        ])
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types
        # ToDo merge nodes

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
