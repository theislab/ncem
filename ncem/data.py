import abc
import numpy as np
import scanpy as sc
import squidpy as sq

from anndata import AnnData, read_h5ad
from typing import Dict, List, Union, Tuple


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
            "node_col": "subclass",
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
        node_type_names = list(np.unique(celldata.obs[metadata["node_col"]]))
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array([  # index in encoding vector
            node_type_names.index(x) for x in celldata.obs[metadata["node_col"]].values
        ])
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types
        # ToDo merge nodes

        # ToDo patient information

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
    pass


class DataLoaderHartmann(DataLoader):
    pass


class DataLoaderPascualReguant(DataLoader):
    pass


class DataLoaderSchuerch(DataLoader):
    pass
