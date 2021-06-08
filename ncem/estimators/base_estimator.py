import abc
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Union, Optional
import warnings
import time
import pickle

from ncem.utils.losses import GaussianLoss, NegBinLoss, KLLoss
from ncem.utils.metrics import custom_mae, custom_mean_sd, custom_mse, custom_mse_scaled, gaussian_reconstruction_loss, \
    nb_reconstruction_loss, custom_kl, r_squared, r_squared_linreg, \
    logp1_custom_mse, logp1_r_squared, logp1_r_squared_linreg


class Estimator:
    """
    Estimator class for models Contains all necessary methods for data loading, model initialization, training,
    evaluation and prediction.
    """

    def __init__(self):
        self.model = None

    def _load_data(
            self,
            data_origin: str,
            data_path: str,
            feature_transformation: str,
            radius: int,
    ):
        """
        Initializes a DataLoader object.
        :param data_origin:
        :param data_path:
        :param feature_transformation:
        :param radius:
        :return:
        """
        if data_origin == 'zhang':
            from ncem.data import DataLoaderZhang as DataLoader
            self.undefined_node_types = ['other']
        else:
            raise ValueError(f"data_origin {data_origin} not recognized")

        self.data = DataLoader(
            data_path,
            radius=radius
        )
        self.data.process_node_features(feature_transformation=feature_transformation)

    def get_data(
            self,
            data_origin: str,
            data_path: str,
            radius: int,
            node_label_space_id: str = "type",
            node_feature_space_id: str = "standard",
            feature_transformation: str = 'none',
            use_covar_node_position: bool = False,
            use_covar_node_label: bool = False,
            use_covar_graph_covar: bool = False,
            hold_out_covariate: Union[str, None] = None,
            domain_type: str = "image",
            merge_node_types_predefined: bool = False,
            remove_diagonal: bool = True
    ):
        #ToDo
        if self.adj_type is None:
            raise ValueError("set adj_type by init_estim() first")

        self._load_data(
            data_origin=data_origin,
            data_path=data_path,
            feature_transformation=feature_transformation,
            radius=radius
        )
        if merge_node_types_predefined:
            self.data.merge_types_predefined()

        self.img_to_patient_dict = self.data.celldata.uns.img_to_patient_dict
        self.a = {k: adata.obsp['adjacency_matrix_connectivities'] for k, adata in self.data.img_celldata.items()}
        if node_label_space_id == 'standard':
            self.h_0 = {k: adata.X for k, adata in self.data.img_celldata.items()}
        elif node_label_space_id == 'type':
            self.h_0 = {k: adata.obsm['node_types'] for k, adata in self.data.img_celldata.items()}
        else:
            raise ValueError("node_label_space_id %s not recognized" % node_label_space_id)
        if node_feature_space_id == 'standard':
            self.h_1 = {k: adata.X for k, adata in self.data.img_celldata.items()}
        elif node_feature_space_id == 'type':
            self.h_1 = {k: adata.obsm['node_types'] for k, adata in self.data.img_celldata.items()}
        else:
            raise ValueError("node_feature_space_id %s not recognized" % node_feature_space_id)
        self.node_types = self.h_1 = {k: adata.obsm['node_types'] for k, adata in self.data.img_celldata.items()}
        self.node_type_names = self.data.celldata.uns["node_type_names"]
        self.n_features_type = list(self.node_types.values())[0].shape[1]
        self.n_features_standard = self.data.celldata.shape[1]
        self.node_feature_names = self.data.celldata.var_names
        self.size_factors = self.data.size_factors()

        self.node_covar = {k: np.empty((adata.shape[0], 0)) for k, adata in self.data.img_celldata.items()}

