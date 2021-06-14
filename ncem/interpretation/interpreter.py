import numpy as np
import pandas as pd
import pickle
from scipy import sparse, stats
import tensorflow as tf
from typing import Dict, List, Tuple, Union
from anndata import AnnData
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix

import ncem.estimators as estimators
import ncem.train as train
import ncem.models.layers as layers
import ncem.models as models
import ncem.utils.gen_losses as losses
import ncem.utils.gen_metrics as metrics

markers = ['o', 'v', '1', 's', 'p', '+', 'x', 'D', '*']


class Interpreter(estimator.Estimator):
    def __init__(self):
        super().__init__()

    def init_model(self):
        raise ValueError("models should not be initialized within interpreter class, use load_model()")

    def load_model(
            self,
            results_path: str,
            gs_id: str,
            cv_idx: int,
            subset_hyperparameters: Union[None, List[Tuple[str, str]]] = None,
            model_id: str = None,
            expected_pickle: Union[None, list] = None
    ):
        """
        Load best or selected model from grid search directory.

        :param results_path:
        :param gs_id:
        :param cv_idx:
        :param subset_hyperparameters:
        :param model_id:
        :param expected_pickle:
        :return:
        """
        if subset_hyperparameters is None:
            subset_hyperparameters = []
        if expected_pickle is None:
            expected_pickle = ['evaluation', 'history', 'hyperparam', 'model_args', 'time']

        gscontainer = train.GridSearchContainerGenerative(results_path, gs_id)
        gscontainer.load_gs(
            expected_pickle=expected_pickle
        )
        if model_id is None:
            model_id = gscontainer.get_best_model_id(
                subset_hyperparameters=subset_hyperparameters,
                metric_select="loss",
                cv_mode="mean",
                partition_select="test"
            )
        cv = gscontainer.select_cv(cv_idx=cv_idx)
        fn_model_kwargs = f"{results_path}{gs_id}/results/{model_id}_{cv}_model_args.pickle"
        fn_weights = f"{results_path}{gs_id}/results/{model_id}_{cv}_model_weights.tf"
        with open(fn_model_kwargs, 'rb') as f:
            model_kwargs = pickle.load(f)

        self._model_kwargs = model_kwargs
        self._fn_model_weights = fn_weights
        self.gscontainer = gscontainer
        self.model_id = model_id
        self.gs_id = gs_id
        self.gscontainer_runparams = self.gscontainer.runparams[self.gs_id][self.model_id]
        self.results_path = results_path
        print("loaded model %s" % model_id)

    def get_data_again(
            self,
            data_path,
            buffered_data_path,
            data_origin
    ):
        """
        Loads data as previously done during model training.

        :param data_path:
        :param buffered_data_path:
        :param data_origin:
        :return:
        """
        self.cond_type = self.gscontainer_runparams['cond_type'] if 'cond_type' in self.gscontainer_runparams else None
        if self.cond_type == "gcn":
            self.adj_type = "scaled"
        elif self.cond_type in ["gat", "max", None]:
            self.adj_type = "full"
        else:
            raise ValueError("cond_type %s not recognized" % self.cond_type)

        self.get_data_unsupervised(
            data_origin=data_origin,
            data_path=data_path,
            buffered_data_path=buffered_data_path,
            max_dist=int(self.gscontainer_runparams['max_dist']),
            write_buffer=False,
            steps=int(self.gscontainer_runparams['steps']),
            graph_covar_selection=self.gscontainer_runparams['graph_covar_selection'],
            node_feature_space_id_0=self.gscontainer_runparams['node_feature_space_id_0'],
            node_feature_space_id_1=self.gscontainer_runparams['node_feature_space_id_1'],
            feature_transformation=self.gscontainer_runparams['feature_transformation'],
            diseased_only=self.gscontainer_runparams['diseased_only'],
            diseased_as_paired=self.gscontainer_runparams['diseased_as_paired'],
            node_fraction=float(self.gscontainer_runparams['node_fraction']),
            use_covar_node_position=self.gscontainer_runparams['use_covar_node_position'],
            use_covar_node_label=self.gscontainer_runparams['use_covar_node_label'],
            use_covar_graph_covar=self.gscontainer_runparams['use_covar_graph_covar'],
            hold_out_covariate=self.gscontainer_runparams['hold_out_covariate'],
            domain_type=self.gscontainer_runparams['domain_type'],
            merge_node_types_predefined=self.gscontainer_runparams['merge_node_types_predefined'],
            remove_diagonal=self.gscontainer_runparams['remove_diagonal']
        )
        self.position_matrix = self.data.position_matrix
        self.node_type_names = self.data.node_type_names
        self.data_path = data_path
        self.n_eval_nodes_per_graph = self.gscontainer_runparams['n_eval_nodes_per_graph']
        self.model_class = self.gscontainer_runparams['model_class']
        self.data_set = self.gscontainer_runparams['data_set']
        self.max_dist = int(self.gscontainer_runparams['max_dist'])
        self.log_transform = self.gscontainer_runparams['log_transform']
        self.cell_names = list(self.node_type_names.values())
