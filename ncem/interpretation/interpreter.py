import pickle
from typing import Dict, List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import tensorflow as tf

from anndata import AnnData
from scipy import sparse, stats
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

import ncem.estimators as estimators
import ncem.models as models
import ncem.models.layers as layers
import ncem.train as train
import ncem.utils.losses as losses
import ncem.utils.metrics as metrics

markers = ["o", "v", "1", "s", "p", "+", "x", "D", "*"]


class InterpreterBase(estimators.Estimator):
    def __init__(self):
        super().__init__()
        self.data_path = None  # path to saved models and results
        self.position_matrix = None
        self.cell_type = None
        self.adj_type = "full"

    def init_model(self):
        raise ValueError("models should not be initialized within interpreter class, use load_model()")

    def load_model(
        self,
        results_path: str,
        gs_id: str,
        cv_idx: int,
        subset_hyperparameters: Union[None, List[Tuple[str, str]]] = None,
        model_id: str = None,
        expected_pickle: Union[None, list] = None,
        lateral_resolution: float = 1.
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
            expected_pickle = ["evaluation", "history", "hyperparam", "model_args", "time"]

        gscontainer = train.GridSearchContainer(results_path, gs_id, lateral_resolution)
        gscontainer.load_gs(expected_pickle=expected_pickle)
        if model_id is None:
            model_id = gscontainer.get_best_model_id(
                subset_hyperparameters=subset_hyperparameters,
                metric_select="loss",
                cv_mode="mean",
                partition_select="test",
            )
        cv = gscontainer.select_cv(cv_idx=cv_idx)
        fn_model_kwargs = f"{results_path}{gs_id}/results/{model_id}_{cv}_model_args.pickle"
        fn_weights = f"{results_path}{gs_id}/results/{model_id}_{cv}_model_weights.tf"
        with open(fn_model_kwargs, "rb") as f:
            model_kwargs = pickle.load(f)

        self._model_kwargs = model_kwargs
        self._fn_model_weights = fn_weights
        self.gscontainer = gscontainer
        self.model_id = model_id
        self.gs_id = gs_id
        self.gscontainer_runparams = self.gscontainer.runparams[self.gs_id][self.model_id]
        self.results_path = results_path
        print("loaded model %s" % model_id)

    def get_data_again(self, data_path, data_origin):
        """
        Loads data as previously done during model training.

        :param data_path:
        :param data_origin:
        :return:
        """
        self.cond_type = self.gscontainer_runparams["cond_type"] if "cond_type" in self.gscontainer_runparams else None
        if self.cond_type == "gcn":
            self.adj_type = "scaled"
        elif self.cond_type in ["max", None]:
            self.adj_type = "full"
        else:
            raise ValueError("cond_type %s not recognized" % self.cond_type)

        self.get_data(
            data_origin=data_origin,
            data_path=data_path,
            radius=int(self.gscontainer_runparams["max_dist"]),
            graph_covar_selection=self.gscontainer_runparams["graph_covar_selection"],
            node_label_space_id=self.gscontainer_runparams["node_feature_space_id_0"],
            node_feature_space_id=self.gscontainer_runparams["node_feature_space_id_1"],
            # feature_transformation=self.gscontainer_runparams["feature_transformation"],
            use_covar_node_position=self.gscontainer_runparams["use_covar_node_position"],
            use_covar_node_label=self.gscontainer_runparams["use_covar_node_label"],
            use_covar_graph_covar=self.gscontainer_runparams["use_covar_graph_covar"],
            # hold_out_covariate=self.gscontainer_runparams["hold_out_covariate"],
            domain_type=self.gscontainer_runparams["domain_type"],
            merge_node_types_predefined=self.gscontainer_runparams["merge_node_types_predefined"],
            # remove_diagonal=self.gscontainer_runparams["remove_diagonal"],
        )
        # self.position_matrix = self.data.position_matrix
        # self.node_type_names = self.data.node_type_names
        self.data_path = data_path
        self.n_eval_nodes_per_graph = self.gscontainer_runparams["n_eval_nodes_per_graph"]
        self.model_class = self.gscontainer_runparams["model_class"]
        self.data_set = self.gscontainer_runparams["data_set"]
        self.radius = int(self.gscontainer_runparams["max_dist"])
        self.cond_depth = self.gscontainer_runparams["cond_depth"] if 'cond_depth' in self.gscontainer_runparams.keys() else None
        self.log_transform = self.gscontainer_runparams["log_transform"]
        self.cell_names = list(self.node_type_names.values())

    def split_data_byidx_again(
            self,
            cv_idx: int
    ):
        """
        Split data into partitions as done during model training.

        :param cv_idx: Index of cross-validation to plot confusion matrix for.
        :return:
        """
        cv = self.gscontainer.select_cv(cv_idx=cv_idx)
        fn = f"{self.results_path}{self.gs_id}/results/{self.model_id}_{cv}_indices.pickle"
        with open(fn, 'rb') as f:
            indices = pickle.load(f)

        self.split_data_given(
            img_keys_test=indices["test"],
            img_keys_train=indices["train"],
            img_keys_eval=indices["val"],
            nodes_idx_test=indices["test_nodes"],
            nodes_idx_train=indices["train_nodes"],
            nodes_idx_eval=indices["val_nodes"]
        )

    def init_model_again(self):
        if self.model_class in ['cvae', 'vae']:
            model = models.ModelCVAE(**self._model_kwargs)
        elif self.model_class == 'cvae_ncem':
            model = models.ModelCVAEncem(**self._model_kwargs)
        elif self.model_class == ['ed', 'lvmnp']:
            model = models.ModelED(**self._model_kwargs)
        elif self.model_class in ['ed_ncem', 'clvmnp']:
            model = models.ModelEDncem(**self._model_kwargs)
        elif self.model_class in ['linear', 'linear_baseline']:
            model = models.ModelLinear(**self._model_kwargs)
        elif self.model_class in ['interactions', 'interactions_baseline']:
            model = models.ModelInteractions(**self._model_kwargs)
        else:
            raise ValueError("model_class not recognized")
        self.model = model

        self.vi_model = False  # variational inference
        if self.model_class in ["vae", "cvae", "cvae_ncem"]:
            self.vi_model = True

    def load_weights_again(self):
        self.model.training_model.load_weights(self._fn_model_weights)

    def reinitialize_model(
            self,
            changed_model_kwargs: dict,
            print_summary: bool = False
    ):
        assert self.model is not None, "no model loaded, run init_model_again() first"
        # updating new model kwargs
        new_model_kwargs = self._model_kwargs.copy()
        new_model_kwargs.update(changed_model_kwargs)

        if self.model_class in ['cvae', 'vae']:
            reinit_model = models.ModelCVAE(**new_model_kwargs)
        elif self.model_class == 'cvae_ncem':
            reinit_model = models.ModelCVAEncem(**new_model_kwargs)
        elif self.model_class == ['ed', 'lvmnp']:
            reinit_model = models.ModelED(**new_model_kwargs)
        elif self.model_class in ['ed_ncem', 'clvmnp']:
            reinit_model = models.ModelEDncem(**new_model_kwargs)
        elif self.model_class in ['linear', 'linear_baseline']:
            reinit_model = models.ModelLinear(**new_model_kwargs)
        elif self.model_class in ['interactions', 'interactions_baseline']:
            reinit_model = models.ModelInteractions(**new_model_kwargs)
        else:
            raise ValueError("model_class not recognized")
        self.reinit_model = reinit_model

        if print_summary:
            self.reinit_model.training_model.summary()
        print(f"setting reinitialized layer weights to layer weights from model {self.model_id}")
        for layer, new_layer in zip(self.model.training_model.layers, self.reinit_model.training_model.layers):
            new_layer.set_weights(layer.get_weights())

    def _pp_saliencies(
            self,
            gradients,
            h_0,
            h_0_full,
            remove_own_gradient: bool = True,
            absolute_saliencies: bool = True
    ):
        if self.cond_type == 'max':
            gradients = np.nan_to_num(gradients)
        # removing gradient self node type
        if remove_own_gradient:
            identity = np.ones(shape=gradients.shape)
            for i in range(gradients.shape[0]):
                identity[i, :] = h_0
            gradients = gradients * (np.ones(shape=gradients.shape) - identity)
        if absolute_saliencies:
            gradients = np.abs(gradients)
        gradients = np.sum(gradients, axis=-1, dtype=np.float64)
        gradients = np.expand_dims(gradients, axis=0)

        sal = np.matmul(gradients, h_0_full)  # 1 x max_nodes
        # print(sal.shape)
        return sal

    def _neighbourhood_frequencies(
            self,
            a,
            h_0_full,
            discretize_adjacency: bool = True
    ):
        neighbourhood = []
        for i, adj in enumerate(a):
            if discretize_adjacency:
                adj = np.asarray(adj > 0, dtype="int")
            neighbourhood.append(
                np.matmul(
                    adj,
                    h_0_full[i].astype(int)
                )  # 1 x node_types
            )
        neighbourhood = pd.DataFrame(
            np.concatenate(neighbourhood, axis=0),
            columns=self.cell_names,
            dtype=np.float
        )
        return neighbourhood


class InterpreterLinear(estimators.EstimatorLinear, InterpreterBase):
    """
    Inherits all relevant functions specific to EstimatorLinear estimators
    """
    def __init__(self):
        super().__init__()

    def _get_np_data(
            self,
            image_keys: Union[np.ndarray, str],
            nodes_idx: Union[dict, str]
    ) -> Tuple[Tuple[list, list, list, list, list], list]:
        """
        :param image_keys: Observation images indices.
        :param nodes_idx: Observation nodes indices.
        :return: Tuple of list of raw data, one list entry per image / graph
        """
        if isinstance(image_keys, (int, np.int32, np.int64)):
            image_keys = [image_keys]
        ds = self._get_dataset(
            image_keys=image_keys,
            nodes_idx=nodes_idx,
            batch_size=1,
            shuffle_buffer_size=1,
            train=False,
            seed=None
        )
        # Loop over sub-selected data set and sum gradients across all selected observations.
        target = []
        source = []
        sf = []
        node_covar = []
        g = []
        h_obs = []

        for step, (x_batch, y_batch) in enumerate(ds):
            target_batch, source_batch, sf_batch, node_covar_batch, g_batch = x_batch
            target.append(target_batch.numpy().squeeze())
            source.append(source_batch.numpy().squeeze())
            sf.append(sf_batch.numpy().squeeze())
            node_covar.append(node_covar_batch.numpy().squeeze())
            g.append(g_batch.numpy().squeeze())
            h_obs.append(y_batch[0].numpy().squeeze())
        return (target, source, sf, node_covar, g), h_obs


class InterpreterInteraction(estimators.EstimatorInteractions, InterpreterBase):
    """
    Inherits all relevant functions specific to EstimatorInteractions estimators
    """
    def __init__(self):
        super().__init__()

    def _get_np_data(
        self,
        image_keys: Union[np.ndarray, str],
        nodes_idx: Union[dict, str]
    ) -> Tuple[Tuple[list, List[csr_matrix], list, list, list], list]:
        """
        :param image_keys: Observation images indices.
        :param nodes_idx: Observation nodes indices.
        :return: Tuple of list of raw data, one list entry per image / graph
        """
        if isinstance(image_keys, (int, np.int32, np.int64)):
            image_keys = [image_keys]
        ds = self._get_dataset(
            image_keys=image_keys,
            nodes_idx=nodes_idx,
            batch_size=1,
            shuffle_buffer_size=1,
            train=False,
            seed=None
        )
        # Loop over sub-selected data set and sum gradients across all selected observations.
        target = []
        interactions = []
        sf = []
        node_covar = []
        g = []
        h_obs = []

        for step, (x_batch, y_batch) in enumerate(ds):
            target_batch, interaction_batch, sf_batch, node_covar_batch, g_batch = x_batch
            target.append(target_batch.numpy().squeeze())
            interactions.append(csr_matrix(
                (
                    interaction_batch.values.numpy(),
                    (
                        interaction_batch.indices.numpy()[:, 1],
                        interaction_batch.indices.numpy()[:, 2]
                    )
                ),
                shape=interaction_batch.dense_shape.numpy()[1:]
            ).todense())
            sf.append(sf_batch.numpy().squeeze())
            node_covar.append(node_covar_batch.numpy().squeeze())
            g.append(g_batch.numpy().squeeze())
            h_obs.append(y_batch[0].numpy().squeeze())
        return (target, interactions, sf, node_covar, g), h_obs

    def target_cell_relative_performance(
        self,
        image_key: str,
        baseline_model,
        target_cell_type: str,
        undefined_type: Optional[str] = None,
        n_neighbors: int=40,
        n_pcs: Optional[int] = None,
        clean_view: bool = False
    ):
        nodes_idx = {image_key: self.nodes_idx_all[image_key]}
        ds = self._get_dataset(
            image_keys=[image_key],
            nodes_idx=nodes_idx,
            batch_size=1,
            shuffle_buffer_size=1,
            train=False,
            seed=None,
            reinit_n_eval=1
        )
        graph = []
        baseline = []
        with tqdm(total=len(self.nodes_idx_all[image_key])) as pbar:
            for step, (x_batch, y_batch) in enumerate(ds):
                out_graph = self.reinit_model.training_model(x_batch)
                out_graph = np.split(ary=out_graph.numpy().squeeze(), indices_or_sections=2, axis=-1)[0]

                out_base = baseline_model.reinit_model.training_model(x_batch)
                out_base = np.split(ary=out_base.numpy().squeeze(), indices_or_sections=2, axis=-1)[0]

                r2_graph = stats.linregress(out_graph, y_batch.numpy().squeeze())[2] ** 2
                graph.append(r2_graph)

                r2_base = stats.linregress(out_base, y_batch.numpy().squeeze())[2] ** 2
                baseline.append(r2_base)
                pbar.update(1)

        adata = self.data.img_celldata[image_key].copy()
        if undefined_type:
            adata = adata[adata.obs[adata.uns['metadata']['cluster_col_preprocessed']] != undefined_type]
        adata.obs['relative_r_squared'] = np.array(graph) - np.array(baseline)
            
        adata_tc =adata[(adata.obs[adata.uns['metadata']['cluster_col_preprocessed']] == target_cell_type)].copy()
        sc.pp.neighbors(adata_tc, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.louvain(adata_tc)
        sc.tl.umap(adata_tc)

        print('n cells: ', adata_tc.shape[0])
        adata_tc.obs[f"{target_cell_type} substates"] = f"{target_cell_type} " + adata_tc.obs.louvain.astype(str)
        adata_tc.obs[f"{target_cell_type} substates"] = adata_tc.obs[f"{target_cell_type} substates"].astype("category")
        
        print(adata_tc.obs[f"{target_cell_type} substates"].value_counts())
        
        return adata, adata_tc
    
    def plot_substate_performance(
        self,
        adata,
        target_cell_type: str,
        relative_performance_key: str = 'relative_r_squared',
        fontsize: int =14,
        figsize: Tuple[float, float] = (4., 4.),
        palette: list = ['#1f77b4', '#2ca02c', '#8c564b', '#7f7f7f', '#17becf'],
        save: Union[str, None] = None,
        suffix: str = "_substates_performance.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        sc.set_figure_params(scanpy=True, fontsize=fontsize)
        plt.rcParams['axes.grid'] = False
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.boxplot(
            data=adata.obs, x=f"{target_cell_type} substates", y=relative_performance_key, 
            order=list(np.unique(adata.obs[f"{target_cell_type} substates"])),
            palette=palette,
            ax=ax
        )
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelrotation=90)
        
        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + target_cell_type + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None
        
    def plot_spatial_relative_performance(
        self,
        adata,
        target_cell_type: str,
        relative_performance_key: str = 'relative_r_squared',
        figsize: Tuple[float, float] =(7., 5.),
        fontsize: Optional[int] = None,
        spot_size: int = 40,
        clean_view: bool = False,
        save: Union[str, None] = None,
        suffix: str = "_spatial_relative_performance.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        class MidpointNormalize(colors.Normalize):
            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                colors.Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                # I'm ignoring masked values and all kinds of edge cases to make a
                # simple example...
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))
            
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
            
        if clean_view:
            adata = adata[np.argwhere(np.array(adata.obsm['spatial'])[:, 1] < 0).squeeze()]
        sc.pl.spatial(
            adata[adata.obs[adata.uns['metadata']['cluster_col_preprocessed']] != target_cell_type], 
            spot_size=spot_size,  
            ax=ax,
            show=False,
            na_color='whitesmoke',
            title=''
        )
        sc.pl.spatial(
            #adata,
            adata[adata.obs[adata.uns['metadata']['cluster_col_preprocessed']] == target_cell_type], 
            color=relative_performance_key, 
            spot_size=spot_size, 
            cmap='coolwarm',
            norm=MidpointNormalize(midpoint=0.), 
            ax=ax,
            show=False,
            title=''
        )

        ax.invert_yaxis()
        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + target_cell_type + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None
        
        

class InterpreterGraph(estimators.EstimatorGraph, InterpreterBase):
    """
    Inherits all relevant functions specific to EstimatorGraph estimators
    """

    def __init__(self):
        super().__init__()

    def _get_np_data(
        self,
        image_keys: Union[np.ndarray, str],
        nodes_idx: Union[dict, str]
    ) -> Tuple[Tuple[list, list, list, list, List[csr_matrix], List[csr_matrix], list, list], list]:
        """
        :param image_keys: Observation images indices.
        :param nodes_idx: Observation nodes indices.
        :return: Tuple of list of raw data, one list entry per image / graph
        """
        if isinstance(image_keys, (int, np.int32, np.int64)):
            image_keys = [image_keys]
        ds = self._get_dataset(
            image_keys=image_keys,
            nodes_idx=nodes_idx,
            batch_size=1,
            shuffle_buffer_size=1,
            train=False,
            seed=None
        )
        # Loop over sub-selected data set and sum gradients across all selected observations.
        h_1 = []
        sf = []
        h_0 = []
        h_0_full = []
        a = []
        a_full = []
        node_covar = []
        g = []
        h_obs = []

        for step, (x_batch, y_batch) in enumerate(ds):
            h_1_batch, sf_batch, h_0_batch, h_0_full_batch, a_batch, a_full_batch, node_covar_batch, g_batch = x_batch
            h_1.append(h_1_batch.numpy().squeeze())
            sf.append(sf_batch.numpy().squeeze())
            h_0.append(h_0_batch.numpy().squeeze())
            h_0_full.append(h_0_full_batch.numpy().squeeze())
            a.append(csr_matrix(
                (
                    a_batch.values.numpy(),
                    (
                        a_batch.indices.numpy()[:, 1],
                        a_batch.indices.numpy()[:, 2]
                    )
                ),
                shape=a_batch.dense_shape.numpy()[1:]
            ))
            a_full.append(csr_matrix(
                (
                    a_full_batch.values.numpy().squeeze(),
                    (
                        a_full_batch.indices.numpy().squeeze()[:, 1],
                        a_full_batch.indices.numpy().squeeze()[:, 2]
                    )
                ),
                shape=a_full_batch.dense_shape.numpy().squeeze()[1:]
            ))
            node_covar.append(node_covar_batch.numpy().squeeze())
            g.append(g_batch.numpy().squeeze())
            h_obs.append(y_batch[0].numpy().squeeze())
        return (h_1, sf, h_0, h_0_full, a, a_full, node_covar, g), h_obs
    
    
class InterpreterEDncem(estimators.EstimatorEDncem, InterpreterGraph):
    """
    Inherits all relevant functions specific to EstimatorInteractions estimators
    """
    def __init__(self):
        super().__init__()
        
    def target_cell_saliencies(
        self,
        target_cell_type: str,
        drop_columns: Optional[list] = None,
        dop_images: Optional[list] = None,
        partition: str = 'test',
        multicolumns: Optional[list] = None
    ):
        target_cell_idx = list(self.node_type_names.values()).index(target_cell_type)
        
        if partition == 'test':
            img_keys = self.img_keys_test
            node_idxs = self.nodes_idx_test
        elif partition == 'train':
            img_keys = self.img_keys_train
            node_idxs = self.nodes_idx_train
        elif partition == 'val':
            img_keys = self.img_keys_val
            node_idxs = self.nodes_idx_val
        elif partition == 'all':
            img_keys = self.img_keys_all
            node_idxs = self.nodes_idx_all
            
        img_saliency = []
        keys = []
        with tqdm(total=len(img_keys)) as pbar:
            for key in img_keys:
                if key in ['41', '36']:
                    continue
                idx = {key: node_idxs[key]}
                ds = self._get_dataset(
                    image_keys=[key],
                    nodes_idx=idx,
                    batch_size=1,
                    shuffle_buffer_size=1,
                    train=False,
                    seed=None,
                    reinit_n_eval=1
                )
                # inputs extracted for plotting aggregation
                saliencies = []
                h_1 = []
                h_0 = []
                h_0_full = []
                a = []
                for step, (x_batch, y_batch) in enumerate(ds):
                    h_1_batch = x_batch[0].numpy().squeeze()
                    h_0_batch = x_batch[2].numpy().squeeze()  # 1 x 1 x node_types
                    h_0_full_batch = x_batch[3]  # 1 x max_nodes x node_types
                    a_batch = x_batch[4]  # 1 x 1 x max_nodes

                    if h_0_batch[target_cell_idx] == 1.:
                        h_1.append(h_1_batch)
                        h_0.append(h_0_batch)
                        h_0_full.append(h_0_full_batch.numpy().squeeze())
                        a.append(sparse.csr_matrix(
                            (
                                a_batch.values.numpy(),
                                (
                                    a_batch.indices.numpy()[:, 1],
                                    a_batch.indices.numpy()[:, 2]
                                )
                            ),
                            shape=a_batch.dense_shape.numpy()[1:]
                        ).toarray())

                        # gradients with target h_0_full_batch
                        with tf.GradientTape(persistent=True) as tape:
                            tape.watch([h_0_full_batch])
                            model_out = self.reinit_model.training_model(x_batch)[0]  # 1 x max_nodes x node_features
                        grads = tape.gradient(model_out, h_0_full_batch)[0].numpy()  # 1 x max_nodes x node_types
                        grads = self._pp_saliencies(
                            gradients=grads,
                            h_0=h_0_batch,
                            h_0_full=h_0_full_batch.numpy().squeeze(),
                            remove_own_gradient=True,
                            absolute_saliencies=False
                        )
                        saliencies.append(grads)
                if len(saliencies) == 0:
                    continue
                saliencies = np.concatenate(saliencies, axis=0)
                n_cells = saliencies.shape[0]
                saliencies = np.mean(saliencies, axis=0)
                neighbourhood = self._neighbourhood_frequencies(
                        a=a,
                        h_0_full=h_0_full,
                        discretize_adjacency=True
                )
                neighbourhood = np.mean(np.array(neighbourhood), axis=0)
                normalized = saliencies / n_cells
                img_saliency.append(normalized)
                keys.append(key)
                pbar.update(1)
        
        if multicolumns:
            columns = [(x.replace('_', ' ').split()[0], x.replace('_', ' ').split()[1]) for x in keys]
        else:
            columns = keys
        df = pd.DataFrame(
            np.concatenate(np.expand_dims(img_saliency, axis=0), axis=1).T, 
            columns=columns,
            index=list(self.node_type_names.values())
        )
        if multicolumns:
            df.columns = pd.MultiIndex.from_tuples(df.columns, names=multicolumns)
        
        df = df.reindex(sorted(df.columns), axis=1)
        
        if drop_columns:
            df = df.drop(drop_columns)
        if dop_images:
            df = df.drop(dop_images, axis=1, level=1)
            
        return df
    
    def plot_target_cell_saliencies(
        self,
        saliencies,
        multiindex: bool = True,
        fontsize: int = 14,
        figsize: Tuple[float, float] = (30., 7.),
        width_ratios: list = [5, 1],
        save: Union[str, None] = None,
        suffix: str = "_imagewise_target_cell_saliencies.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        from collections import OrderedDict
        sc.set_figure_params(scanpy=True, fontsize=fontsize)
        plt.rcParams['axes.grid'] = False
        plt.ioff()
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, gridspec_kw={'width_ratios': width_ratios})
        
        sns.heatmap(saliencies, cmap='seismic', ax=ax[0], center=0)
        ax[0].set_xlabel('')
        
        if multiindex:
            xlabel_mapping = OrderedDict()
            for index1, index2 in saliencies.columns:
                xlabel_mapping.setdefault(index1, [])
                xlabel_mapping[index1].append(index2)

            hline = []
            new_xlabels = []
            for index1, index2_list in xlabel_mapping.items():
                index2_list[0] = "{}".format(index2_list[0])
                new_xlabels.extend(index2_list)

                if hline:
                    hline.append(len(index2_list) + hline[-1])
                else:
                    hline.append(len(index2_list))
            ax[0].set_xticklabels(new_xlabels)
        
        sns.boxplot(data=saliencies.T, ax=ax[1], color='steelblue', showfliers=False)
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
        
        
            

class InterpreterNoGraph(estimators.EstimatorNoGraph, InterpreterBase):
    """
    Inherits all relevant functions specific to EstimatorEDncem estimators and InterpreterBase
    """
    def __init__(self):
        super().__init__()

    def _get_np_data(
        self,
        image_keys: Union[np.ndarray, str],
        nodes_idx: Union[dict, str]
    ) -> Tuple[Tuple[list, list, list, list], list]:
        """
        :param image_keys: Observation images indices.
        :param nodes_idx: Observation nodes indices.
        :return: Tuple of list of raw data, one list entry per image / graph
        """
        if isinstance(image_keys, (int, np.int32, np.int64)):
            image_keys = [image_keys]
        ds = self._get_dataset(
            image_keys=image_keys,
            nodes_idx=nodes_idx,
            batch_size=1,
            shuffle_buffer_size=1,
            train=False,
            seed=None
        )
        # Loop over sub-selected data set and sum gradients across all selected observations.
        h_1 = []
        sf = []
        node_covar = []
        g = []
        h_obs = []

        for step, (x_batch, y_batch) in enumerate(ds):
            h_1_batch, sf_batch, node_covar_batch, g_batch = x_batch
            h_1.append(h_1_batch.numpy().squeeze())
            sf.append(sf_batch.numpy().squeeze())
            node_covar.append(node_covar_batch.numpy().squeeze())
            g.append(g_batch.numpy().squeeze())
            h_obs.append(y_batch[0].numpy().squeeze())
        return (h_1, sf, node_covar, g), h_obs

    


    