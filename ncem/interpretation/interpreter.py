import pickle
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import tensorflow as tf
from anndata import AnnData
from diffxpy.testing.correction import correct
from scipy import sparse, stats
from scipy.sparse import csr_matrix
from tqdm import tqdm
import networkx as nx

import ncem.estimators as estimators
import ncem.models as models
import ncem.train as train
from ncem.utils.wald_test import get_fim_inv, wald_test
from ncem.utils.ols_fit import ols_fit
from patsy import dmatrix


def _get_scanpy_colors():
    from typing import Mapping, Sequence
    from matplotlib import cm, colors

    # Colorblindness adjusted vega_10
    # See https://github.com/theislab/scanpy/issues/387
    vega_10 = list(map(colors.to_hex, cm.tab10.colors))
    vega_10_scanpy = vega_10.copy()
    vega_10_scanpy[2] = '#279e68'  # green
    vega_10_scanpy[4] = '#aa40fc'  # purple
    vega_10_scanpy[8] = '#b5bd61'  # kakhi

    # default matplotlib 2.0 palette
    # see 'category20' on https://github.com/vega/vega/wiki/Scales#scale-range-literals
    vega_20 = list(map(colors.to_hex, cm.tab20.colors))

    # reorderd, some removed, some added
    vega_20_scanpy = [
        # dark without grey:
        *vega_20[0:14:2],
        *vega_20[16::2],
        # light without grey:
        *vega_20[1:15:2],
        *vega_20[17::2],
        # manual additions:
        '#ad494a',
        '#8c6d31',
    ]
    vega_20_scanpy[2] = vega_10_scanpy[2]
    vega_20_scanpy[4] = vega_10_scanpy[4]
    vega_20_scanpy[7] = vega_10_scanpy[8]  # kakhi shifted by missing grey
    
    zeileis_28 = [
        "#023fa5",
        "#7d87b9",
        "#bec1d4",
        "#d6bcc0",
        "#bb7784",
        "#8e063b",
        "#4a6fe3",
        "#8595e1",
        "#b5bbe3",
        "#e6afb9",
        "#e07b91",
        "#d33f6a",
        "#11c638",
        "#8dd593",
        "#c6dec7",
        "#ead3c6",
        "#f0b98d",
        "#ef9708",
        "#0fcfc0",
        "#9cded6",
        "#d5eae7",
        "#f3e1eb",
        "#f6c4e1",
        "#f79cd4",
        # these last ones were added:
        '#7f7f7f',
        "#c7c7c7",
        "#1CE6FF",
        "#336600",
    ]

    default_28 = zeileis_28
    
    return vega_10_scanpy, vega_20_scanpy, default_28


class InterpreterBase(estimators.Estimator):
    """InterpreterBase class."""

    data_path: str
    results_path: str
    model_class: str
    data_set: str
    radius: int
    cell_names: list

    def __init__(self):
        """Initialize InterpreterBase."""
        super().__init__()
        self.cell_type = None
        self.position_matrix = None
        self.adj_type = "full"

        self._model_kwargs = None
        self._fn_model_weights = None
        self.gscontainer = None
        self.model_id = None
        self.gs_id = None
        self.gscontainer_runparams = None

        self.reinit_model = None

    def init_model(self):
        """Init model function in interpreter class.

        Raises:
            ValueError: Models should not be initialized within interpreter class, use load_model().
        """
        raise ValueError("models should not be initialized within interpreter class, use load_model()")

    def load_model(
        self,
        results_path: str,
        gs_id: str,
        cv_idx: int,
        subset_hyperparameters: Optional[List[Tuple[str, str]]] = None,
        model_id: Optional[str] = None,
        expected_pickle: Optional[list] = None,
        lateral_resolution: float = 1.0,
    ):
        """Load best or selected model from grid search directory.

        Args:
            results_path (str): Path to results.
            gs_id (str): Grid search identifier.
            cv_idx (int): Cross-validation index.
            subset_hyperparameters: Subset of hyperparameters.
            model_id (str): Model identifier.
            expected_pickle (list): List of expected pickle files, default "evaluation", "history", "hyperparam",
                "model_args", "time"
            lateral_resolution (float): Lateral resolution.
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
        self.cv_idx = cv_idx
        print("loaded model %s" % model_id)

    def _get_dataset(
        self,
        image_keys: List[str],
        nodes_idx: Dict[str, np.ndarray],
        batch_size: int,
        shuffle_buffer_size: int,
        train: bool,
        seed: Optional[int],
        prefetch: int = 100,
        reinit_n_eval: Optional[int] = None,
    ):
        """Prepares a dataset.

        Args:
            image_keys (list): Image keys in partition.
            nodes_idx (dict): Dictionary of nodes per image in partition.
            batch_size (int): Batch size.
            shuffle_buffer_size (int): Shuffle buffer size.
            train (bool): Whether dataset is used for training or not (influences shuffling of nodes).
            seed (int): Random seed.
            prefetch (int): Prefetch of dataset.
            reinit_n_eval (int): Used if model is reinitialized to different number of nodes per graph.
        """
        pass

    def _get_resampled_dataset(
        self, image_keys: np.ndarray, nodes_idx: dict, batch_size: int, seed: Optional[int] = None, prefetch: int = 100
    ):
        """Evaluates model based on resampled dataset for posterior resampling.
        node_1 + domain_1 -> encoder -> z_1 + domain_2 -> decoder -> reconstruction_2

        Args:
            image_keys (list): Image keys in partition.
            nodes_idx (dict): Dictionary of nodes per image in partition.
            batch_size (int): Batch size.
            seed (int): Random seed.
            prefetch (int): Prefetch.
        """
        pass

    def get_data_again(self, data_path: str, data_origin: str, n_top_genes = None):
        """Loads data as previously done during model training.

        Args:
            data_path (str): Data path.
            data_origin (str): Data origin.

        Raises:
            ValueError: If `cond_type` is not recognized.
        """
        self.cond_type = self.gscontainer_runparams["cond_type"] if "cond_type" in self.gscontainer_runparams else None
        if self.cond_type == "gcn":
            self.adj_type = "scaled"
        elif self.cond_type in ["max", 'lr_gat', None]:
            self.adj_type = "full"
        else:
            raise ValueError("cond_type %s not recognized" % self.cond_type)
        if "radius" in self.gscontainer_runparams.keys():
            radius = int(self.gscontainer_runparams["radius"])
        else:
            radius = int(self.gscontainer_runparams["max_dist"])

        if "node_label_space_id" in self.gscontainer_runparams.keys():
            node_label_space_id = self.gscontainer_runparams["node_label_space_id"]
        else:
            node_label_space_id = self.gscontainer_runparams["node_feature_space_id_0"]

        if "node_feature_space_id" in self.gscontainer_runparams.keys():
            node_feature_space_id = self.gscontainer_runparams["node_feature_space_id"]
        else:
            node_feature_space_id = self.gscontainer_runparams["node_feature_space_id_1"]

        self.get_data(
            data_origin=data_origin,
            data_path=data_path,
            radius=radius,
            graph_covar_selection=self.gscontainer_runparams["graph_covar_selection"],
            node_label_space_id=node_label_space_id,
            node_feature_space_id=node_feature_space_id,
            use_covar_node_position=self.gscontainer_runparams["use_covar_node_position"],
            use_covar_node_label=self.gscontainer_runparams["use_covar_node_label"],
            use_covar_graph_covar=self.gscontainer_runparams["use_covar_graph_covar"],
            domain_type=self.gscontainer_runparams["domain_type"],
            n_top_genes=n_top_genes
        )
        self.data_path = data_path
        self.n_eval_nodes_per_graph = self.gscontainer_runparams["n_eval_nodes_per_graph"]
        self.model_class = self.gscontainer_runparams["model_class"]
        self.data_set = self.gscontainer_runparams["data_set"]
        self.radius = radius
        self.cond_depth = (
            self.gscontainer_runparams["cond_depth"] if "cond_depth" in self.gscontainer_runparams.keys() else None
        )
        self.log_transform = self.gscontainer_runparams["log_transform"]
        self.cell_names = list(self.node_type_names.values())

    def split_data_byidx_again(self, cv_idx: int):
        """Split data into partitions as done during model training.

        Args:
            cv_idx (int): Index of cross-validation to plot confusion matrix for.
        """
        cv = self.gscontainer.select_cv(cv_idx=cv_idx)
        fn = f"{self.results_path}{self.gs_id}/results/{self.model_id}_{cv}_indices.pickle"
        with open(fn, "rb") as f:
            indices = pickle.load(f)

        self.split_data_given(
            img_keys_test=indices["test"],
            img_keys_train=indices["train"],
            img_keys_eval=indices["val"],
            nodes_idx_test=indices["test_nodes"],
            nodes_idx_train=indices["train_nodes"],
            nodes_idx_eval=indices["val_nodes"],
        )

    def init_model_again(self):
        """Initialize model in interpreter class.

        Raises:
            ValueError: If model_class not recognized
        """
        if self.model_class in ["vae"]:
            model = models.ModelCVAE(**self._model_kwargs)
        elif self.model_class in ["cvae", "cvae_ncem"]:
            model = models.ModelCVAEncem(**self._model_kwargs)
        elif self.model_class == ["ed", "lvmnp"]:
            model = models.ModelED(**self._model_kwargs)
        elif self.model_class in ["ed_ncem", "clvmnp"]:
            model = models.ModelEDncem(**self._model_kwargs)
        elif self.model_class in ["linear", "linear_baseline"]:
            model = models.ModelLinear(**self._model_kwargs)
        elif self.model_class in ["interactions", "interactions_baseline"]:
            model = models.ModelInteractions(**self._model_kwargs)
        elif self.model_class in ['ed_ncem2']:
            model = models.ModelEd2Ncem(cond_type='lr_gat', **self._model_kwargs)
        elif self.model_class in ["deconvolution", "deconvolution_baseline"]:
            model = models.ModelInteractions(**self._model_kwargs)
        else:
            raise ValueError("model_class not recognized")
        self.model = model

        self.vi_model = False  # variational inference
        if self.model_class in ["vae", "cvae", "cvae_ncem"]:
            self.vi_model = True

    def load_weights_again(self):
        """Load model weights again in interpreter class."""
        self.model.training_model.load_weights(self._fn_model_weights)

    def reinitialize_model(self, changed_model_kwargs: dict, print_summary: bool = False):
        """Reinitialize model with changed model kwargs.

        Args:
            changed_model_kwargs (dict): Dictionary over changed model kwargs.
            print_summary (bool): Whether to print summary.

        Raises:
            ValueError: If model_class not recognized
        """
        assert self.model is not None, "no model loaded, run init_model_again() first"
        # updating new model kwargs
        new_model_kwargs = self._model_kwargs.copy()
        new_model_kwargs.update(changed_model_kwargs)

        if self.model_class in ["vae"]:
            reinit_model = models.ModelCVAE(**new_model_kwargs)
        elif self.model_class in ["cvae", "cvae_ncem"]:
            reinit_model = models.ModelCVAEncem(**new_model_kwargs)
        elif self.model_class == ["ed", "lvmnp"]:
            reinit_model = models.ModelED(**new_model_kwargs)
        elif self.model_class in ["ed_ncem", "clvmnp"]:
            reinit_model = models.ModelEDncem(**new_model_kwargs)
        elif self.model_class in ["linear", "linear_baseline"]:
            reinit_model = models.ModelLinear(**new_model_kwargs)
        elif self.model_class in ["interactions", "interactions_baseline"]:
            reinit_model = models.ModelInteractions(**new_model_kwargs)
        elif self.model_class in ["deconvolution", "deconvolution_baseline"]:
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
        self, gradients, h_0, h_0_full, remove_own_gradient: bool = True, absolute_saliencies: bool = True
    ):
        """Preprocessing of saliencies.

        Args:
            gradients (np.array): Array of gradients.
            h_0 (np.array): h_0.
            h_0_full (np.array): h_0_full.
            remove_own_gradient (bool): Whether to remove own gradient.
            absolute_saliencies (bool): Whether absolute saliencies should be aggregated.

        Returns:
            (np.array): Preprocessed saliencies.
        """
        if self.cond_type == "max":
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
        return sal

    def _neighbourhood_frequencies(self, a, h_0_full, discretize_adjacency: bool = True):
        """Computes neighbourhood frequencies.

        Args:
            a (list):  List of adjacency matrices.
            h_0_full (list): List of h_0_full matrices.
            discretize_adjacency (bool): Whether to discretize the adjacency matrices.

        Returns:
            (np.array): Neighbourhood frequencies.
        """
        neighbourhood = []
        for i, adj in enumerate(a):
            if discretize_adjacency:
                adj = np.asarray(adj > 0, dtype="int")
            neighbourhood.append(np.matmul(adj, h_0_full[i].astype(int)))  # 1 x node_types
        neighbourhood = pd.DataFrame(np.concatenate(neighbourhood, axis=0), columns=self.cell_names, dtype=np.float)
        return neighbourhood


class InterpreterLinear(estimators.EstimatorLinear, InterpreterBase):
    """Inherits all relevant functions specific to EstimatorLinear estimators."""

    def __init__(self):
        """Initialize InterpreterLinear."""
        super().__init__()

    def _get_np_data(
        self, image_keys: Union[np.ndarray, str], nodes_idx: Union[dict, str]
    ) -> Tuple[Tuple[list, list, list, list, list], list]:
        """Collects numpy objects from tensorflow dataset.

        Args:
            image_keys: Observation images indices.
            nodes_idx: Observation nodes indices.

        Returns:
            Tuple of list of raw data, one list entry per image / graph
        """
        if isinstance(image_keys, (int, np.int32, np.int64)):
            image_keys = [image_keys]
        ds = self._get_dataset(
            image_keys=image_keys, nodes_idx=nodes_idx, batch_size=1, shuffle_buffer_size=1, train=False, seed=None
        )
        # Loop over sub-selected data set and sum gradients across all selected observations.
        target = []
        source = []
        sf = []
        node_covar = []
        g = []
        h_obs = []

        for _step, (x_batch, y_batch) in enumerate(ds):
            target_batch, source_batch, sf_batch, node_covar_batch, g_batch = x_batch
            target.append(target_batch.numpy().squeeze())
            source.append(source_batch.numpy().squeeze())
            sf.append(sf_batch.numpy().squeeze())
            node_covar.append(node_covar_batch.numpy().squeeze())
            g.append(g_batch.numpy().squeeze())
            h_obs.append(y_batch[0].numpy().squeeze())
        return (target, source, sf, node_covar, g), h_obs


class InterpreterInteraction(estimators.EstimatorInteractions, InterpreterBase):
    """Inherits all relevant functions specific to EstimatorInteractions estimators."""

    def __init__(self):
        """Initialize InterepreterInteraction."""
        super().__init__()

    def _get_np_data(
        self, image_keys: Union[np.ndarray, str], nodes_idx: Union[dict, str]
    ) -> Tuple[Tuple[list, List[csr_matrix], list, list, list], list]:
        """Collects numpy objects from tensorflow dataset.

        Args:
            image_keys: Observation images indices.
            nodes_idx: Observation nodes indices.

        Returns:
            Tuple of list of raw data, one list entry per image / graph
        """
        if isinstance(image_keys, (int, np.int32, np.int64)):
            image_keys = [image_keys]
        ds = self._get_dataset(
            image_keys=image_keys, nodes_idx=nodes_idx, batch_size=1, shuffle_buffer_size=1, train=False, seed=None
        )
        # Loop over sub-selected data set and sum gradients across all selected observations.
        target = []
        interactions = []
        sf = []
        node_covar = []
        g = []
        h_obs = []
        
        count = 0
        for k, v in nodes_idx.items():
            count = count + len(v)
        
        with tqdm(total=np.int(count / self.n_eval_nodes_per_graph)) as pbar:
            for _step, (x_batch, y_batch) in enumerate(ds):
                target_batch, interaction_batch, sf_batch, node_covar_batch, g_batch = x_batch
                target.append(target_batch.numpy().squeeze())
                interactions.append(
                    csr_matrix(
                        (
                            interaction_batch.values.numpy(),
                            (interaction_batch.indices.numpy()[:, 1], interaction_batch.indices.numpy()[:, 2]),
                        ),
                        shape=interaction_batch.dense_shape.numpy()[1:],
                    ).todense()
                )
                sf.append(sf_batch.numpy().squeeze())
                node_covar.append(node_covar_batch.numpy().squeeze())
                g.append(g_batch.numpy().squeeze())
                h_obs.append(y_batch[0].numpy().squeeze())
                pbar.update(1)
        
        target = np.concatenate(target, axis=0)
        interactions = np.concatenate(interactions, axis=0)
        sf = np.concatenate(sf, axis=0)
        node_covar = np.concatenate(node_covar, axis=0)
        g = np.array(g)
        h_obs = np.concatenate(h_obs, axis=0)
        
        return (target, interactions, sf, node_covar, g), h_obs

    def target_cell_relative_performance(
        self,
        image_key: str,
        baseline_model,
        target_cell_type: str,
        undefined_type: Optional[str] = None,
        n_neighbors: int = 15,
        n_pcs: Optional[int] = None,
    ):
        """Compute relative performance of spatial model compared to baseline model and subset to target cell type.

        Args:
            image_key (str): Image key.
            baseline_model: Non-spatial baseline model.
            target_cell_type (str): Target cell type.
            undefined_type (str): Undefined type.
            n_neighbors (int): The size of local neighborhood (in terms of number of neighboring data points) used
                for manifold approximation.
            n_pcs (int): Use this many PCs.

        Returns:
            (Tuple): AnnData object of image and AnnData object subsetted to only target cells.
        """
        cluster_col = self.data.celldata.uns["metadata"]["cluster_col_preprocessed"]
        if isinstance(image_key, str):
            image_key = [image_key]
        adata_list = []
        tqdm_total = 0
        for key in image_key:
            tqdm_total = tqdm_total + len(self.nodes_idx_all[str(key)])
        with tqdm(total=tqdm_total) as pbar:
            for key in image_key:
                nodes_idx = {str(key): self.nodes_idx_all[str(key)]}
                ds = self._get_dataset(
                    image_keys=[str(key)],
                    nodes_idx=nodes_idx,
                    batch_size=1,
                    shuffle_buffer_size=1,
                    train=False,
                    seed=None,
                    reinit_n_eval=1,
                )
                graph = []
                baseline = []
                for _step, (x_batch, y_batch) in enumerate(ds):
                    out_graph = self.reinit_model.training_model(x_batch)
                    out_graph = np.split(ary=out_graph.numpy().squeeze(), indices_or_sections=2, axis=-1)[0]

                    out_base = baseline_model.reinit_model.training_model(x_batch)
                    out_base = np.split(ary=out_base.numpy().squeeze(), indices_or_sections=2, axis=-1)[0]

                    r2_graph = stats.linregress(out_graph, y_batch.numpy().squeeze())[2] ** 2
                    graph.append(r2_graph)

                    r2_base = stats.linregress(out_base, y_batch.numpy().squeeze())[2] ** 2
                    baseline.append(r2_base)
                    pbar.update(1)
                temp_adata = self.data.img_celldata[key].copy()
                if undefined_type:
                    temp_adata = temp_adata[temp_adata.obs[cluster_col] != undefined_type]
                temp_adata.obs["relative_r_squared"] = np.array(graph) - np.array(baseline)
                adata_list.append(temp_adata)

        adata = adata_list[0].concatenate(adata_list[1:], uns_merge="same") if len(adata_list) > 0 else adata_list
        adata_tc = adata[(adata.obs[cluster_col] == target_cell_type)].copy()
        sc.pp.neighbors(adata_tc, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.louvain(adata_tc)
        sc.tl.umap(adata_tc)

        print("n cells: ", adata_tc.shape[0])
        adata_tc.obs[f"{target_cell_type} substates"] = f"{target_cell_type} " + adata_tc.obs.louvain.astype(str)
        adata_tc.obs[f"{target_cell_type} substates"] = adata_tc.obs[f"{target_cell_type} substates"].astype("category")

        print(adata_tc.obs[f"{target_cell_type} substates"].value_counts())

        return adata, adata_tc

    @staticmethod
    def plot_substate_performance(
        adata,
        target_cell_type: str,
        relative_performance_key: str = "relative_r_squared",
        fontsize: Optional[int] = None,
        figsize: Tuple[float, float] = (4.0, 4.0),
        palette: Optional[list] = None,
        save: Optional[str] = None,
        suffix: str = "_substates_performance.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        """Plots the relative the substate performance of a target cell type.

        Args:
            adata (AnnData): AnnData object. Output of `target_cell_relative_performance`.
            target_cell_type (str): Target cell type.
            relative_performance_key (str): Key for relative performance.
            fontsize (int): Fontsize.
            figsize (Tuple): Figure size.
            palette (list): Palette.
            save (str): Path to save directory.
            suffix (str): Saving suffix.
            show (bool): Whether to show figure.
            return_axs (bool): Whether to return axes.

        Returns:
            Optionally returns axes or nothing.
        """
        if palette is None:
            palette = ["#1f77b4", "#2ca02c", "#8c564b", "#7f7f7f", "#17becf"]
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        plt.rcParams["axes.grid"] = False
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.boxplot(
            data=adata.obs,
            x=f"{target_cell_type} substates",
            y=relative_performance_key,
            order=list(np.unique(adata.obs[f"{target_cell_type} substates"])),
            palette=palette,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=90)

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
        relative_performance_key: str = "relative_r_squared",
        figsize: Tuple[float, float] = (7.0, 5.0),
        fontsize: Optional[int] = None,
        spot_size: int = 40,
        clean_view: bool = False,
        save: Optional[str] = None,
        suffix: str = "_spatial_relative_performance.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        """Plots relative performance of spatial versus non-spatial model on spatial allocation of nodes.

        Args:
            adata (AnnData): AnnData object. Output of `target_cell_relative_performance`.
            target_cell_type (str): Target cell type.
            relative_performance_key (str): Key for relative performance.
            figsize (Tuple): Figure size.
            fontsize (int): Fontsize.
            spot_size (int): Spot size.
            clean_view (bool): Whether to remove cells outside of range from plotting.
            save (str): Path to save directory.
            suffix (str): Saving suffix.
            show (bool): Whether to show figure.
            return_axs (bool): Whether to return axes.

        Returns:
            Optionally returns axes or nothing.
        """

        class MidpointNormalize(colors.Normalize):
            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                colors.Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)

        if clean_view:
            adata = adata[np.argwhere(np.array(adata.obsm["spatial"])[:, 1] < 0).squeeze()]
        sc.pl.spatial(
            adata[adata.obs[adata.uns["metadata"]["cluster_col_preprocessed"]] != target_cell_type],
            spot_size=spot_size,
            ax=ax,
            show=False,
            na_color="whitesmoke",
            title="",
        )
        sc.pl.spatial(
            adata[adata.obs[adata.uns["metadata"]["cluster_col_preprocessed"]] == target_cell_type],
            color=relative_performance_key,
            spot_size=spot_size,
            cmap="coolwarm",
            norm=MidpointNormalize(midpoint=0.0),
            ax=ax,
            show=False,
            title="",
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

    def relative_performance_grid(
        self,
        image_keys: Union[np.ndarray, str],
        nodes_idx: Union[dict, str],
        base_interpreter,
        scale_node_frequencies: int,
        metric: str = "r_squared_linreg",
        mode: str = "mean",
        figsize: Tuple[float, float] = (6.0, 5.0),
        save: Optional[str] = None,
        suffix: str = "_expression_grid.pdf",
        show: bool = True,
    ):
        """Plots a cell-cell grid of relative performance in terms of R2 or MAE.

        Args:
            image_keys (Union[np.array, str]): Image keys.
            nodes_idx (Union[dict, str]): Node indices.
            base_interpreter: Baseline model interpreter.
            scale_node_frequencies (int): Value by which to scale the dots in the plot with.
            metric (str): R2 or MAE.
            mode (str): Mean or variance.
            figsize (Tuple): figsize.
            save (str): Path to save directory.
            suffix (str): Saving suffix.
            show (bool): Whether to show figure.

        Raises:
            ValueError: If metric or mode not recognized.
        """

        class MidpointNormalize(colors.Normalize):
            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                colors.Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))

        ds = self._get_dataset(
            image_keys=image_keys, nodes_idx=nodes_idx, batch_size=1, shuffle_buffer_size=1, train=False, seed=None
        )
        node_names = list(self.node_type_names.values())
        interaction_names = []
        for source in node_names:
            for target in node_names:
                interaction_names.append(f"{target} - {source}")
        expression = []
        target = []
        interactions = []

        base_pred = []
        graph_pred = []
        for _step, (x_batch, y_batch) in enumerate(ds):
            target_batch, interactions_batch, sf_batch, node_covar_batch, g_batch = x_batch

            base_pred.append(np.squeeze(base_interpreter.model.training_model(x_batch)[0].numpy()))
            graph_pred.append(np.squeeze(self.model.training_model(x_batch)[0].numpy()))
            expression.append(y_batch[0].numpy().squeeze())
            target.append(target_batch.numpy().squeeze())
            interactions.append(
                sparse.csr_matrix(
                    (
                        interactions_batch.values.numpy(),
                        (interactions_batch.indices.numpy()[:, 1], interactions_batch.indices.numpy()[:, 2]),
                    ),
                    shape=interactions_batch.dense_shape.numpy()[1:],
                ).todense()
            )

        interactions = pd.DataFrame(np.concatenate(interactions, axis=0), columns=interaction_names)
        expression = np.concatenate(expression, axis=0)
        base_pred = np.split(ary=np.concatenate(base_pred, axis=0), indices_or_sections=2, axis=1)[0]
        graph_pred = np.split(ary=np.concatenate(graph_pred, axis=0), indices_or_sections=2, axis=1)[0]

        plt.ioff()
        # function returns a n_features_type x n_features_type grid
        grid_summary = []
        for k in interaction_names:
            temp_interactions = interactions[interactions[k] > 0][k]

            if mode == "mean":
                true = np.mean(expression[list(temp_interactions.index), :], axis=0)
                base = np.mean(base_pred[list(temp_interactions.index), :], axis=0)
                graph = np.mean(graph_pred[list(temp_interactions.index), :], axis=0)
            elif mode == "var":
                true = np.var(expression[list(temp_interactions.index), :], axis=0)
                base = np.var(base_pred[list(temp_interactions.index), :], axis=0)
                graph = np.var(graph_pred[list(temp_interactions.index), :], axis=0)
            else:
                raise ValueError(f"{mode} not implemented")

            if metric == "r_squared_linreg":
                base_metric = stats.linregress(true, base)[2] ** 2
                graph_metric = stats.linregress(true, graph)[2] ** 2
            elif metric == "mae":
                base_metric = np.mean(np.abs(base - true))
                graph_metric = np.mean(np.abs(graph - true))
            else:
                raise ValueError(f"{metric} not implemented")

            target = k.split(" - ")[0]
            source = k.split(" - ")[1]

            grid_summary.append(
                np.array(
                    [target, source, np.sum(np.array(temp_interactions), dtype=np.int32), base_metric, graph_metric]
                )
            )

        expression_grid_summary = np.concatenate(np.expand_dims(grid_summary, axis=0), axis=0)
        plt.ioff()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        temp_df = (
            pd.DataFrame(
                expression_grid_summary, columns=["target", "source", "contact_frequency", "baseline", "graph"]
            )
            .astype({"baseline": "float32", "graph": "float32", "contact_frequency": "int32"})
            .sort_values(["target", "source"], ascending=[False, True])
        )
        if metric == "r_squared_linreg":
            temp_df["relative_performance"] = temp_df.graph - temp_df.baseline
            metric_name = "R2 (linear regression)"
        elif metric == "mae":
            temp_df["relative_performance"] = temp_df.baseline - temp_df.graph
            metric_name = "MAE"
        else:
            raise ValueError(f"{metric} not implemented")
        img0 = ax.scatter(
            x=temp_df.source,
            y=temp_df.target,
            s=temp_df.contact_frequency / scale_node_frequencies,
            c=temp_df.relative_performance,
            cmap="seismic",
            norm=MidpointNormalize(midpoint=0.0),
        )
        cbar = plt.colorbar(img0, ax=ax)
        cbar.set_label(f"relative {metric_name}", rotation=90)
        ax.tick_params(axis="x", labelrotation=90)
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + "_" + metric + suffix)
        if show:
            plt.show()
        plt.close(fig)
        plt.ion()
    
    def get_sender_receiver_effects(
        self,
        params_type: str = 'ols',
        significance_threshold: float = 0.05
    ):
        (target, interactions, _, _, _), y = self._get_np_data(
            image_keys=self.img_keys_all, nodes_idx=self.nodes_idx_all)

        print('using ols parameters.')
        if params_type == 'ols':
            x_design = np.concatenate([target, interactions], axis=1)
            ols = ols_fit(x_=x_design, y_=y)
            params = ols.squeeze()
        else:
            params = (
                self.model.training_model.weights[0]
                .numpy()
                .T
            )

        # get inverse fisher information matrix
        print('calculating inv fim.')
        fim_inv = get_fim_inv(x_design, y)

        is_sign, pvalues, qvalues = wald_test(
            params=params, fisher_inv=fim_inv, significance_threshold=significance_threshold
        )
        interaction_shape = np.int(self.n_features_0**2)
        # subset to interaction terms
        is_sign = is_sign[self.n_features_0 : interaction_shape + self.n_features_0, :]
        pvalues = pvalues[self.n_features_0 : interaction_shape + self.n_features_0, :]
        qvalues = qvalues[self.n_features_0 : interaction_shape + self.n_features_0, :]

        self.pvalues = np.concatenate(
            np.expand_dims(np.split(pvalues, indices_or_sections=np.sqrt(pvalues.shape[0]), axis=0), axis=0),
            axis=0,
        )
        self.qvalues = np.concatenate(
            np.expand_dims(np.split(qvalues, indices_or_sections=np.sqrt(qvalues.shape[0]), axis=0), axis=0),
            axis=0,
        )
        self.is_sign = np.concatenate(
            np.expand_dims(np.split(is_sign, indices_or_sections=np.sqrt(is_sign.shape[0]), axis=0), axis=0),
            axis=0,
        )

        interaction_params = params[:, self.n_features_0 : interaction_shape + self.n_features_0]
        self.fold_change = np.concatenate(
            np.expand_dims(np.split(interaction_params.T, indices_or_sections=np.sqrt(interaction_params.T.shape[0]), axis=0), axis=0),
            axis=0,
        )
        
    def type_coupling_analysis(
        self,
        undefined_types: Optional[List[str]] = None,
        fontsize: Optional[int] = None,
        figsize: Tuple[float, float] = (11.0, 10.0),
        save: Optional[str] = None,
        suffix: str = "_type_coupling_analysis.pdf",
        show: bool = True,
    ):
        
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
            plt.rcParams['axes.grid'] = False
        sig_df = pd.DataFrame(
            np.sum(self.is_sign, axis=-1), 
            columns=self.cell_names,
            index=self.cell_names
        )
        if undefined_types:
            sig_df = sig_df.drop(columns=undefined_types, index=undefined_types)
        np.fill_diagonal(sig_df.values, 0)
        plt.ioff()   
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.heatmap(sig_df, cmap='Greys',)
        plt.xlabel('sender')
        plt.ylabel('receiver')
        plt.tight_layout()
        if save is not None:
            plt.savefig(f"{save}_cv{str(self.cv_idx)}_{suffix}")
        if show:
            plt.show()
        plt.close(fig)
        plt.ion()
        
    def type_coupling_analysis_circular(
        self,
        edge_attr: str,
        edge_width_scale: float = 3.,
        fontsize: Optional[int] = None,
        figsize: Tuple[float, float] = (9, 8),
        de_genes_threshold: float = 0,
        magnitude_threshold: Optional[float] = None,
        save: Optional[str] = None,
        suffix: str = "_type_coupling_analysis_circular.pdf",
        show: bool = True,
        undefined_types: Optional[List[str]] = None,
        text_space: float = 1.15
    ):
        coeff = self.fold_change * self.is_sign
        coeff_df = pd.DataFrame(
            np.sqrt(np.sum(coeff**2, axis=-1)), 
            columns=self.cell_names,
            index=self.cell_names
        )
        network_coeff_df = pd.DataFrame(coeff_df.unstack()).reset_index().rename(columns={'level_0': 'sender', 'level_1': 'receiver'})
        network_coeff_df = network_coeff_df[network_coeff_df['receiver'] != network_coeff_df['sender']]
        
        sig_df = pd.DataFrame(
            np.sum(self.is_sign, axis=-1), 
            columns=self.cell_names,
            index=self.cell_names
        )
        network_df = pd.DataFrame(sig_df.unstack()).reset_index().rename(columns={'level_0': 'sender', 'level_1': 'receiver'})
        network_df = network_df[network_df['receiver'] != network_df['sender']]
        network_df["magnitude"] = network_coeff_df[0]

        network_df["de_genes"] = [
            (np.abs(x) - np.min(np.abs(network_df[0].values))) / 
            (np.max(np.abs(network_df[0].values)) - np.min(np.abs(network_df[0].values)))
            for x in network_df[0].values]
        if undefined_types:
            network_df = network_df[network_df['receiver'] != undefined_types]
            network_df = network_df[network_df['sender'] != undefined_types]
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        vega_10_scanpy, vega_20_scanpy, default_28 = _get_scanpy_colors()
        plt.ioff()   
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax.axis('off')
        if de_genes_threshold:
            network_df = network_df[network_df[0]>de_genes_threshold]
        if magnitude_threshold:
            network_df = network_df[network_df['magnitude']>magnitude_threshold]
        G=nx.from_pandas_edgelist(
            network_df, source='sender', target='receiver', 
            edge_attr=["magnitude", 'de_genes'], 
            create_using=nx.DiGraph()
        ) 
        nodes = np.unique(network_df['receiver'])
        pos=nx.circular_layout(G)
        labels_width = nx.get_edge_attributes(G, edge_attr)
        if len(nodes) <= 10:
            nx.set_node_attributes(G, dict([(x,vega_10_scanpy[i]) for i, x in enumerate(nodes)]), "color")
        elif len(nodes) <= 20:
            nx.set_node_attributes(G, dict([(x,vega_20_scanpy[i]) for i, x in enumerate(nodes)]), "color")
        else:
            nx.set_node_attributes(G, dict([(x,default_28[i]) for i, x in enumerate(nodes)]), "color")
            
        node_color = nx.get_node_attributes(G, 'color')

        description = nx.draw_networkx_labels(G,pos, font_size=17)
        n = len(self.cell_names)
        node_list = sorted(G.nodes())
        angle = []
        angle_dict = {}
        for i, node in zip(range(n),node_list):
            theta = 2.0*np.pi*i/n
            angle.append((np.cos(theta),np.sin(theta)))
            angle_dict[node] = theta
        pos = {}
        for node_i, node in enumerate(node_list):
            pos[node] = angle[node_i]

        r = fig.canvas.get_renderer()
        trans = plt.gca().transData.inverted()
        for node, t in description.items():
            bb = t.get_window_extent(renderer=r)
            bbdata = bb.transformed(trans)
            radius = text_space +bbdata.width/1.
            position = (radius*np.cos(angle_dict[node]),radius* np.sin(angle_dict[node]))
            t.set_position(position)
            t.set_rotation(angle_dict[node]*360.0/(2.0*np.pi))
            t.set_clip_on(False)

        nx.draw_networkx(
            G, pos, with_labels=False, node_size=500,
            width=[x * edge_width_scale for x in list(labels_width.values())], 
            edge_vmin=0., edge_vmax=1., edge_cmap=plt.cm.seismic, arrowstyle='-|>',
            vmin=0., vmax=1., cmap=plt.cm.binary, node_color=list(node_color.values()),
            ax=ax, connectionstyle='arc3, rad = 0.1'
        )
        plt.tight_layout()
        if save is not None:
            plt.savefig(f"{save}_cv{str(self.cv_idx)}_{suffix}")
        if show:
            plt.show()
        plt.close(fig)
        plt.ion()
        
    def sender_receiver_values(
        self,
        receiver: str,
        sender: str,
    ):
        receiver_idx = self.cell_names.index(receiver)
        sender_idx = self.cell_names.index(sender)
        
        fold_change = self.fold_change[receiver_idx,sender_idx,:]
        pvals = self.pvalues[receiver_idx,sender_idx,:]
        qvals = self.qvalues[receiver_idx,sender_idx,:]
        h_0 = pd.DataFrame(
            self.data.celldata.obsm['node_types'], columns=self.cell_names
        )
        target_type = pd.DataFrame(np.array(h_0.idxmax(axis=1)), columns=["target_cell"]).reset_index()
        self.data.celldata.obs = target_type
        means = self.data.celldata[self.data.celldata.obs['target_cell'] == receiver].X.mean(axis=0)
        
        df = pd.DataFrame(
            np.array([means, pvals, qvals, fold_change]).T,
            index=self.data.celldata.var_names, columns=['mean expression', 'pvalue', 'qvalue', 'fold change']
        )
        
        return df
    
    def sender_similarity_analysis(
        self,
        receiver: str,
        fontsize: Optional[int] = None,
        figsize: Tuple[float, float] = (8.0, 8.0),
        save: Optional[str] = None,
        suffix: str = "_sender_similarity_analysis.pdf",
        show: bool = True,
        cbar_pos: Tuple[float, float, float, float] = (-0.3, .1, .4, .02)
    ):
        receiver_idx = self.cell_names.index(receiver)
        
        corrcoef = np.corrcoef(self.fold_change[receiver_idx,:,:])
        corrcoef = pd.DataFrame(
            corrcoef,
            columns=self.cell_names,
            index=self.cell_names
        )
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
            plt.rcParams['axes.grid'] = False
        plt.ioff()   
        clustermap = sns.clustermap(
            corrcoef, cmap='Purples',  
            figsize=figsize, 
            row_cluster=True, 
            cbar_kws={'label': "correlation", "orientation": "horizontal"},
            cbar_pos=cbar_pos
        )
        
        if save is not None:
            clustermap.savefig()
        if show:
            plt.show()
        plt.ion()
        
    def sender_receiver_effect_vulcanoplot(
        self,
        receiver: str,
        sender: str,
        significance_threshold: float = 0.05,
        fold_change_threshold: float = 0.021671495152134755,
        fontsize: Optional[int] = None,
        figsize: Tuple[float, float] = (4.5, 7.0),
        save: Optional[str] = None,
        suffix: str = "_sender_receiver_volcanoplot.pdf",
        show: bool = True,
    ):
        receiver_idx = self.cell_names.index(receiver)
        sender_idx = self.cell_names.index(sender)
        
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
            plt.rcParams['axes.grid'] = False
        fig, ax = plt.subplots(1,1, figsize=figsize)

        # only significant ones 
        qval_filter = np.where(self.qvalues[receiver_idx,sender_idx,:]>=significance_threshold)
        vmax = np.max(np.abs(self.fold_change[receiver_idx,sender_idx,:]))
        print(vmax)
        
        # overlaying significant ones with orange
        sns.scatterplot(
            x=self.fold_change[receiver_idx,sender_idx,:][qval_filter], 
            y=-np.log10(self.qvalues[receiver_idx,sender_idx,:])[qval_filter], 
            color='white', edgecolor = 'black', s=100, ax=ax)

        qval_filter = np.where(self.qvalues[receiver_idx,sender_idx,:]<significance_threshold)
        x = self.fold_change[receiver_idx,sender_idx,:][qval_filter]
        fc_filter = np.where(x < fold_change_threshold)
        y = -np.nan_to_num(np.log10(self.qvalues[receiver_idx,sender_idx,:])[qval_filter])
        sns.scatterplot(
            x=x[fc_filter], 
            y=y[fc_filter], 
            color='darkgrey', edgecolor = 'black', s=100, ax=ax)

        x = self.fold_change[receiver_idx,sender_idx,:][qval_filter]
        fc_filter = np.where(x<= -fold_change_threshold)
        y = -np.nan_to_num(np.log10(self.qvalues[receiver_idx,sender_idx,:])[qval_filter], neginf=-14.5)
        sns.scatterplot(
            x=x[fc_filter], 
            y=y[fc_filter], 
            color='blue', edgecolor = 'black', s=100, ax=ax)

        x = self.fold_change[receiver_idx,sender_idx,:][qval_filter]
        fc_filter = np.where(x>= fold_change_threshold)
        y = -np.nan_to_num(np.log10(self.qvalues[receiver_idx,sender_idx,:])[qval_filter], neginf=-14.5)
        sns.scatterplot(
            x=x[fc_filter], 
            y=y[fc_filter], 
            color='red', edgecolor = 'black', s=100, ax=ax)

        ax.set_xlim((-vmax*1.1, vmax*1.1))
        ax.set_ylim((-0.5, 15))
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.axvline(-fold_change_threshold, color='black', linestyle='--', )
        plt.axvline(fold_change_threshold, color='black', linestyle='--', )
        plt.axhline(-np.log10(significance_threshold), linestyle='--', color='black')
        
        plt.tight_layout()
        if save is not None:
            plt.savefig(f"{save}_cv{str(self.cv_idx)}_{suffix}")
        if show:
            plt.show()
        plt.close(fig)
        plt.ion()
        
    def sender_receiver_gene_subset(
        self,
        receiver: str,
        sender: str,
        significance_threshold: float = 0.05,
        fold_change_quantile: float = 0.2
    ):

        receiver_idx = self.cell_names.index(receiver)
        sender_idx = self.cell_names.index(sender)

        qvals = pd.DataFrame(
            self.qvalues[receiver_idx, sender_idx, :], 
            index=self.data.celldata.var_names
        )
        fold_change = pd.DataFrame(
            np.abs(self.fold_change[receiver_idx, sender_idx, :]), 
            index=self.data.celldata.var_names
        )
        qvals = qvals.replace(0.0, 0.0000001)
        qvals = qvals[qvals <= significance_threshold]
        mask_rows = qvals.any(axis=1)
        qvals = qvals.loc[mask_rows]
        fold_change = fold_change.loc[mask_rows]
        fold_change = fold_change[
            fold_change >= np.max(np.array(fold_change))*fold_change_quantile
        ]
        mask_rows = fold_change.any(axis=1)
        fold_change = fold_change.loc[mask_rows]
        qvals = qvals.loc[mask_rows]
        
        return list(qvals.index)
    
    def sender_effect(
        self,
        receiver: str,
        plot_mode: str = 'fold_change',
        gene_subset: Optional[List[str]] = None,
        significance_threshold: float = 0.05,
        cut_pvals: float = -5,
        fontsize: Optional[int] = None,
        figsize: Tuple[float, float] = (6, 10),
        save: Optional[str] = None,
        suffix: str = "_sender_efect.pdf",
        show: bool = True,
    ):
        receiver_idx = self.cell_names.index(receiver)
        
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        
        if plot_mode == 'qvals':
            arr = np.log(self.qvalues[receiver_idx, :, :])
            arr[arr < cut_pvals] = cut_pvals
            df = pd.DataFrame(
                arr, 
                index=self.cell_names,
                columns=self.data.celldata.var_names
            )
            if gene_subset:
                df = df.drop(index=receiver)[gene_subset]

            plt.ioff()
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            sns.heatmap(
                df.T,
                cbar_kws={'label': "$\log_{10}$ FDR-corrected pvalues"},
                cmap='Greys_r', vmin=-5, vmax=0.
            )
        elif plot_mode == 'fold_change':
            arr = self.fold_change[receiver_idx, :, :]
            arr[np.where(self.qvalues[receiver_idx, :, :] > significance_threshold)] = 0
            df = pd.DataFrame(
                arr, 
                index=self.cell_names,
                columns=self.data.celldata.var_names
            )
            plt.ioff()
            if gene_subset:
                df = df.drop(index=receiver)[gene_subset]
            vmax = np.max(np.abs(df.values))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            sns.heatmap(
                df.T,
                cbar_kws={'label': "fold change", 
                           "location": "top"},
                cmap="seismic", vmin=-vmax, vmax=vmax, 
            )
        plt.xlabel("sender cell type")
        plt.tight_layout()
        if save is not None:
            plt.savefig(f"{save}_cv{str(self.cv_idx)}_{receiver}_{suffix}")
        if show:
            plt.show()
        plt.close(fig)
        plt.ion()
        
    def receiver_effect(
        self,
        sender: str,
        plot_mode: str = 'fold_change',
        gene_subset: Optional[List[str]] = None,
        significance_threshold: float = 0.05,
        cut_pvals: float = -5,
        fontsize: Optional[int] = None,
        figsize: Tuple[float, float] = (6, 10),
        save: Optional[str] = None,
        suffix: str = "_receiver_efect.pdf",
        show: bool = True,
    ):
        sender_idx = self.cell_names.index(sender)
        
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        
        if plot_mode == 'qvals':
            arr = np.log(self.qvalues[:, sender_idx, :])
            arr[arr < cut_pvals] = cut_pvals
            df = pd.DataFrame(
                arr, 
                index=self.cell_names,
                columns=self.data.celldata.var_names
            )
            if gene_subset:
                df = df.drop(index=sender)[gene_subset]

            plt.ioff()
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            sns.heatmap(
                df.T,
                cbar_kws={'label': "$\log_{10}$ FDR-corrected pvalues"},
                cmap='Greys_r', vmin=-5, vmax=0.
            )
        elif plot_mode == 'fold_change':
            arr = self.fold_change[:, sender_idx, :]
            arr[np.where(self.qvalues[:, sender_idx, :] > significance_threshold)] = 0
            df = pd.DataFrame(
                arr, 
                index=self.cell_names,
                columns=self.data.celldata.var_names
            )
            plt.ioff()
            if gene_subset:
                df = df.drop(index=sender)[gene_subset]
            vmax = np.max(np.abs(df.values))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            sns.heatmap(
                df.T,
                cbar_kws={'label': "fold change", 
                           "location": "top"},
                cmap="seismic", vmin=-vmax, vmax=vmax, 
            )
        plt.xlabel("receiver cell type")
        plt.tight_layout()
        if save is not None:
            plt.savefig(f"{save}_cv{str(self.cv_idx)}_{sender}_{suffix}")
        if show:
            plt.show()
        plt.close(fig)
        plt.ion()
        

    def interaction_significance(
        self,
        image_keys,
        nodes_idx,
        significance_threshold: float = 0.01,
    ):
        """Compute interaction parameter significance.

        Args:
            image_keys: Image keys.
            nodes_idx: Node indices.
            significance_threshold (float): Significance threshold for p-values.

        Returns:
            (Tuple): Interaction parameters of model, aaray of boolean significance, significance values.
        """
        ds = self._get_dataset(
            image_keys=image_keys, nodes_idx=nodes_idx, batch_size=1, shuffle_buffer_size=1, train=False, seed=None
        )
        interaction_shape = self.model.training_model.inputs[1].shape
        interaction_params = (
            self.model.training_model.weights[0]
            .numpy()[self.n_features_0 : interaction_shape[-1] + self.n_features_0, :]
            .T
        )

        interactions = []
        y = []
        count = 0
        for k, v in nodes_idx.items():
            count = count + len(v)

        with tqdm(total=count) as pbar:
            for _step, (x_batch, y_batch) in enumerate(ds):
                target_batch, interactions_batch, sf_batch, node_covar_batch, g_batch = x_batch
                interactions.append(
                    sparse.csr_matrix(
                        (
                            interactions_batch.values.numpy(),
                            (interactions_batch.indices.numpy()[:, 1], interactions_batch.indices.numpy()[:, 2]),
                        ),
                        shape=interactions_batch.dense_shape.numpy()[1:],
                    ).todense()
                )
                y.append(y_batch[0].numpy().squeeze())
                pbar.update(1)

        interactions = np.concatenate(interactions, axis=0)
        y = np.concatenate(y, axis=0)

        fim_inv = get_fim_inv(interactions, y)
        print(fim_inv.shape)

        bool_significance, significance = wald_test(
            params=interaction_params, fisher_inv=fim_inv, significance_threshold=significance_threshold
        )
        return interaction_params, bool_significance, significance


class InterpreterGraph(estimators.EstimatorGraph, InterpreterBase):
    """Inherits all relevant functions specific to EstimatorGraph estimators."""

    def __init__(self):
        """Initialize InterpreterGraph."""
        super().__init__()

    def _get_np_data(
        self, image_keys: Union[np.ndarray, str], nodes_idx: Union[dict, str]
    ) -> Tuple[Tuple[list, list, list, list, List[csr_matrix], List[csr_matrix], list, list], list]:
        """Collects numpy objects from tensorflow dataset.

        Args:
            image_keys: Observation images indices.
            nodes_idx: Observation nodes indices.

        Returns:
            Tuple of list of raw data, one list entry per image / graph
        """
        if isinstance(image_keys, (int, np.int32, np.int64)):
            image_keys = [image_keys]
        ds = self._get_dataset(
            image_keys=image_keys, nodes_idx=nodes_idx, batch_size=1, shuffle_buffer_size=1, train=False, seed=None
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

        for _step, (x_batch, y_batch) in enumerate(ds):
            h_1_batch, sf_batch, h_0_batch, h_0_full_batch, a_batch, a_full_batch, node_covar_batch, g_batch = x_batch
            h_1.append(h_1_batch.numpy().squeeze())
            sf.append(sf_batch.numpy().squeeze())
            h_0.append(h_0_batch.numpy().squeeze())
            h_0_full.append(h_0_full_batch.numpy().squeeze())
            a.append(
                csr_matrix(
                    (a_batch.values.numpy(), (a_batch.indices.numpy()[:, 1], a_batch.indices.numpy()[:, 2])),
                    shape=a_batch.dense_shape.numpy()[1:],
                )
            )
            a_full.append(
                csr_matrix(
                    (
                        a_full_batch.values.numpy().squeeze(),
                        (a_full_batch.indices.numpy().squeeze()[:, 1], a_full_batch.indices.numpy().squeeze()[:, 2]),
                    ),
                    shape=a_full_batch.dense_shape.numpy().squeeze()[1:],
                )
            )
            node_covar.append(node_covar_batch.numpy().squeeze())
            g.append(g_batch.numpy().squeeze())
            h_obs.append(y_batch[0].numpy().squeeze())
        return (h_1, sf, h_0, h_0_full, a, a_full, node_covar, g), h_obs


class InterpreterEDncem(estimators.EstimatorEDncem, InterpreterGraph):
    """Inherits all relevant functions specific to EstimatorInteractions estimators."""

    def __init__(self):
        """Initialize InterepreterEDncem."""
        super().__init__()

    def target_cell_saliencies(
        self,
        target_cell_type: str,
        drop_columns: Optional[list] = None,
        dop_images: Optional[list] = None,
        partition: str = "test",
        multicolumns: Optional[list] = None,
    ):
        """Compute target cell saliencies.

        Args:
            target_cell_type (str): Target cell type.
            drop_columns: List of columns to drop, i.e. neighbouring cell types.
            dop_images: List of images to drop.
            partition (str): All, train, test or val partition. Default is test.
            multicolumns: List of items in multiindex annotation.
        Returns:
            Dataframe of target cell saliencies in partition.
        """
        target_cell_idx = list(self.node_type_names.values()).index(target_cell_type)

        if partition == "test":
            img_keys = self.img_keys_test
            node_idxs = self.nodes_idx_test
        elif partition == "train":
            img_keys = self.img_keys_train
            node_idxs = self.nodes_idx_train
        elif partition == "val":
            img_keys = self.img_keys_eval
            node_idxs = self.nodes_idx_eval
        elif partition == "all":
            img_keys = self.img_keys_all
            node_idxs = self.nodes_idx_all
        else:
            print("unknown partition selected, saliencies are computed on test partition")
            img_keys = self.img_keys_test
            node_idxs = self.nodes_idx_test

        img_saliency = []
        keys = []
        with tqdm(total=len(img_keys)) as pbar:
            for key in img_keys:
                idx = {key: node_idxs[key]}
                ds = self._get_dataset(
                    image_keys=[key],
                    nodes_idx=idx,
                    batch_size=1,
                    shuffle_buffer_size=1,
                    train=False,
                    seed=None,
                    reinit_n_eval=1,
                )
                # inputs extracted for plotting aggregation
                saliencies = []
                h_1 = []
                h_0 = []
                h_0_full = []
                a = []
                for _step, (x_batch, _y_batch) in enumerate(ds):
                    h_1_batch = x_batch[0].numpy().squeeze()
                    h_0_batch = x_batch[2].numpy().squeeze()  # 1 x 1 x node_types
                    h_0_full_batch = x_batch[3]  # 1 x max_nodes x node_types
                    a_batch = x_batch[4]  # 1 x 1 x max_nodes

                    if h_0_batch[target_cell_idx] == 1.0:
                        h_1.append(h_1_batch)
                        h_0.append(h_0_batch)
                        h_0_full.append(h_0_full_batch.numpy().squeeze())
                        a.append(
                            sparse.csr_matrix(
                                (
                                    a_batch.values.numpy(),
                                    (a_batch.indices.numpy()[:, 1], a_batch.indices.numpy()[:, 2]),
                                ),
                                shape=a_batch.dense_shape.numpy()[1:],
                            ).toarray()
                        )

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
                            absolute_saliencies=False,
                        )
                        saliencies.append(grads)
                if len(saliencies) == 0:
                    continue
                saliencies = np.concatenate(saliencies, axis=0)
                saliencies = np.sum(saliencies, axis=0)
                neighbourhood = self._neighbourhood_frequencies(a=a, h_0_full=h_0_full, discretize_adjacency=True)
                neighbourhood = np.sum(np.array(neighbourhood), axis=0)
                img_saliency.append(saliencies / neighbourhood)
                keys.append(key)
                pbar.update(1)
        columns = (
            [(x.replace("_", " ").split()[0], x.replace("_", " ").split()[1]) for x in keys] if multicolumns else keys
        )

        df = pd.DataFrame(
            np.concatenate(np.expand_dims(img_saliency, axis=0), axis=1).T,
            columns=columns,
            index=list(self.node_type_names.values()),
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=multicolumns) if multicolumns else df.columns

        df = df.reindex(sorted(df.columns), axis=1)

        if drop_columns:
            df = df.drop(drop_columns)
        if dop_images:
            df = df.drop(dop_images, axis=1, level=1)

        return df

    @staticmethod
    def plot_target_cell_saliencies(
        saliencies,
        multiindex: bool = True,
        fontsize: Optional[int] = None,
        figsize: Tuple[float, float] = (30.0, 7.0),
        width_ratios: Optional[list] = None,
        save: Optional[str] = None,
        suffix: str = "_imagewise_target_cell_saliencies.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        """Plot target cell saliencies from dataframe.

        Args:
            saliencies: Precomputed saliencies from 'target_cell_saliencies'.
            multiindex (bool): Whether saliencies have multiindex.
            fontsize (int): Fontsize.
            figsize (Tuple): Figsize.
            width_ratios (list): Width ratios.
            save (str): Path to save directory.
            suffix (str): Saving suffix.
            show (bool): Whether to show figure.
            return_axs (bool): Whether to return axes.
        """
        if width_ratios is None:
            width_ratios = [5, 1]
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        plt.rcParams["axes.grid"] = False
        plt.ioff()
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, gridspec_kw={"width_ratios": width_ratios})
        sns.heatmap(saliencies, cmap="seismic", ax=ax[0], center=0)
        ax[0].set_xlabel("")

        if multiindex:
            xlabel_mapping = OrderedDict()
            for index1, index2 in saliencies.columns:
                xlabel_mapping.setdefault(index1, [])
                xlabel_mapping[index1].append(index2.replace("slice", ""))

            hline = []
            new_xlabels = []
            for _index1, index2_list in xlabel_mapping.items():
                index2_list[0] = "{}".format(index2_list[0])
                new_xlabels.extend(index2_list)

                if hline:
                    hline.append(len(index2_list) + hline[-1])
                else:
                    hline.append(len(index2_list))
            ax[0].set_xticklabels(new_xlabels, rotation=90)

        sns.boxplot(data=saliencies.T, ax=ax[1], orient="h", color="steelblue", showfliers=True)
        ax[1].set_yticklabels("")

        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)
        if show:
            plt.show()
        plt.close(fig)
        plt.ion()
        return ax if return_axs else None


class InterpreterCVAEncem(estimators.EstimatorCVAEncem, InterpreterGraph):
    """Inherits all relevant functions specific to EstimatorInteractions estimators."""

    def __init__(self):
        """Initialize InterepreterCVAEncem."""
        super().__init__()

    def compute_latent_space_cluster_enrichment(
        self,
        image_key,
        target_cell_type: str,
        n_neighbors: int = 15,
        n_pcs: Optional[int] = None,
        filter_titles: Optional[List[str]] = None,
        clip_pvalues: Optional[int] = -5,
    ):
        """Compute latent space cluster enrichment.

        Args:
            image_key: Image key.
            target_cell_type (str): Target cell type.
            n_neighbors (int): The size of local neighborhood (in terms of number of neighboring data points) used
                for manifold approximation.
            n_pcs (int): Use this many PCs.
            filter_titles (list): Filter titles.
            clip_pvalues (int): Clip p-values by values.

        Returns:
            (Tuple): Copy of latent space AnnData object, dataframe of log10 p-values, dataframe of fold_change
        """
        node_type_names = list(self.data.celldata.uns["node_type_names"].values())
        ds = self._get_dataset(
            image_keys=[image_key],
            nodes_idx={image_key: self.nodes_idx_all[image_key]},
            batch_size=1,
            shuffle_buffer_size=1,
            seed=None,
            train=False,
            reinit_n_eval=1,
        )

        cond_encoder = tf.keras.Model(
            self.reinit_model.training_model.input, self.reinit_model.training_model.get_layer("cond_encoder").output
        )
        cond_latent_z_mean = []
        h_0_full = []
        h_0 = []
        a = []
        for _step, (x_batch, _y_batch) in enumerate(ds):
            h_1_batch, sf_batch, h_0_batch, h_0_full_batch, a_batch, a_full_batch, node_covar_batch, g_batch = x_batch

            cond_z, cond_z_mean, cond_z_log = cond_encoder(x_batch)
            cond_latent_z_mean.append(cond_z_mean.numpy())

            h_0.append(h_0_batch.numpy())
            h_0_full.append(h_0_full_batch.numpy().squeeze())
            a.append(
                sparse.csr_matrix(
                    (a_batch.values.numpy(), (a_batch.indices.numpy()[:, 1], a_batch.indices.numpy()[:, 2])),
                    shape=a_batch.dense_shape.numpy()[1:],
                ).toarray()
            )

        cond_latent_z_mean = np.concatenate(cond_latent_z_mean, axis=0)
        source_type = self._neighbourhood_frequencies(a=a, h_0_full=h_0_full, discretize_adjacency=True)
        source_type = (
            pd.DataFrame((source_type > 0).astype(str), columns=node_type_names)
            .replace({"True": "in neighbourhood", "False": "not in neighbourhood"}, regex=True)
            .astype("category")
        )
        h_0 = pd.DataFrame(np.concatenate(h_0, axis=0).squeeze(), columns=node_type_names)
        target_type = pd.DataFrame(np.array(h_0.idxmax(axis=1)), columns=["target_cell"])

        metadata = pd.concat([target_type, source_type], axis=1)
        cond_adata = AnnData(cond_latent_z_mean, obs=metadata)
        sc.pp.neighbors(cond_adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.louvain(cond_adata)
        sc.tl.umap(cond_adata)

        for x in node_type_names:
            cond_adata.uns[f"{x}_colors"] = ["darkgreen", "lightgrey"]

        cond_adata.obs["substates"] = target_cell_type + " " + cond_adata.obs.louvain.astype(str)
        cond_adata.obs["substates"] = cond_adata.obs["substates"].astype("category")
        print("n cells: ", cond_adata.shape[0])
        substate_counts = cond_adata.obs["substates"].value_counts()
        print(substate_counts)

        one_hot = pd.get_dummies(cond_adata.obs.louvain, dtype=np.bool)
        # Join the encoded df
        df = cond_adata.obs.join(one_hot)

        distinct_louvain = len(np.unique(cond_adata.obs.louvain))
        pval_source_type = []
        for st in node_type_names:
            pval_cluster = []
            for j in range(distinct_louvain):
                crosstab = np.array(pd.crosstab(df[f"{st}"], df[str(j)]))
                if crosstab.shape[0] < 2:
                    crosstab = np.vstack([crosstab, [0, 0]])
                oddsratio, pvalue = stats.fisher_exact(crosstab)
                pvalue = correct(np.array([pvalue]))
                pval_cluster.append(pvalue)
            pval_source_type.append(pval_cluster)

        columns = [f"{target_cell_type} {x}" for x in np.unique(cond_adata.obs.louvain)]
        pval = pd.DataFrame(np.array(pval_source_type).squeeze(), index=node_type_names, columns=columns)
        log_pval = np.log10(pval)

        if filter_titles:
            log_pval = log_pval.sort_values(columns, ascending=True).filter(items=filter_titles, axis=0)
        if clip_pvalues:
            log_pval[log_pval < clip_pvalues] = clip_pvalues

        fold_change_df = cond_adata.obs[["target_cell", "substates"] + node_type_names]
        counts = pd.pivot_table(
            fold_change_df.replace({"in neighbourhood": 1, "not in neighbourhood": 0}),
            index=["substates"],
            aggfunc=np.sum,
            margins=True,
        ).T

        fold_change = counts.loc[:, columns].div(np.array(substate_counts), axis=1)
        fold_change = fold_change.subtract(np.array(counts["All"] / cond_adata.shape[0]), axis=0)

        if filter_titles:
            fold_change = fold_change.fillna(0).filter(items=filter_titles, axis=0)
        return cond_adata.copy(), log_pval, fold_change


class InterpreterNoGraph(estimators.EstimatorNoGraph, InterpreterBase):
    """Inherits all relevant functions specific to EstimatorEDncem estimators and InterpreterBase."""

    def __init__(self):
        """Initialize InterepreterNoGraph.."""
        super().__init__()

    def _get_np_data(
        self, image_keys: Union[np.ndarray, str], nodes_idx: Union[dict, str]
    ) -> Tuple[Tuple[list, list, list, list], list]:
        """Collects numpy objects from tensorflow dataset.

        Args:
            image_keys: Observation images indices.
            nodes_idx: Observation nodes indices.

        Returns:
            Tuple of list of raw data, one list entry per image / graph
        """
        if isinstance(image_keys, (int, np.int32, np.int64)):
            image_keys = [image_keys]
        ds = self._get_dataset(
            image_keys=image_keys, nodes_idx=nodes_idx, batch_size=1, shuffle_buffer_size=1, train=False, seed=None
        )
        # Loop over sub-selected data set and sum gradients across all selected observations.
        h_1 = []
        sf = []
        node_covar = []
        g = []
        h_obs = []

        for _step, (x_batch, y_batch) in enumerate(ds):
            h_1_batch, sf_batch, node_covar_batch, g_batch = x_batch
            h_1.append(h_1_batch.numpy().squeeze())
            sf.append(sf_batch.numpy().squeeze())
            node_covar.append(node_covar_batch.numpy().squeeze())
            g.append(g_batch.numpy().squeeze())
            h_obs.append(y_batch[0].numpy().squeeze())
        return (h_1, sf, node_covar, g), h_obs


class InterpreterDeconvolution(estimators.EstimatorDeconvolution, InterpreterInteraction):
    """Inherits all relevant functions specific to EstimatorDeconvolution estimators."""

    def __init__(self):
        """Initialize InterepreterInteraction."""
        super().__init__()

    def _get_np_data(
        self, image_keys: Union[np.ndarray, str], nodes_idx: Union[dict, str]
    ) -> Tuple[Tuple[list, List[csr_matrix], list, list, list], list]:
        """Collects numpy objects from tensorflow dataset.

        Args:
            image_keys: Observation images indices.
            nodes_idx: Observation nodes indices.

        Returns:
            Tuple of list of raw data, one list entry per image / graph
        """
        if isinstance(image_keys, (int, np.int32, np.int64)):
            image_keys = [image_keys]
        ds = self._get_dataset(
            image_keys=image_keys, nodes_idx=nodes_idx, batch_size=1, shuffle_buffer_size=1, train=False, seed=None
        )
        # Loop over sub-selected data set and sum gradients across all selected observations.
        target = []
        interactions = []
        sf = []
        node_covar = []
        g = []
        h_obs = []

        for _step, (x_batch, y_batch) in enumerate(ds):
            target_batch, interaction_batch, sf_batch, node_covar_batch, g_batch = x_batch
            target.append(target_batch.numpy().squeeze())
            interactions.append(
                csr_matrix(
                    (
                        interaction_batch.values.numpy(),
                        (interaction_batch.indices.numpy()[:, 1], interaction_batch.indices.numpy()[:, 2]),
                    ),
                    shape=interaction_batch.dense_shape.numpy()[1:],
                ).todense()
            )
            sf.append(sf_batch.numpy().squeeze())
            node_covar.append(node_covar_batch.numpy().squeeze())
            g.append(g_batch.numpy().squeeze())
            h_obs.append(y_batch[0].numpy().squeeze())
        return (target, interactions, sf, node_covar, g), h_obs
    
    def get_sender_receiver_effects(
        self,
        params_type: str = 'ols',
        significance_threshold: float = 0.05
    ):
        data = {
            "target": self.data.celldata.obsm['node_types'], 
            "proportions": self.data.celldata.obsm['proportions']
        }
        target = np.asarray(dmatrix("target-1", data))
        interaction_shape = self.model.training_model.inputs[1].shape
        interactions = np.asarray(dmatrix("target:proportions-1", data))

        y = self.data.celldata.X[self.nodes_idx_all['1'],:]

        print('using ols parameters.')
        if params_type == 'ols':
            x_design = np.concatenate([target, interactions], axis=1)
            ols = ols_fit(x_=x_design, y_=y)
            params = ols.squeeze()
        else:
            params = (
                self.model.training_model.weights[0]
                .numpy()
                .T
            )

        # get inverse fisher information matrix
        print('calculating inv fim.')
        fim_inv = get_fim_inv(x_design, y)

        is_sign, pvalues, qvalues = wald_test(
            params=params, fisher_inv=fim_inv, significance_threshold=significance_threshold
        )
        interaction_shape = self.model.training_model.inputs[1].shape
        # subset to interaction terms
        is_sign = is_sign[self.n_features_0 : interaction_shape[-1] + self.n_features_0, :]
        pvalues = pvalues[self.n_features_0 : interaction_shape[-1] + self.n_features_0, :]
        qvalues = qvalues[self.n_features_0 : interaction_shape[-1] + self.n_features_0, :]

        self.pvalues = np.concatenate(
            np.expand_dims(np.split(pvalues, indices_or_sections=np.sqrt(pvalues.shape[0]), axis=0), axis=0),
            axis=0,
        )
        self.qvalues = np.concatenate(
            np.expand_dims(np.split(qvalues, indices_or_sections=np.sqrt(qvalues.shape[0]), axis=0), axis=0),
            axis=0,
        )
        self.is_sign = np.concatenate(
            np.expand_dims(np.split(is_sign, indices_or_sections=np.sqrt(is_sign.shape[0]), axis=0), axis=0),
            axis=0,
        )

        interaction_params = params[:, self.n_features_0 : interaction_shape[-1] + self.n_features_0]
        self.fold_change = np.concatenate(
            np.expand_dims(np.split(interaction_params.T, indices_or_sections=np.sqrt(interaction_params.T.shape[0]), axis=0), axis=0),
            axis=0,
        )

    def interaction_significance(
        self,
        image_keys,
        nodes_idx,
        significance_threshold: float = 0.01,
    ):
        """Compute interaction parameter significance.

        Args:
            image_keys: Image keys.
            nodes_idx: Node indices.
            significance_threshold (float): Significance threshold for p-values.

        Returns:
            (Tuple): Interaction parameters of model, aaray of boolean significance, significance values.
        """
        ds = self._get_dataset(
            image_keys=image_keys, nodes_idx=nodes_idx, batch_size=1, shuffle_buffer_size=1, train=False, seed=None
        )
        interaction_shape = self.model.training_model.inputs[1].shape
        interaction_params = (
            self.model.training_model.weights[0]
            .numpy()[self.n_features_0 : interaction_shape[-1] + self.n_features_0, :]
            .T
        )

        interaction_params = np.concatenate(
            np.expand_dims(np.split(interaction_params, indices_or_sections=self.n_features_0, axis=1), axis=-1),
            axis=-1,
        )

        interactions = []
        y = []
        for _step, (x_batch, y_batch) in enumerate(ds):
            target_batch, interactions_batch, sf_batch, node_covar_batch, g_batch = x_batch
            interactions.append(
                sparse.csr_matrix(
                    (
                        interactions_batch.values.numpy(),
                        (interactions_batch.indices.numpy()[:, 1], interactions_batch.indices.numpy()[:, 2]),
                    ),
                    shape=interactions_batch.dense_shape.numpy()[1:],
                ).todense()
            )
            y.append(y_batch[0].numpy().squeeze())

        interactions = np.concatenate(interactions, axis=0)
        y = np.concatenate(y, axis=0)

        interactions = np.concatenate(
            np.expand_dims(np.split(interactions, indices_or_sections=self.n_features_0, axis=1), axis=-1), axis=-1
        )

        fisher_inv = []
        for i in range(interactions.shape[1]):
            target_x = interactions[:, i, :]
            target_fisher_inv = np.divide(
                np.expand_dims(np.matmul(target_x.T, target_x), axis=0),
                np.expand_dims(np.expand_dims(np.var(y, axis=0), axis=-1), axis=-1),
            )
            fisher_inv.append(target_fisher_inv)

        bool_significance, significance = wald_test(
            params=interaction_params, fisher_inv=fisher_inv, significance_threshold=significance_threshold
        )
        return interaction_params, bool_significance, significance
