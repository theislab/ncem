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

import ncem.estimators as estimators
import ncem.models as models
import ncem.train as train
from ncem.utils.wald_test import wald_test


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

    def get_data_again(self, data_path: str, data_origin: str):
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
            use_covar_node_position=self.gscontainer_runparams["use_covar_node_position"],
            use_covar_node_label=self.gscontainer_runparams["use_covar_node_label"],
            use_covar_graph_covar=self.gscontainer_runparams["use_covar_graph_covar"],
            domain_type=self.gscontainer_runparams["domain_type"],
        )
        self.data_path = data_path
        self.n_eval_nodes_per_graph = self.gscontainer_runparams["n_eval_nodes_per_graph"]
        self.model_class = self.gscontainer_runparams["model_class"]
        self.data_set = self.gscontainer_runparams["data_set"]
        self.radius = int(self.gscontainer_runparams["max_dist"])
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
            parameters=interaction_params, fisher_inv=fisher_inv, significance_threshold=significance_threshold
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
