import abc
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ncem.utils.losses import GaussianLoss, KLLoss, NegBinLoss
from ncem.utils.metrics import (custom_kl, custom_mae, custom_mean_sd,
                                custom_mse, custom_mse_scaled,
                                gaussian_reconstruction_loss, logp1_custom_mse,
                                logp1_r_squared, logp1_r_squared_linreg,
                                nb_reconstruction_loss, r_squared,
                                r_squared_linreg)


def transfer_layers(model1, model2):
    """
    Transfer layer weights from model 1 to model 2.

    :param model1: Input model.
    :param model2: Output model.
    """
    layer_names_model1 = [x.name for x in model1.layers]
    layer_names_model2 = [x.name for x in model2.layers]
    layers_updated = []
    layer_not_updated = set(layer_names_model2)
    for x in layer_names_model1:
        w = model1.get_layer(name=x).get_weights()
        if x in layer_names_model2:
            # Only update layers with parameters:
            if len(w) > 0:
                model2.get_layer(x).set_weights(w)
                layers_updated.append(x)
                layer_not_updated = layer_not_updated.difference({x})
    print(f"updated layers: {layers_updated}")
    print(f"did not update layers: {layer_not_updated}")


class Estimator:
    """Estimator class for models.

    Contains all necessary methods for data loading, model initialization, training, evaluation and prediction.
    """

    img_to_patient_dict: Dict[str, str]
    complete_img_keys: List[str]

    a: dict  # dict of adjacency matrices of shape (max_nodes, max_nodes)
    h_0: Dict[str, np.ndarray]  # dict of adjacency matrices of shape (max_nodes, n_features_0)
    h_1: Dict[str, np.ndarray]  # dict of adjacency matrices of shape (max_nodes, n_features_1)
    size_factors: Dict[str, np.ndarray]
    graph_covar: Dict[str, np.ndarray]
    node_covar: Dict[str, np.ndarray]
    domains: Dict[str, np.ndarray]

    covar_selection: Union[List[str], Tuple[str], None]

    node_types: Dict[str, np.ndarray]
    node_type_names: Dict[str, str]
    graph_covar_names: Dict[str, List[str]]
    node_feature_names: List[str]

    n_features_type: int
    n_features_standard: int
    n_features_0: int
    n_features_1: int
    n_graph_covariates: int
    n_node_covariates: int
    n_domains: int
    max_nodes: int
    n_eval_nodes_per_graph: int

    vi_model: bool
    log_transform: bool
    model_type: str
    adj_type: str
    cond_type: str
    cond_depth: int
    output_layer: str

    img_keys_test = list
    img_keys_eval = list
    img_keys_train = list

    nodes_idx_test = Dict[str, list]
    nodes_idx_eval = Dict[str, list]
    nodes_idx_train = Dict[str, list]

    steps_per_epoch: int
    validation_steps: int

    def __init__(self):
        """Initialize Estimator class."""
        self.model = None
        self.loss = []
        self.metrics = []
        self.optimizer = None
        self.beta = None
        self.max_beta = None
        self.pre_warm_up = None
        self.train_hyperparam = {}
        self.history = {}
        self.pretrain_history = {}

        self.train_dataset = None
        self.eval_dataset = None

    def _load_data(
        self,
        data_origin: str,
        data_path: str,
        radius: Optional[int] = None,
        n_rings: int = 1,
        label_selection: Optional[List[str]] = None,
        n_top_genes: Optional[int] = None
    ):
        """Initialize a DataLoader object.

        Parameters
        ----------
        data_origin : str
            Data origin.
        data_path : str
            Data path.
        radius : int
            Radius.
        label_selection : list, optional
            Label selection.
        n_top_genes: int, optional
            N top genes for highly variable gene selection.

        Raises
        ------
        ValueError
            If `data_origin` not recognized.
        """
        coord_type = 'generic'
        self.targeted_assay = True
        if data_origin.startswith("zhang"):
            from ncem.data import DataLoaderZhang as DataLoader

            self.undefined_node_types = ["other"]
        elif data_origin.startswith("jarosch"):
            from ncem.data import DataLoaderJarosch as DataLoader

            self.undefined_node_types = None
        elif data_origin.startswith("hartmann"):
            from ncem.data import DataLoaderHartmann as DataLoader

            self.undefined_node_types = None
        elif data_origin == "pascualreguant":
            from ncem.data import DataLoaderPascualReguant as DataLoader

            self.undefined_node_types = ["other"]
        elif data_origin.startswith("schuerch"):
            from ncem.data import DataLoaderSchuerch as DataLoader

            self.undefined_node_types = [
                "dirt",
                "undefined",
                "tumor cells / immune cells",
                "immune cells / vasculature",
            ]
        elif data_origin.startswith('lohoff'):
            from ncem.data import DataLoaderLohoff as DataLoader
            self.undefined_node_types = ['Low quality']
        elif data_origin.startswith("luwt"):
            if data_origin == "luwt_imputation":
                from ncem.data import DataLoaderLuWTimputed as DataLoader
            else:
                from ncem.data import DataLoaderLuWT as DataLoader

            self.undefined_node_types = ['Unknown']
        elif data_origin.startswith("lutet2"):
            from ncem.data import DataLoaderLuTET2 as DataLoader

            self.undefined_node_types = ['Unknown']
        elif data_origin == "10xvisium":
            from ncem.data import DataLoader10xVisiumMouseBrain as DataLoader

            self.undefined_node_types = None
            if n_rings > 1:
                coord_type = 'grid'
            else:
                n_rings = 1
                coord_type = 'generic'
                radius = 0
        elif data_origin == "10xvisium_lymphnode":
            from ncem.data import DataLoader10xLymphnode as DataLoader

            self.undefined_node_types = None
            if n_rings > 1:
                coord_type = 'grid'
            else:
                n_rings = 1
                coord_type = 'generic'
                radius = 0

        elif data_origin.startswith('destvi_lymphnode'):
            self.targeted_assay = False
            from ncem.data import DataLoaderDestViLymphnode as DataLoader

            self.undefined_node_types = None
        elif data_origin.startswith('destvi_mousebrain'):
            self.targeted_assay = False
            from ncem.data import DataLoaderDestViMousebrain as DataLoader

            self.undefined_node_types = None

        elif data_origin.startswith('cell2location_lymphnode'):
            self.targeted_assay = False
            from ncem.data import DataLoaderCell2locationLymphnode as DataLoader

            self.undefined_node_types = None

        elif data_origin == "salasiss":
            from ncem.data import DataLoaderSalasIss as DataLoader

            self.undefined_node_types = None

        self.data = DataLoader(
            data_path, radius=radius, coord_type=coord_type, n_rings=n_rings, label_selection=label_selection,
            n_top_genes=n_top_genes
        )

    def get_data(
        self,
        data_origin: str,
        data_path: str,
        radius: Optional[int],
        n_rings: int = 1,
        graph_covar_selection: Optional[Union[List[str], Tuple[str]]] = None,
        node_label_space_id: str = "type",
        node_feature_space_id: str = "standard",
        use_covar_node_position: bool = False,
        use_covar_node_label: bool = False,
        use_covar_graph_covar: bool = False,
        domain_type: str = "image",
        robustness: Optional[float] = None,
        robustness_seed: int = 1,
        n_top_genes: Optional[int] = None,
        segmentation_robustness: Optional[List[float]] = None,
        resimulate_nodes: bool = False,
        resimulate_nodes_w_depdency: bool = False,
        resimulate_nodes_sparsity_rate: float = 0.5,
    ):
        """Get data used in estimator classes.

        Parameters
        ----------
        data_origin : str
            Data origin.
        data_path : str
            Data path.
        radius : int, optional
            Radius.
        n_rings : int
            Number of rings of neighbors for grid data.
        graph_covar_selection : list, tuple, optional
            Selected graph covariates.
        node_label_space_id : str
            Node label space id.
        node_feature_space_id : str
            Node feature space id.
        use_covar_node_position : bool
            Whether to use node position as covariate.
        use_covar_node_label : bool
            Whether to use node label as covariate.
        use_covar_graph_covar : bool
            Whether to use graph covariates.
        domain_type : str
            Covariate that is used as domain.
        robustness : float, optional
            Optional fraction of images for robustness test.
        robustness_seed: int
            Seed for robustness analysis
        n_top_genes: int, optional
            N top genes for highly variable gene selection.
        segmentation_robustness: list, optional
            Parameters for segmentation robustness fit, float for fraction of nodes and float for signal overflow.
        Raises
        ------
        ValueError
            If sub-selected covar_selection could not be found, `node_label_space_id` or `node_feature_space_id`
            not recognized
        """
        if self.adj_type is None:
            raise ValueError("set adj_type by init_estim() first")
        if graph_covar_selection is None:
            graph_covar_selection = []
        labels_to_load = graph_covar_selection
        self._load_data(
            data_origin=data_origin,
            data_path=data_path,
            radius=radius,
            n_rings=n_rings,
            label_selection=labels_to_load,
            n_top_genes=n_top_genes
        )
        if robustness:
            np.random.seed(robustness_seed)
            n_images = np.int(len(self.data.img_celldata) * robustness)
            print(n_images)
            image_keys = list(np.random.choice(
                a=list(self.data.img_celldata.keys()),
                size=n_images,
                replace=False,
            ))
            self.data.img_celldata = {k: self.data.img_celldata[k] for k in image_keys}
            metadata = self.data.celldata.uns["metadata"]

            self.data.celldata = self.data.celldata[self.data.celldata.obs[metadata['image_col']].isin(image_keys)]

            print(
                "\nAttention: Running robustness model with a fraction %f images, so [%i] images. \n"
                "\nThis also adjusts celldata and img_celldata."
                % (
                    robustness,
                    n_images,
                )
            )
        if segmentation_robustness:
            node_fraction = segmentation_robustness[0]
            overflow_fraction = segmentation_robustness[1]
            total_size = np.int(self.data.celldata.shape[0] * node_fraction)

            for key, ad in self.data.img_celldata.items():
                size = np.int(ad.shape[0] * node_fraction)
                random_indices = np.random.choice(ad.shape[0], size=size, replace=False)
                a = ad.obsp['adjacency_matrix_connectivities'].toarray()
                err_ad = ad.copy()
                for idx in random_indices:
                    adj = a[idx, :]
                    neigh_idx = np.random.choice(np.where(adj == 1.)[0], size=1, replace=False)
                    err_ad.X[idx, :] = ad.X[idx, :] + overflow_fraction * ad.X[neigh_idx, :]
                    err_ad.X[neigh_idx, :] = (1. - overflow_fraction) * ad.X[neigh_idx, :]
                self.data.img_celldata[key] = err_ad

            print(
                "\nAttention: Running segmentation robustness model on %f of all nodes, so [%i] nodes. \n"
                "\nSignal overflow is set to %f . This adjusts img_celldata, celldata remains unchanged.\n"
                % (
                    node_fraction,
                    total_size,
                    overflow_fraction
                )
            )
        self.simulation = False
        if resimulate_nodes:
            self.simulation = True
            n_target_cell_types = 2 if resimulate_nodes_w_depdency else 1
            dependencies_per_type = 1

            # Create map from real cell types to simulated ones (can be coarser):
            found_all_types = False
            futile_counter = 0
            node_type_map_idx = None
            while not found_all_types:
                node_type_map_idx = np.array([
                    np.random.randint(low=0, high=n_target_cell_types)
                    for _ in self.data.celldata.uns["node_type_names"].keys()
                ])
                futile_counter += 1
                if np.all([x in node_type_map_idx for x in range(n_target_cell_types)]):
                    found_all_types = True
                if futile_counter > 100:
                    raise ValueError("did not manage to sample all target cell types")
            node_type_names = dict([
                (x, "sim_" + str(y))
                for x, y in zip(self.data.celldata.uns["node_type_names"].keys(), node_type_map_idx)
            ])
            self.data.celldata.uns["node_type_names"] = node_type_names

            nfeatures = self.data.img_celldata[list(self.data.img_celldata.keys())[0]].n_vars
            # Mean effect per simulated cell types:
            effect_ct = np.random.uniform(low=0., high=10., size=(n_target_cell_types, nfeatures))
            # Create dependency structure of cell types.
            # Base line dependency structure with all dependencies as 0.
            cov_ct = np.zeros((n_target_cell_types, n_target_cell_types))
            if resimulate_nodes_w_depdency:
                # Add dependencies_per_type for each cell type:
                for i in range(n_target_cell_types):
                    # Sample desired dependencies from non-self cell types:
                    js = np.random.choice(a=[ii for ii in range(n_target_cell_types) if i != ii],
                                          size=dependencies_per_type, replace=False)
                    cov_ct[i, js] = 1.
                # Pairwise dependencies: Effect (self cell type, neighbor cell type, feature)
                effect_neighbors = np.random.uniform(low=4., high=6.,
                                                     size=(n_target_cell_types, n_target_cell_types, nfeatures))
                # Simulate sparse effects:
                sparsity_rate = resimulate_nodes_sparsity_rate  # fraction of zero effects
                effect_neighbors[np.random.binomial(n=1, p=sparsity_rate, size=effect_neighbors.shape) == 1.] = 0.
            else:
                effect_neighbors = np.zeros((n_target_cell_types, n_target_cell_types, nfeatures))
            sigma_sq = 1.

            self._simulation_parameters = {
                "effect_ct": effect_ct,
                "cov_ct": cov_ct,
                "effect_neighbors": effect_neighbors,
                "sigma_sq": sigma_sq,
                "adatas": {}
            }
            for key, ad in self.data.img_celldata.items():
                adj = ad.obsp['adjacency_matrix_connectivities'].toarray()
                sim_ad = ad.copy()
                nobs = sim_ad.n_obs
                # Assign all cells from old cell type sets to corresponding new cell types, assumes one hot encoding.
                sim_ad.obsm["node_types"] = np.concatenate([
                    np.expand_dims(
                        np.max(sim_ad.obsm["node_types"][:, np.where(node_type_map_idx == i)[0]], axis=1),
                        axis=1
                    )
                    for i in range(n_target_cell_types)
                ], axis=1)
                assert np.all(sim_ad.obsm["node_types"].sum(axis=1) == 1.)
                # Simulate count matrix:
                dmat_ct = sim_ad.obsm["node_types"]
                loc_neighbors = np.zeros((nobs, nfeatures))
                for i in range(nobs):
                    ct = np.where(dmat_ct[i, :] == 1.)[0]
                    ct = ct[0]  # flatten list of length 1
                    dmat_neighhors_i = np.zeros((1, n_target_cell_types,))
                    if resimulate_nodes_w_depdency:
                        for j in np.where(np.asarray(adj[i, :]).flatten() > 0)[0]:
                            ct_j = np.where(dmat_ct[j, :] == 1.)[0]
                            ct_j = ct_j[0]  # flatten list of length 1
                            dmat_neighhors_i[0, ct_j] = 1.
                    loc_neighbors[i, :] = np.matmul(dmat_neighhors_i, effect_neighbors[ct])[0]
                loc = np.matmul(dmat_ct, effect_ct) + loc_neighbors
                sim_ad.X = np.random.normal(loc=loc, scale=sigma_sq)
                self.data.img_celldata[key] = sim_ad
                # Record simulation:
                self._simulation_parameters["adatas"][key] = {
                    "adj": adj,
                    "ct": sim_ad.obsm["node_types"],
                    "x": sim_ad.X,
                }

            print(
                "\nAttention: Running simulation-based expression augmentation. \n"
                "\nThis adjusts img_celldata, celldata remains unchanged.\n"
            )

        # Validate graph-wise covariate selection:
        if len(graph_covar_selection) > 0:
            if (
                np.sum(
                    [
                        x not in self.data.celldata.uns["graph_covariates"]["label_selection"]
                        for x in graph_covar_selection
                    ]
                )
                > 0
            ):
                raise ValueError(
                    "could not find some sub-selected covar_selection %s in %s"
                    % (str(graph_covar_selection), str(self.data.celldata.uns["graph_covariates"]["label_selection"]))
                )
        self.img_to_patient_dict = self.data.celldata.uns["img_to_patient_dict"]
        self.complete_img_keys = list(self.data.img_celldata.keys())

        self.a = {k: adata.obsp["adjacency_matrix_connectivities"] for k, adata in self.data.img_celldata.items()}
        if self.adj_type == "scaled":
            self.a = self.data._transform_all_a(self.a)
        if node_label_space_id == "standard":
            self.h_0 = {k: adata.X for k, adata in self.data.img_celldata.items()}
        elif node_label_space_id == "type":
            self.h_0 = {k: adata.obsm["node_types"] for k, adata in self.data.img_celldata.items()}
        elif node_label_space_id == 'proportions':
            self.h_0 = {k: adata.obsm["proportions"] for k, adata in self.data.img_celldata.items()}
        else:
            raise ValueError("node_label_space_id %s not recognized" % node_label_space_id)
        if node_feature_space_id == "standard":
            self.h_1 = {k: adata.X for k, adata in self.data.img_celldata.items()}
        elif node_feature_space_id == "type":
            self.h_1 = {k: adata.obsm["node_types"] for k, adata in self.data.img_celldata.items()}
        else:
            raise ValueError("node_feature_space_id %s not recognized" % node_feature_space_id)
        self.node_types = {k: adata.obsm["node_types"] for k, adata in self.data.img_celldata.items()}
        self.node_type_names = self.data.celldata.uns["node_type_names"]
        self.n_features_type = list(self.node_types.values())[0].shape[1]
        self.n_features_standard = self.data.celldata.shape[1]
        self.node_feature_names = list(self.data.celldata.var_names)
        self.size_factors = self.data.size_factors()

        # Add covariates:
        # Add graph-level covariate information
        self.covar_selection = graph_covar_selection
        self.graph_covar_names = self.data.celldata.uns["graph_covariates"]["label_names"]

        # Split loaded graph-wise covariates into labels (output, Y) and covariates / features (input, C)
        if len(graph_covar_selection) > 0:
            self.graph_covar = {  # Single 1D array per observation: concatenate all covariates!
                k: np.concatenate(
                    [adata.uns["graph_covariates"]["label_tensors"][kk] for kk in self.covar_selection], axis=0
                )
                for k, adata in self.data.img_celldata.items()
            }
            # Replace masked entries (np.nan) by zeros: (masking can be handled properly in output but not here):
            for k, v in self.graph_covar.items():
                if np.any(np.isnan(v)):
                    self.graph_covar[k][np.isnan(v)] = 0.0
        else:
            # Create empty covariate arrays:
            self.graph_covar = {k: np.array([], ndmin=1) for k, adata in self.data.img_celldata.items()}
        # Add node-level conditional information
        self.node_covar = {k: np.empty((adata.shape[0], 0)) for k, adata in self.data.img_celldata.items()}
        # Cell position in image:
        if use_covar_node_position:
            for k in self.complete_img_keys:
                self.node_covar[k] = np.append(self.node_covar[k], self.data.img_celldata[k].obsm["spatial"], axis=1)
            print("Position_matrix added to categorical predictor matrix")
        # Add graph-level covariates to node covariates:
        if use_covar_graph_covar:
            for k in self.complete_img_keys:
                # Broadcast graph-level covariate to nodes:
                c = np.repeat(self.graph_covar[k][np.newaxis, :], self.data.img_celldata[k].shape[0], axis=0)
                self.node_covar[k] = np.append(self.node_covar[k], c, axis=1)
            print("Node_covar_selection broadcasted to categorical predictor matrix")
        # Add node
        if use_covar_node_label:
            for k in self.complete_img_keys:
                node_types = self.data.img_celldata[k].obsm["node_types"]
                self.node_covar[k] = np.append(self.node_covar[k], node_types, axis=1)
            print("Node_type added to categorical predictor matrix")

        # Set selection-specific tensor dimensions:
        self.n_features_0 = list(self.h_0.values())[0].shape[1]
        self.n_features_1 = list(self.h_1.values())[0].shape[1]
        self.n_graph_covariates = list(self.graph_covar.values())[0].shape[0]
        self.n_node_covariates = list(self.node_covar.values())[0].shape[1]
        self.max_nodes = max([self.a[i].shape[0] for i in self.complete_img_keys])

        # Define domains
        if domain_type == "image":
            self.domains = {key: i for i, key in enumerate(self.complete_img_keys)}
        elif domain_type == "patient":
            self.domains = {
                key: list(self.patient_ids_unique).index(self.img_to_patient_dict[key])
                for i, key in enumerate(self.complete_img_keys)
            }
        else:
            assert False
        self.n_domains = len(np.unique(list(self.domains.values())))

        if self.targeted_assay:
            self.proportions = None
        else:
            self.proportions = {k: adata.obsm["proportions"] for k, adata in self.data.img_celldata.items()}

        # Report summary statistics of loaded graph:
        print(
            "Mean of mean node degree per images across images: %f"
            % np.mean([np.mean(v.sum(axis=1)) for k, v in self.a.items()])
        )

    @abc.abstractmethod
    def _get_dataset(
        self,
        image_keys: List[str],
        nodes_idx: Dict[str, np.ndarray],
        batch_size: int,
        shuffle_buffer_size: Optional[int],
        train: bool,
        seed: Optional[int],
        prefetch: int = 100,
        reinit_n_eval: Optional[int] = None,
    ):
        """Prepare a dataset.

        Parameters
        ----------
        image_keys : np.array
            Image keys in partition.
        nodes_idx : dict, str
            Dictionary of nodes per image in partition.
        batch_size : int
            Batch size.
        shuffle_buffer_size : int, optional
            Shuffle buffer size.
        train : bool
            Whether dataset is used for training or not (influences shuffling of nodes).
        seed : int, optional
            Random seed.
        prefetch: int
            Prefetch of dataset.
        reinit_n_eval : int, optional
            Used if model is reinitialized to different number of nodes per graph.
        """
        pass

    @abc.abstractmethod
    def _get_resampled_dataset(
        self,
        image_keys: np.ndarray,
        nodes_idx: dict,
        batch_size: int,
        seed: Optional[int] = None,
        prefetch: int = 100,
    ):
        """Evaluate model based on resampled dataset for posterior resampling.

        node_1 + domain_1 -> encoder -> z_1 + domain_2 -> decoder -> reconstruction_2.

        Parameters
        ----------
        image_keys: np.array
            Image keys in partition.
        nodes_idx : dict
            Dictionary of nodes per image in partition.
        batch_size : int
            Batch size.
        seed : int, optional
            Seed.
        prefetch : int
            Prefetch.
        """
        pass

    @abc.abstractmethod
    def init_model(self, **kwargs):
        """Initialize and compiles the model.

        Parameters
        ----------
        kwargs
            Arbitrary keyword arguments.
        """
        pass

    @property
    def patient_ids_bytarget(self) -> np.ndarray:
        """Return patient identifiers by target.

        Returns
        -------
        patient_ids_bytarget
        """
        return np.array([self.img_to_patient_dict[x] for x in self.complete_img_keys])

    @property
    def patient_ids_unique(self) -> np.ndarray:
        """Return unique patient identifiers.

        Returns
        -------
        patient_ids_unique
        """
        return np.unique(self.patient_ids_bytarget)

    @property
    def img_keys_all(self):
        """Return all image keys.

        Returns
        -------
        img_keys_all
        """
        return np.unique(np.concatenate([self.img_keys_train, self.img_keys_eval, self.img_keys_test])).tolist()

    @property
    def nodes_idx_all(self):
        """Return all node indices.

        Returns
        -------
        nodes_idx_all
        """
        return dict(
            [
                (
                    x,
                    np.unique(
                        np.concatenate(
                            [
                                self.nodes_idx_train[x] if x in self.nodes_idx_train.keys() else np.array([]),
                                self.nodes_idx_eval[x] if x in self.nodes_idx_eval.keys() else np.array([]),
                                self.nodes_idx_test[x] if x in self.nodes_idx_test.keys() else np.array([]),
                            ]
                        )
                    ),
                )
                for x in list(self.img_keys_all)
            ]
        )

    @staticmethod
    def _prepare_sf(x):
        """Prepare size factors.

        Parameters
        ----------
        x
            Inout array.

        Returns
        -------
        size_factors

        Raises
        ------
        ValueError
            x.shape > 2
        """
        if len(x.shape) == 2:
            sf = np.asarray(x.sum(axis=1)).flatten()
        elif len(x.shape) == 1:
            sf = np.asarray(x.sum()).flatten()
        else:
            raise ValueError("x.shape > 2")
        sf = sf / np.mean(sf)
        return sf

    def _compile_model(self, optimizer: tf.keras.optimizers.Optimizer, output_layer: str):
        """Compile all necessary models.

        ATTENTION: Decoder compiled with same optimizer instance as training model if an instance is passed!

        Parameters
        ----------
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer to be used for training model (and decoder).
        output_layer : str
            Output layer to be used (e.g. gaussian).

        Raises
        ------
        ValueError
            If `output_layer` is not recognized.
        """
        self.vi_model = False  # variational inference
        if self.model_type in ["cvae", "cvae_ncem"]:
            self.vi_model = True
        enc_dec_model = self.model_type == "cvae" or self.model_type == "cvae_ncem"

        if output_layer in ["gaussian", "gaussian_const_disp", "linear", "linear_const_disp"]:
            reconstruction_loss = GaussianLoss()
            reconstruction_metrics = [
                custom_mae,
                custom_mean_sd,
                custom_mse,
                custom_mse_scaled,
                gaussian_reconstruction_loss,
                r_squared,
                r_squared_linreg,
            ]
        elif output_layer == "nb" or output_layer == "nb_shared_disp" or output_layer == "nb_const_disp":
            reconstruction_loss = NegBinLoss()
            reconstruction_metrics = [
                custom_mae,
                custom_mean_sd,
                nb_reconstruction_loss,
                logp1_custom_mse,
                logp1_r_squared,
                logp1_r_squared_linreg,
            ]
        else:
            raise ValueError("output_layer %s not recognized" % output_layer)
        self.output_layer = output_layer
        if self.vi_model:
            self.loss = [
                reconstruction_loss,
                KLLoss(beta=self.beta, max_beta=self.max_beta, pre_warm_up=self.pre_warm_up),
            ]
            self.metrics = [reconstruction_metrics, [custom_kl]]
        else:
            self.loss = [reconstruction_loss]
            self.metrics = reconstruction_metrics

        self.model.training_model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        # Also compile sampling model / decoder if available:
        if enc_dec_model:
            self.model.decoder.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        if self.vi_model:
            self.model.decoder_sampling.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)

    def _remove_unidentified_nodes(self, node_idx) -> Tuple[int, dict]:
        """Exclude undefined cells from data set.

        Parameters
        ----------
        node_idx
            Data set to remove unidentified nodes from.

        Returns
        -------
        tuple
            number of unidentifed nodes removed, data set with unidentified nodes removed from.
        """
        if self.undefined_node_types is not None:
            # Identify cells with undefined cell_type in all target images
            undefined_label_idx = np.where([x in self.undefined_node_types for x in self.node_type_names.values()])[0]
            # Extract shape and number for print statement
            n_undefined_nodes = np.sum(
                [
                    np.sum(np.any(self.node_types[k][v, :][:, undefined_label_idx] == 1, axis=1))
                    for k, v in node_idx.items()
                ]
            )
            node_idx.update(
                {
                    k: v[
                        np.where(np.logical_not(np.any(self.node_types[k][v, :][:, undefined_label_idx] == 1, axis=1)))[
                            0
                        ]
                    ]
                    for k, v in node_idx.items()
                }
            )
        else:
            n_undefined_nodes = 0
        return n_undefined_nodes, node_idx

    def split_data_given(
        self, img_keys_test, img_keys_train, img_keys_eval, nodes_idx_test, nodes_idx_train, nodes_idx_eval
    ):
        """Split data by given partition.

        Parameters
        ----------
        img_keys_test
            Test image keys.
        img_keys_train
            Train image keys.
        img_keys_eval
            Evaluation image keys.
        nodes_idx_test
            Test node indices.
        nodes_idx_train
            Train node indices.
        nodes_idx_eval
            Evaluation node indices.
        """
        self.img_keys_test = img_keys_test
        self.img_keys_train = img_keys_train
        self.img_keys_eval = img_keys_eval

        self.nodes_idx_test = nodes_idx_test
        self.nodes_idx_train = nodes_idx_train
        self.nodes_idx_eval = nodes_idx_eval

    def split_data_node(self, test_split: float, validation_split: float, seed: int = 1):
        """Split nodes randomly into partitions.

        Parameters
        ----------
        test_split : float
            Fraction of total nodes to be in test set.
        validation_split : float
            Fraction of train-eval nodes to be in validation split.
        seed : int
            Seed for random selection of observations.

        Raises
        ------
        ValueError
            If evaluation or test dataset are empty.
        """
        print(
            "Using split method: node. \n Train-test-validation split is based on total number of nodes "
            "per patients over all images."
        )

        np.random.seed(seed)

        h_nodes_dict = {a: b.shape[0] for a, b in self.h_0.items()}
        all_nodes = sum(h_nodes_dict.values())
        nodes_all_idx = {a: np.arange(0, b) for a, b in h_nodes_dict.items()}

        self.img_keys_test = list(self.complete_img_keys)
        self.img_keys_train = list(self.complete_img_keys)
        self.img_keys_eval = list(self.complete_img_keys)

        n_undefined_nodes, nodes_all_idx = self._remove_unidentified_nodes(node_idx=nodes_all_idx)
        # updating h_nodes_dict to only include the number of identified cells
        h_nodes_dict = {a: b.shape[0] for a, b in nodes_all_idx.items()}

        # Do Test-Val-Train split by patients and put all images for a patient into the chosen partition:
        if isinstance(test_split, str) and test_split == "one sample":
            h_test_dict = {i: 1 for i, b in h_nodes_dict.items()}
        else:
            h_test_dict = {i: round(b * test_split) for i, b in h_nodes_dict.items()}

        self.nodes_idx_test = {
            k: np.random.choice(a=nodes_all_idx[k], size=h_test_dict[k], replace=False) for k in self.h_0.keys()
        }
        nodes_idx_test_shapes = {a: b.shape[0] for a, b in self.nodes_idx_test.items()}
        test_nodes = sum(nodes_idx_test_shapes.values())

        nodes_idx_train_eval = {
            i: np.array([x for x in nodes_all_idx[i] if x not in self.nodes_idx_test[i]]) for i in self.h_0.keys()
        }

        self.nodes_idx_eval = {
            i: np.random.choice(a=nodes_idx_train_eval[i], size=round(len(nodes_idx_train_eval[i]) * validation_split))
            for i in self.h_0.keys()
        }
        nodes_idx_eval_shapes = {a: b.shape[0] for a, b in self.nodes_idx_eval.items()}
        eval_nodes = sum(nodes_idx_eval_shapes.values())

        self.nodes_idx_train = {
            i: np.array([x for x in nodes_idx_train_eval[i] if x not in self.nodes_idx_eval[i]])
            for i in self.h_0.keys()
        }
        nodes_idx_train_shapes = {a: b.shape[0] for a, b in self.nodes_idx_train.items()}
        train_nodes = sum(nodes_idx_train_shapes.values())

        print(
            "\nExcluded %i cells with the following unannotated cell type: [%s] \n"
            "\nWhole dataset: %i cells out of %i images from %i patients."
            % (
                n_undefined_nodes,
                self.undefined_node_types,
                all_nodes,
                len(list(self.complete_img_keys)),
                len(self.patient_ids_unique),
            )
        )
        print(
            "Test dataset: %i cells out of %i images from %i patients."
            % (
                test_nodes,
                len(self.img_keys_test),
                len(self.patient_ids_unique),
            )
        )
        print(
            "Training dataset: %i cells out of %i images from %i patients."
            % (
                train_nodes,
                len(self.img_keys_train),
                len(self.patient_ids_unique),
            )
        )
        print(
            "Validation dataset: %i cells out of %i images from %i patients. \n"
            % (
                eval_nodes,
                len(self.img_keys_eval),
                len(self.patient_ids_unique),
            )
        )

        # Check that none of the train, eval partitions are empty
        if not eval_nodes:
            raise ValueError("The evaluation dataset is empty.")
        if not train_nodes:
            raise ValueError("The train dataset is empty.")

    def split_data_target_cell(self, target_cell: str, test_split: float, validation_split: float, seed: int = 1):
        """Split nodes randomly into partitions.

        Parameters
        ----------
        target_cell : str
            Target cell type.
        test_split : float
            Fraction of total nodes to be in test set.
        validation_split : float
            Fraction of train-eval nodes to be in validation split.
        seed : int
            Seed for random selection of observations.

        Raises
        ------
        ValueError
            If evaluation or test dataset are empty.
        """
        print(
            "Using split method: target cell. \n Train-test-validation split is based on total number of nodes "
            "per patients over all images."
        )

        np.random.seed(seed)

        h_nodes_dict = {a: b.shape[0] for a, b in self.h_0.items()}
        all_nodes = sum(h_nodes_dict.values())
        nodes_all_idx = {a: np.arange(0, b) for a, b in h_nodes_dict.items()}

        target_cell_id = list(self.node_type_names.values()).index(target_cell)

        # Assign images to partitions:
        self.img_keys_train = list(self.complete_img_keys)
        self.img_keys_eval = self.img_keys_train.copy()
        self.img_keys_test = self.img_keys_train.copy()

        n_undefined_nodes, nodes_all_idx = self._remove_unidentified_nodes(node_idx=nodes_all_idx)

        # Dictionary of all nodes within a target cell type
        nodes_all_idx = {k: np.where(self.node_types[k][:, target_cell_id] == 1)[0] for k in self.img_keys_train}
        # updating h_nodes_dict to only include the number of identified cells of specific target cell
        h_nodes_dict = {a: b.shape[0] for a, b in nodes_all_idx.items()}

        nodes_all_idx_shapes = {a: b.shape[0] for a, b in nodes_all_idx.items()}
        target_cell_nodes = sum(nodes_all_idx_shapes.values())

        # Do Test-Val-Train split by patients and put all images for a patient into the chosen partition:
        if isinstance(test_split, str) and test_split == "one sample":
            h_test_dict = {i: 1 for i, b in h_nodes_dict.items()}
        else:
            h_test_dict = {i: round(b * test_split) for i, b in h_nodes_dict.items()}

        # Assign nodes to partitions:
        # Test partition:
        self.nodes_idx_test = {
            k: np.random.choice(a=nodes_all_idx[k], size=h_test_dict[k], replace=False) for k in self.h_0.keys()
        }
        nodes_idx_test_shapes = {a: b.shape[0] for a, b in self.nodes_idx_test.items()}
        test_nodes = sum(nodes_idx_test_shapes.values())

        # Define train-eval partition:
        nodes_idx_train_eval = {
            i: np.array([x for x in nodes_all_idx[i] if x not in self.nodes_idx_test[i]]) for i in self.h_0.keys()
        }
        # Randomly partition train-eval into train and eval:
        self.nodes_idx_eval = {
            i: np.random.choice(a=nodes_idx_train_eval[i], size=round(len(nodes_idx_train_eval[i]) * validation_split))
            for i in self.h_0.keys()
        }
        nodes_idx_eval_shapes = {a: b.shape[0] for a, b in self.nodes_idx_eval.items()}
        eval_nodes = sum(nodes_idx_eval_shapes.values())

        # Assign all nodes in train-eval that are not assigned to eval to train:
        self.nodes_idx_train = {
            i: np.array([x for x in nodes_idx_train_eval[i] if x not in self.nodes_idx_eval[i]])
            for i in self.h_0.keys()
        }
        nodes_idx_train_shapes = {a: b.shape[0] for a, b in self.nodes_idx_train.items()}
        train_nodes = sum(nodes_idx_train_shapes.values())

        print(
            "\nExcluded %i cells with the following unannotated cell type: [%s] \n"
            "\nWhole dataset: %i cells out of %i images from %i patients."
            % (
                n_undefined_nodes,
                self.undefined_node_types,
                all_nodes,
                len(list(self.complete_img_keys)),
                len(self.patient_ids_unique),
            )
        )
        print(
            "\nCell type used for training %s: %i cells out of %i images from %i patients. "
            % (
                target_cell,
                target_cell_nodes,
                len(list(self.complete_img_keys)),
                len(self.patient_ids_unique),
            )
        )
        print(
            "Test dataset: %i cells out of %i images from %i patients. "
            % (
                test_nodes,
                len(self.img_keys_test),
                len(self.patient_ids_unique),
            )
        )
        print(
            "Training dataset: %i cells out of %i images from %i patients."
            % (
                train_nodes,
                len(self.img_keys_train),
                len(self.patient_ids_unique),
            )
        )
        print(
            "Validation dataset: %i cells out of %i images from %i patients.\n"
            % (
                eval_nodes,
                len(self.img_keys_eval),
                len(self.patient_ids_unique),
            )
        )

        # Check that none of the train, eval partitions are empty
        if not eval_nodes:
            raise ValueError("The evaluation dataset is empty.")
        if not train_nodes:
            raise ValueError("The train dataset is empty.")

    def train(
        self,
        epochs: int = 1000,
        epochs_warmup: int = 0,
        max_steps_per_epoch: Optional[int] = 20,
        batch_size: int = 16,
        validation_batch_size: int = 16,
        max_validation_steps: Optional[int] = 10,
        shuffle_buffer_size: Optional[int] = int(1e4),
        patience: int = 20,
        lr_schedule_min_lr: float = 1e-5,
        lr_schedule_factor: float = 0.2,
        lr_schedule_patience: int = 5,
        initial_epoch: int = 0,
        monitor_partition: str = "val",
        monitor_metric: str = "loss",
        log_dir: Optional[str] = None,
        callbacks: Optional[list] = None,
        early_stopping: bool = True,
        reduce_lr_plateau: bool = True,
        pretrain_decoder: bool = False,
        decoder_epochs: int = 1000,
        decoder_patience: int = 20,
        decoder_callbacks: Optional[list] = None,
        aggressive: bool = False,
        aggressive_enc_patience: int = 10,
        aggressive_epochs: int = 5,
        seed: int = 1234,
        **kwargs,
    ):
        """Train model.

        Use validation loss and maximum number of epochs as termination criteria.

        Parameters
        ----------
        epochs : int
            Integer number of times to iterate over the training data arrays. If unspecified, it will default to 1000.
        epochs_warmup : int
            Integer number of times to iterate over the training data arrays in warm up (without early stopping). If
            unspecified, it will default to 0.
        max_steps_per_epoch : int, optional
            Maximal steps per epoch. If unspecified, it will default to 20.
        batch_size : int
            Number of samples per gradient update. If unspecified, it will default to 16.
        validation_batch_size : int
            Number of samples in validation. If unspecified, it will default to 16.
        max_validation_steps : int
            Maximal steps per validation. If unspecified, it will default to 10.
        shuffle_buffer_size : int, optional
            Shuffle buffer size. If unspecified, it will default to 1e4.
        patience : int
            Number of epochs with no improvement. If unspecified, it will default to 20.
        lr_schedule_min_lr : float
            Lower bound on the learning rate. If unspecified, it will default to 1e-5.
        lr_schedule_factor : float
            Factor by which the learning rate will be reduced. new_lr = lr * factor. If unspecified, it will default
            to 0.2.
        lr_schedule_patience : int
            Number of epochs with no improvement after which learning rate will be reduced. If unspecified, it will
            default to 5.
        initial_epoch : int
            Epoch at which to start training (useful for resuming a previous training run). If unspecified, it will
            default to 0.
        monitor_partition : str
            Monitor partition.
        monitor_metric : str
            Monitor metric.
        log_dir : str, optional
            Logging directory.
        callbacks : list, optional
             List of callbacks to be called during training.
        early_stopping : bool
            Whether to activate early stopping.
        reduce_lr_plateau : bool
            Whether to reduce learning rate on plateau.
        pretrain_decoder : bool
            Whether to pretrain the decoder model.
        decoder_epochs : int
            Integer number of times to iterate over the training data arrays in decoder pretraining. If unspecified, it
            will default to 1000.
        decoder_patience : int
            Number of epochs with no improvement in decoder pretraining. If unspecified, it will default to 20.
        decoder_callbacks : list, optional
             List of callbacks to be called during decoder pretraining.
        aggressive : bool
            Whether to train aggressive.
        aggressive_enc_patience : int
             Number of epochs with no improvement in aggressive training. If unspecified, it will default to 10.
        aggressive_epochs : int
            Integer number of times to iterate over the training data arrays in aggressive training. If unspecified, it
            will default to 5.
        seed : int
            Random seed for reproduability.
        kwargs
            Arbitrary keyword arguments.
        """
        # Save training settings to allow model restoring.
        self.train_hyperparam = {
            "epochs": epochs,
            "epochs_warmup": epochs_warmup,
            "max_steps_per_epoch": max_steps_per_epoch,
            "batch_size": batch_size,
            "validation_batch_size": validation_batch_size,
            "max_validation_steps": max_validation_steps,
            "shuffle_buffer_size": shuffle_buffer_size,
            "patience": patience,
            "lr_schedule_min_lr": lr_schedule_min_lr,
            "lr_schedule_factor": lr_schedule_factor,
            "lr_schedule_patience": lr_schedule_patience,
            "log_dir": log_dir,
            "pretrain_decoder": pretrain_decoder,
            "decoder_epochs": decoder_epochs,
            "decoder_patience": decoder_patience,
            "aggressive": aggressive,
            "aggressive_enc_patience": aggressive_enc_patience,
            "aggressive_epochs": aggressive_epochs,
        }
        self.train_dataset = self._get_dataset(
            image_keys=self.img_keys_train,
            nodes_idx=self.nodes_idx_train,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            train=True,
            seed=seed,
            reinit_n_eval=None,
        )
        self.eval_dataset = self._get_dataset(
            image_keys=self.img_keys_eval,
            nodes_idx=self.nodes_idx_eval,
            batch_size=validation_batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            train=True,
            seed=seed,
            reinit_n_eval=None,
        )

        self.steps_per_epoch = min(max(len(self.img_keys_train) // batch_size, 1), max_steps_per_epoch)
        self.validation_steps = min(max(len(self.img_keys_eval) // validation_batch_size, 1), max_validation_steps)

        if pretrain_decoder:
            self.pretrain_decoder(
                decoder_epochs=decoder_epochs,
                patience=decoder_patience,
                lr_schedule_min_lr=lr_schedule_min_lr,
                lr_schedule_factor=lr_schedule_factor,
                lr_schedule_patience=lr_schedule_patience,
                initial_epoch=initial_epoch,
                monitor_partition=monitor_partition,
                monitor_metric=monitor_metric,
                log_dir=log_dir,
                callbacks=decoder_callbacks,
                early_stopping=early_stopping,
                reduce_lr_plateau=reduce_lr_plateau,
                **kwargs,
            )
        if aggressive:
            self.train_aggressive(aggressive_enc_patience=aggressive_enc_patience, aggressive_epochs=aggressive_epochs)

        if epochs_warmup > 0:
            self.train_normal(
                epochs=epochs_warmup,
                patience=patience,
                lr_schedule_min_lr=lr_schedule_min_lr,
                lr_schedule_factor=lr_schedule_factor,
                lr_schedule_patience=int(10000),  # dont reduce
                initial_epoch=initial_epoch,
                monitor_partition=monitor_partition,
                monitor_metric=monitor_metric,
                log_dir=log_dir,
                callbacks=callbacks,
                early_stopping=False,
                reduce_lr_plateau=reduce_lr_plateau,
                **kwargs,
            )
            initial_epoch += epochs_warmup

        self.train_normal(
            epochs=epochs,
            patience=patience,
            lr_schedule_min_lr=lr_schedule_min_lr,
            lr_schedule_factor=lr_schedule_factor,
            lr_schedule_patience=lr_schedule_patience,
            initial_epoch=initial_epoch,
            monitor_partition=monitor_partition,
            monitor_metric=monitor_metric,
            log_dir=log_dir,
            callbacks=callbacks,
            early_stopping=early_stopping,
            reduce_lr_plateau=reduce_lr_plateau,
            **kwargs,
        )

    def train_normal(
        self,
        epochs: int = 1000,
        patience: int = 20,
        lr_schedule_min_lr: float = 1e-5,
        lr_schedule_factor: float = 0.2,
        lr_schedule_patience: int = 5,
        initial_epoch: int = 0,
        monitor_partition: str = "val",
        monitor_metric: str = "loss",
        log_dir: Optional[str] = None,
        callbacks: Optional[list] = None,
        early_stopping: bool = True,
        reduce_lr_plateau: bool = True,
        **kwargs,
    ):
        """Train model normal.

        Use validation loss and maximum number of epochs as termination criteria.

        Parameters
        ----------
        epochs : int
            Integer number of times to iterate over the training data arrays. If unspecified, it will default to 1000.
        patience : int
            Number of epochs with no improvement. If unspecified, it will default to 20.
        lr_schedule_min_lr : float
            Lower bound on the learning rate. If unspecified, it will default to 1e-5.
        lr_schedule_factor : float
            Factor by which the learning rate will be reduced. new_lr = lr * factor. If unspecified, it will default
            to 0.2.
        lr_schedule_patience : int
            Number of epochs with no improvement after which learning rate will be reduced. If unspecified, it will
            default to 5.
        initial_epoch : int
            Epoch at which to start training (useful for resuming a previous training run). If unspecified, it will
            default to 0.
        monitor_partition : str
            Monitor partition.
        monitor_metric : str
            Monitor metric.
        log_dir : str, optional
            Logging directory.
        callbacks : list, optional
             List of callbacks to be called during training.
        early_stopping : bool
            Whether to activate early stopping.
        reduce_lr_plateau : bool
            Whether to reduce learning rate on plateau.
        kwargs
            Arbitrary keyword arguments.
        """
        # Set callbacks.
        cbs = []
        if reduce_lr_plateau:
            cbs.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor_partition + "_" + monitor_metric,
                    factor=lr_schedule_factor,
                    patience=lr_schedule_patience,
                    min_lr=lr_schedule_min_lr,
                )
            )
        if early_stopping:
            cbs.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor_partition + "_" + monitor_metric, patience=patience, restore_best_weights=True
                )
            )
        if log_dir is not None:
            cbs.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                    write_graph=False,
                    write_grads=False,
                    write_images=False,
                    embeddings_freq=0,
                    embeddings_layer_names=None,
                    embeddings_metadata=None,
                    embeddings_data=None,
                    update_freq="epoch",
                )
            )
        if callbacks is not None:
            # callbacks needs to be a list
            cbs += callbacks

        history = self.model.training_model.fit(
            x=self.train_dataset,
            epochs=epochs,
            initial_epoch=initial_epoch,
            steps_per_epoch=self.steps_per_epoch,
            callbacks=cbs,
            validation_data=self.eval_dataset,
            validation_steps=self.validation_steps,
            verbose=2,
            **kwargs,
        ).history
        for k, v in history.items():  # append to history if train() has been called before.
            if k in self.history.keys():
                self.history[k].extend(v)
            else:
                self.history[k] = v

    def pretrain_decoder(
        self,
        decoder_epochs: int = 1000,
        patience: int = 20,
        lr_schedule_min_lr: float = 1e-5,
        lr_schedule_factor: float = 0.2,
        lr_schedule_patience: int = 5,
        initial_epoch: int = 0,
        monitor_partition: str = "val",
        monitor_metric: str = "loss",
        log_dir: Optional[str] = None,
        callbacks: Optional[list] = None,
        early_stopping: bool = True,
        reduce_lr_plateau: bool = True,
        **kwargs,
    ):
        """Pre-train decoder model.

        Use validation loss and maximum number of epochs as termination criteria.

        Parameters
        ----------
        patience : int
            Number of epochs with no improvement. If unspecified, it will default to 20.
        lr_schedule_min_lr : float
            Lower bound on the learning rate. If unspecified, it will default to 1e-5.
        lr_schedule_factor : float
            Factor by which the learning rate will be reduced. new_lr = lr * factor. If unspecified, it will default
            to 0.2.
        lr_schedule_patience : int
            Number of epochs with no improvement after which learning rate will be reduced. If unspecified, it will
            default to 5.
        initial_epoch : int
            Epoch at which to start training (useful for resuming a previous training run). If unspecified, it will
            default to 0.
        monitor_partition : str
            Monitor partition.
        monitor_metric : str
            Monitor metric.
        log_dir : str, optional
            Logging directory.
        callbacks : list, optional
             List of callbacks to be called during training.
        early_stopping : bool
            Whether to activate early stopping.
        reduce_lr_plateau : bool
            Whether to reduce learning rate on plateau.
        decoder_epochs : int
            Integer number of times to iterate over the training data arrays in decoder pretraining. If unspecified, it
            will default to 1000.
        kwargs
            Arbitrary keyword arguments.
        """
        # Set callbacks.
        cbs = []
        if reduce_lr_plateau:
            cbs.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor_partition + "_" + monitor_metric,
                    factor=lr_schedule_factor,
                    patience=lr_schedule_patience,
                    min_lr=lr_schedule_min_lr,
                )
            )
        if early_stopping:
            cbs.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor_partition + "_" + monitor_metric, patience=patience, restore_best_weights=True
                )
            )
        if log_dir is not None:
            cbs.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                    write_graph=False,
                    write_grads=False,
                    write_images=False,
                    embeddings_freq=0,
                    embeddings_layer_names=None,
                    embeddings_metadata=None,
                    embeddings_data=None,
                    update_freq="epoch",
                )
            )
        if callbacks is not None:
            # callbacks needs to be a list
            cbs += callbacks

        history = self.model.decoder_sampling.fit(
            x=self.train_dataset,
            epochs=decoder_epochs,
            initial_epoch=initial_epoch,
            steps_per_epoch=self.steps_per_epoch,
            callbacks=cbs,
            validation_data=self.eval_dataset,
            validation_steps=self.validation_steps,
            verbose=2,
            **kwargs,
        ).history
        for k, v in history.items():  # append to history if train() has been called before.
            if k in self.history.keys():
                self.history[k].extend(v)
            else:
                self.history[k] = v
        # Transfer weights:
        layer_names_training_model = [x.name for x in self.model.training_model.layers]
        layer_names_decoder_model = [x.name for x in self.model.decoder_sampling.layers]
        layers_updated = []
        for x in layer_names_decoder_model:
            w = self.model.decoder_sampling.get_layer(name=x).get_weights()
            if x in layer_names_training_model:
                # Only update layers with parameters:
                if len(w) > 0:
                    self.model.training_model.get_layer(x).set_weights(w)
                    layers_updated.append(x)
            elif "Output_sampling" in x:
                # Find output layer *Output_decoder matched to *Output_sampling:
                x_out = [y for y in layer_names_training_model if "Output_decoder" in y][0]
                self.model.training_model.get_layer(x_out).set_weights(w)
                layers_updated.append(x_out)
        print(f"updated layers: {layers_updated}")

    def train_aggressive(
        self,
        aggressive_enc_patience: int = 10,
        aggressive_epochs: int = 5,
    ):
        """Train model aggressive.

        Parameters
        ----------
        aggressive_enc_patience : int
             Number of epochs with no improvement in aggressive training. If unspecified, it will default to 10.
        aggressive_epochs : int
            Integer number of times to iterate over the training data arrays in aggressive training. If unspecified, it
            will default to 5.
        """
        # @tf.function
        def train_iter(
            x_batch_aggressive,
            y_batch_aggressive,
            train_dec,
            train_enc,
        ):
            with tf.GradientTape() as g:
                output_decoder_concat, latent_space = self.model.training_model(x_batch_aggressive)
                losses_aggressive = {
                    "reconstruction_loss": self.loss[0](y_batch_aggressive[0], output_decoder_concat),
                    "bottleneck_loss": self.loss[1](y_batch_aggressive[1], latent_space),
                }
                losses_aggressive["loss"] = (
                    losses_aggressive["reconstruction_loss"] + losses_aggressive["bottleneck_loss"]
                )

            if train_enc:
                grad_enc = g.gradient(
                    target=losses_aggressive["loss"], sources=self.model.encoder_model.trainable_variables
                )
                self.optimizer.apply_gradients(zip(grad_enc, self.model.encoder_model.trainable_variables))
            if train_dec:
                grad_dec = g.gradient(
                    target=losses_aggressive["loss"], sources=self.model.decoder_model.trainable_variables
                )
                self.optimizer.apply_gradients(zip(grad_dec, self.model.decoder_model.trainable_variables))

            metrics_values_output = {
                "reconstruction_" + metric.__name__: metric(y_batch_aggressive[0], output_decoder_concat)
                for metric in self.metrics[0]
            }
            metrics_values_latent = {
                "bottleneck_" + metric.__name__: metric(y_batch_aggressive[1], latent_space)
                for metric in self.metrics[1]
            }
            losses_aggressive.update(metrics_values_output)
            losses_aggressive.update(metrics_values_latent)

            # Add non-scaled ELBO to model as metric (ie no annealing or beta-VAE scaling):
            log2pi = tf.math.log(2.0 * np.pi)
            z, z_mean, z_log_var = tf.split(latent_space, num_or_size_splits=3, axis=1)
            logqz_x = -0.5 * tf.reduce_mean(tf.square(z - z_mean) * tf.exp(-z_log_var) + z_log_var + log2pi)
            logpz = -0.5 * tf.reduce_mean(tf.square(z) + log2pi)
            d_kl = logqz_x - logpz
            loc, scale = tf.split(output_decoder_concat, num_or_size_splits=2, axis=2)
            input_x = x_batch_aggressive[0]
            if self.output_layer == "gaussian" or self.output_layer == "gaussian_const_disp":
                neg_ll = tf.math.log(tf.sqrt(2 * np.math.pi) * scale) + 0.5 * tf.math.square(
                    loc - input_x
                ) / tf.math.square(scale)
            elif (
                self.output_layer == "nb"
                or self.output_layer == "nb_const_disp"
                or self.output_layer == "nb_shared_disp"
            ):
                eta_loc = tf.math.log(loc)
                eta_scale = tf.math.log(scale)

                log_r_plus_mu = tf.math.log(scale + loc)

                ll = tf.math.lgamma(scale + input_x)
                ll = ll - tf.math.lgamma(input_x + tf.ones_like(input_x))
                ll = ll - tf.math.lgamma(scale)
                ll = ll + tf.multiply(input_x, eta_loc - log_r_plus_mu) + tf.multiply(scale, eta_scale - log_r_plus_mu)

                neg_ll = -tf.clip_by_value(ll, -300, 300, "log_probs")
            else:
                neg_ll = None
            neg_ll = tf.reduce_mean(tf.reduce_sum(neg_ll, axis=-1))
            losses_aggressive["elbo"] = neg_ll + d_kl

            return losses_aggressive

        aggressive = True
        history = {}
        ep = 0
        while aggressive:
            ep += 1
            print("Epoch (aggressive) {}/{} - ".format(ep, aggressive_epochs), end="")
            start = time.time()
            no_improvement = 0
            best_result = None
            # inner loop training only encoder until no further improvement in ELBO val loss
            enc_updates = 0
            count = 0
            while no_improvement < aggressive_enc_patience:
                enc_updates += 1
                for step, (x_batch, y_batch) in enumerate(self.train_dataset):
                    count = step
                    if step >= self.steps_per_epoch:
                        break
                    _ = train_iter(
                        x_batch_aggressive=x_batch,
                        y_batch_aggressive=y_batch,
                        train_enc=True,
                        train_dec=False,
                    )
                elbo_enc_epoch_eval = 0
                for step, (x_batch, y_batch) in enumerate(self.eval_dataset):
                    count = step
                    if step >= self.validation_steps:
                        break
                    losses = train_iter(
                        x_batch_aggressive=x_batch,
                        y_batch_aggressive=y_batch,
                        train_enc=False,
                        train_dec=False,
                    )
                    elbo_enc_epoch_eval += losses["elbo"]
                elbo_enc_epoch_eval /= count

                if best_result is None:
                    best_result = elbo_enc_epoch_eval
                elif elbo_enc_epoch_eval < best_result:
                    best_result = elbo_enc_epoch_eval
                    no_improvement = 0
                else:
                    no_improvement += 1
            print("Performed %d encoder updates" % enc_updates)

            # one step decoder training
            hist = {}
            for step, (x_batch, y_batch) in enumerate(self.train_dataset):
                count = step
                if step >= self.steps_per_epoch:
                    break
                losses = train_iter(
                    x_batch_aggressive=x_batch,
                    y_batch_aggressive=y_batch,
                    train_enc=False,
                    train_dec=True,
                )
                for k, v in losses.items():
                    if k in hist.keys():
                        hist[k] += np.mean(v)
                    else:
                        hist[k] = np.mean(v)
            hist = {key: value / count for key, value in hist.items()}

            for k, v in hist.items():
                if k in history.keys():
                    history[k].append(v)
                else:
                    history[k] = [v]
            hist_eval = {}
            for step, (x_batch, y_batch) in enumerate(self.eval_dataset):
                count = step
                if step >= self.validation_steps:
                    break
                losses = train_iter(
                    x_batch_aggressive=x_batch,
                    y_batch_aggressive=y_batch,
                    train_enc=False,
                    train_dec=False,
                )
                for k, v in losses.items():
                    if k in hist_eval.keys():
                        hist_eval[k] += np.mean(v)
                    else:
                        hist_eval[k] = np.mean(v)
            hist_eval = {"val_" + key: value / count for key, value in hist_eval.items()}
            for k, v in hist_eval.items():
                if k in history.keys():
                    history[k].append(v)
                else:
                    history[k] = [v]
            if "lr" in history.keys():
                history["lr"].append(self.optimizer.lr.numpy())
            else:
                history["lr"] = [self.optimizer.lr.numpy()]

            if len(history["loss"]) >= aggressive_epochs:
                aggressive = False

            print("%d/%d - %ds" % (self.steps_per_epoch, self.steps_per_epoch, time.time() - start), end="")
            for key, loss in history.items():
                print(" - %s: %f" % (key, loss[-1]), end="")
            print()

        for k, v in history.items():  # append to history if train() has been called before.
            if k in self.history.keys():
                self.history[k].extend(v)
            else:
                self.history[k] = v

    def _get_dataset_test(self, batch_size: int = 1):
        """Get test dataset.

        Parameters
        ----------
        batch_size : int
            Number of samples. If unspecified, it will default to 1.

        Returns
        -------
        A tensorflow dataset.


        """
        if self.img_keys_test is not None and len(self.img_keys_test) != 0:
            image_keys = self.img_keys_test
        else:
            warnings.warn("Image keys for test set empty. Evaluating on all images in whole dataset!")
            image_keys = list(self.complete_img_keys)
        if self.nodes_idx_test is not None:
            nodes_idx = self.nodes_idx_test
        else:
            warnings.warn("Node idx for test set empty. Evaluating on all nodes in whole dataset!")
            nodes_idx = "all"

        return self._get_dataset(
            image_keys=image_keys,
            nodes_idx=nodes_idx,
            batch_size=batch_size,
            shuffle_buffer_size=1,
            train=False,
            seed=None,
            reinit_n_eval=None,
        )

    def predict(self, batch_size: int = 1) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Return observed labels and full predictions (including scale model) grouped exactly as in nodes_idx_test.

        Parameters
        ----------
        batch_size : int
            Number of samples. If unspecified, it will default to 1.

        Returns
        -------
        predict
        """
        ds = self._get_dataset_test(batch_size=batch_size)
        return self.model.training_model.predict(ds)

    def evaluate_any(self, img_keys, node_idx, batch_size: int = 1):
        """Evaluate model on any given data set.

        Parameters
        ----------
        img_keys
            Image keys.
        node_idx
            Nodes indices.
        batch_size : int
            Number of samples. If unspecified, it will default to 1.

        Returns
        -------
        eval_dict
        """
        ds = self._get_dataset(
            image_keys=img_keys,
            nodes_idx=node_idx,
            batch_size=batch_size,
            shuffle_buffer_size=1,
            train=False,
            seed=None,
            reinit_n_eval=None,
        )
        results = self.model.training_model.evaluate(ds, verbose=False)
        eval_dict = dict(zip(self.model.training_model.metrics_names, results))
        return eval_dict

    def evaluate_per_node_type(self, batch_size: int = 1):
        """Evaluate model for each node type seperately.

        Parameters
        ----------
        batch_size : int
            Number of samples. If unspecified, it will default to 1.

        Returns
        -------
        split_per_node_type, evaluation_per_node_type
        """
        if self.simulation:
            split_per_node_type = None
            evaluation_per_node_type = None
        else:
            evaluation_per_node_type = {}
            split_per_node_type = {}
            node_types = list(self.node_type_names.keys())
            for nt in node_types:
                img_keys = list(self.complete_img_keys)
                nodes_idx = {k: np.where(self.node_types[k][:, node_types.index(nt)] == 1)[0] for k in img_keys}
                split_per_node_type.update({nt: {"img_keys": img_keys, "nodes_idx": nodes_idx}})
                test = {k: len(np.where(self.node_types[k][:, node_types.index(nt)] == 1)[0]) for k in img_keys}
                print("Evaluation for %s with %i cells" % (nt, sum(test.values())))
                ds = self._get_dataset(
                    image_keys=img_keys,
                    nodes_idx=nodes_idx,
                    batch_size=batch_size,
                    shuffle_buffer_size=1,
                    train=False,
                    seed=None,
                    reinit_n_eval=None,
                )
                results = self.model.training_model.evaluate(ds, verbose=False)
                eval_dict = dict(zip(self.model.training_model.metrics_names, results))
                print(eval_dict)
                evaluation_per_node_type.update({nt: eval_dict})
        return split_per_node_type, evaluation_per_node_type


class EstimatorGraph(Estimator):
    """EstimatorGraph class for spatial models."""

    def init_model(self, **kwargs):
        """Initialize EstimatorGraph.

        Parameters
        ----------
        kwargs
            Arbitrary keyword arguments.
        """
        pass

    def _get_output_signature(self, resampled: bool = False):
        """Get output signatures.

        Parameters
        ----------
        resampled : bool
            Whether dataset is resampled or not.

        Returns
        -------
        output_signature
        """
        h_1 = tf.TensorSpec(
            shape=(self.n_eval_nodes_per_graph, self.n_features_1), dtype=tf.float32
        )  # input node features
        sf = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, 1), dtype=tf.float32)  # input node size factors
        h_0 = tf.TensorSpec(
            shape=(self.n_eval_nodes_per_graph, self.n_features_0), dtype=tf.float32
        )  # input node features conditional
        h_0_full = tf.TensorSpec(
            shape=(self.max_nodes, self.n_features_0), dtype=tf.float32
        )  # input node features conditional
        a = tf.SparseTensorSpec(shape=None, dtype=tf.float32)  # adjacency matrix
        a_full = tf.SparseTensorSpec(shape=None, dtype=tf.float32)  # adjacency matrix
        node_covar = tf.TensorSpec(
            shape=(self.n_eval_nodes_per_graph, self.n_node_covariates), dtype=tf.float32
        )  # node-level covariates
        domain = tf.TensorSpec(shape=(self.n_domains,), dtype=tf.int32)  # domain
        reconstruction = tf.TensorSpec(
            shape=(self.n_eval_nodes_per_graph, self.n_features_1), dtype=tf.float32
        )  # node features to reconstruct
        kl_dummy = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph,), dtype=tf.float32)  # dummy for kl loss

        if self.vi_model:
            if resampled:
                output_signature = (
                    (h_1, sf, h_0, h_0_full, a, a_full, node_covar, domain),
                    (reconstruction, kl_dummy),
                    (h_1, sf, h_0, h_0_full, a, a_full, node_covar, domain),
                    (reconstruction, kl_dummy),
                )
            else:
                output_signature = ((h_1, sf, h_0, h_0_full, a, a_full, node_covar, domain), (reconstruction, kl_dummy))
        else:
            if resampled:
                output_signature = (
                    (h_1, sf, h_0, h_0_full, a, a_full, node_covar, domain),
                    reconstruction,
                    (h_1, sf, h_0, h_0_full, a, a_full, node_covar, domain),
                    reconstruction,
                )
            else:
                output_signature = ((h_1, sf, h_0, h_0_full, a, a_full, node_covar, domain), reconstruction)
        return output_signature

    def _get_dataset(
        self,
        image_keys: List[str],
        nodes_idx: Dict[str, np.ndarray],
        batch_size: int,
        shuffle_buffer_size: Optional[int],
        train: bool = True,
        seed: Optional[int] = None,
        prefetch: int = 100,
        reinit_n_eval: Optional[int] = None,
    ):
        """Prepare a dataset.

        Parameters
        ----------
        image_keys : np.array
            Image keys in partition.
        nodes_idx : dict, str
            Dictionary of nodes per image in partition.
        batch_size : int
            Batch size.
        shuffle_buffer_size : int, optional
            Shuffle buffer size.
        train : bool
            Whether dataset is used for training or not (influences shuffling of nodes).
        seed : int, optional
            Random seed.
        prefetch: int
            Prefetch of dataset.
        reinit_n_eval : int, optional
            Used if model is reinitialized to different number of nodes per graph.

        Returns
        -------
        A tensorflow dataset.
        """
        np.random.seed(seed)
        if reinit_n_eval is not None and reinit_n_eval != self.n_eval_nodes_per_graph:
            print(
                "ATTENTION: specifying reinit_n_eval will change class argument n_eval_nodes_per_graph "
                "from %i to %i" % (self.n_eval_nodes_per_graph, reinit_n_eval)
            )
            self.n_eval_nodes_per_graph = reinit_n_eval

        def generator():
            for key in image_keys:
                if nodes_idx[key].size == 0:  # needed for images where no nodes are selected
                    continue
                idx_nodes = np.arange(0, self.a[key].shape[0])

                if train:
                    index_list = [
                        np.asarray(
                            np.random.choice(
                                a=nodes_idx[key],
                                size=self.n_eval_nodes_per_graph,
                                replace=True,
                            ),
                            dtype=np.int32,
                        )
                    ]
                else:
                    # dropping
                    index_list = [
                        np.asarray(
                            nodes_idx[key][self.n_eval_nodes_per_graph * i : self.n_eval_nodes_per_graph * (i + 1)],
                            dtype=np.int32,
                        )
                        for i in range(len(nodes_idx[key]) // self.n_eval_nodes_per_graph)
                    ]

                for indices in index_list:
                    h_0 = self.h_0[key][idx_nodes]
                    diff = self.max_nodes - h_0.shape[0]
                    zeros = np.zeros((diff, h_0.shape[1]), dtype="float32")
                    h_0_full = np.asarray(np.concatenate((h_0, zeros), axis=0), dtype="float32")
                    h_0 = h_0_full[indices]

                    h_1 = self.h_1[key][idx_nodes]
                    diff = self.max_nodes - h_1.shape[0]
                    zeros = np.zeros((diff, h_1.shape[1]), dtype="float32")
                    h_1 = np.asarray(np.concatenate((h_1, zeros), axis=0), dtype="float32")
                    h_1 = h_1[indices]
                    if self.log_transform:
                        h_1 = np.log(h_1 + 1.0)

                    # indexing adjacency matrix that yield only selected cells (final graph layer)
                    a = self.a[key][idx_nodes, :][:, idx_nodes]
                    a = a[indices, :]
                    coo = a.tocoo()
                    a_ind = np.asarray(np.mat([coo.row, coo.col]).transpose(), dtype="int64")
                    a_val = np.asarray(coo.data, dtype="float32")
                    a_shape = np.asarray((self.n_eval_nodes_per_graph, self.max_nodes), dtype="int64")
                    a = tf.SparseTensor(indices=a_ind, values=a_val, dense_shape=a_shape)
                    # propagating adjacency matrix that yield all cells (before final graph layer)
                    if self.cond_depth > 1:
                        a_full = self.a[key][idx_nodes, :][:, idx_nodes]
                        a_full = a_full.tocoo()
                        afull_ind = np.asarray(np.mat([a_full.row, a_full.col]).transpose(), dtype="int64")
                        afull_val = np.asarray(a_full.data, dtype="float32")
                    else:
                        afull_ind = np.asarray(np.mat([np.zeros((0,)), np.zeros((0,))]).transpose(), dtype="int64")
                        afull_val = np.asarray(np.zeros((0,)), dtype="float32")
                    afull_shape = np.asarray((self.max_nodes, self.max_nodes), dtype="int64")
                    a_full = tf.SparseTensor(indices=afull_ind, values=afull_val, dense_shape=afull_shape)

                    node_covar = self.node_covar[key][idx_nodes]
                    diff = self.max_nodes - node_covar.shape[0]
                    zeros = np.zeros((diff, node_covar.shape[1]))
                    node_covar = np.asarray(np.concatenate([node_covar, zeros], axis=0), dtype="float32")
                    node_covar = node_covar[indices, :]

                    sf = np.expand_dims(self.size_factors[key][idx_nodes], axis=1)
                    diff = self.max_nodes - sf.shape[0]
                    zeros = np.zeros((diff, sf.shape[1]))
                    sf = np.asarray(np.concatenate([sf, zeros], axis=0), dtype="float32")
                    sf = sf[indices, :]

                    g = np.zeros((self.n_domains,), dtype="int32")
                    g[self.domains[key]] = 1

                    if self.vi_model:
                        kl_dummy = np.zeros((self.n_eval_nodes_per_graph,), dtype="float32")
                        yield (h_1, sf, h_0, h_0_full, a, a_full, node_covar, g), (h_1, kl_dummy)
                    else:
                        yield (h_1, sf, h_0, h_0_full, a, a_full, node_covar, g), h_1

        output_signature = self._get_output_signature(resampled=False)

        dataset = tf.data.Dataset.from_generator(generator=generator, output_signature=output_signature)
        if train:
            if shuffle_buffer_size is not None:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=None, reshuffle_each_iteration=True)
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch)
        return dataset

    def _get_resampled_dataset(
        self,
        image_keys: np.ndarray,
        nodes_idx: dict,
        batch_size: int,
        seed: Optional[int] = None,
        prefetch: int = 100,
        reinit_n_eval: Optional[int] = None,
    ):
        """Evaluate model based on resampled dataset for posterior resampling.

        node_1 + domain_1 -> encoder -> z_1 + domain_2 -> decoder -> reconstruction_2.

        Parameters
        ----------
        image_keys: np.array
            Image keys in partition.
        nodes_idx : dict
            Dictionary of nodes per image in partition.
        batch_size : int
            Batch size.
        seed : int, optional
            Seed.
        prefetch : int
            Prefetch.
        reinit_n_eval : int, optional
            Used if model is reinitialized to different number of nodes per graph.

        Returns
        -------
        A tensorflow dataset.
        """
        np.random.seed(seed)
        if reinit_n_eval is not None:
            print(
                "ATTENTION: specifying reinit_n_eval will change class argument n_eval_nodes_per_graph "
                "from %i to %i" % (self.n_eval_nodes_per_graph, reinit_n_eval)
            )
            self.n_eval_nodes_per_graph = reinit_n_eval

        def generator():
            for key in image_keys:
                if nodes_idx[key].size == 0:  # needed for images where no nodes are selected
                    continue
                idx_nodes = np.arange(0, self.a[key].shape[0])

                index_list = [
                    np.asarray(
                        nodes_idx[key][self.n_eval_nodes_per_graph * i : self.n_eval_nodes_per_graph * (i + 1)],
                        dtype=np.int32,
                    )
                    for i in range(len(nodes_idx[key]) // self.n_eval_nodes_per_graph)
                ]
                resampled_index_list = [
                    np.asarray(
                        np.random.choice(
                            a=nodes_idx[key],
                            size=self.n_eval_nodes_per_graph,
                            replace=True,
                        ),
                        dtype=np.int32,
                    )
                    for i in range(len(nodes_idx[key]) // self.n_eval_nodes_per_graph)
                ]

                for i, indices in enumerate(index_list):
                    re_indices = resampled_index_list[i]

                    h_0 = self.h_0[key][idx_nodes]
                    diff = self.max_nodes - h_0.shape[0]
                    zeros = np.zeros((diff, h_0.shape[1]), dtype="float32")
                    h_0_full = np.asarray(np.concatenate((h_0, zeros), axis=0), dtype="float32")
                    re_h_0 = h_0_full[re_indices]
                    h_0 = h_0_full[indices]

                    h_1 = self.h_1[key][idx_nodes]
                    diff = self.max_nodes - h_1.shape[0]
                    zeros = np.zeros((diff, h_1.shape[1]), dtype="float32")
                    h_1 = np.asarray(np.concatenate((h_1, zeros), axis=0), dtype="float32")
                    re_h_1 = h_1[re_indices]
                    h_1 = h_1[indices]
                    if self.log_transform:
                        h_1 = np.log(h_1 + 1.0)
                        re_h_1 = np.log(re_h_1 + 1.0)

                    # indexing adjacency matrix that yield only selected cells (final graph layer)
                    a = self.a[key][idx_nodes, :][:, idx_nodes]
                    re_a = a[re_indices, :]
                    re_coo = re_a.tocoo()
                    re_a_ind = np.asarray(np.mat([re_coo.row, re_coo.col]).transpose(), dtype="int64")
                    re_a_val = np.asarray(re_coo.data, dtype="float32")
                    re_a_shape = np.asarray((self.n_eval_nodes_per_graph, self.max_nodes), dtype="int64")
                    re_a = tf.SparseTensor(indices=re_a_ind, values=re_a_val, dense_shape=re_a_shape)

                    a = a[indices, :]
                    coo = a.tocoo()
                    a_ind = np.asarray(np.mat([coo.row, coo.col]).transpose(), dtype="int64")
                    a_val = np.asarray(coo.data, dtype="float32")
                    a_shape = np.asarray((self.n_eval_nodes_per_graph, self.max_nodes), dtype="int64")
                    a = tf.SparseTensor(indices=a_ind, values=a_val, dense_shape=a_shape)

                    # propagating adjacency matrix that yield all cells (before final graph layer)
                    if self.model.args["cond_depth"] > 1:
                        a_full = self.a[key][idx_nodes, :][:, idx_nodes]
                        a_full = a_full.tocoo()
                        afull_ind = np.asarray(np.mat([a_full.row, a_full.col]).transpose(), dtype="int64")
                        afull_val = np.asarray(a_full.data, dtype="float32")
                    else:
                        afull_ind = np.asarray(np.mat([np.zeros((0,)), np.zeros((0,))]).transpose(), dtype="int64")
                        afull_val = np.asarray(np.zeros((0,)), dtype="float32")
                    afull_shape = np.asarray((self.max_nodes, self.max_nodes), dtype="int64")
                    a_full = tf.SparseTensor(indices=afull_ind, values=afull_val, dense_shape=afull_shape)

                    node_covar = self.node_covar[key][idx_nodes]
                    diff = self.max_nodes - node_covar.shape[0]
                    zeros = np.zeros((diff, node_covar.shape[1]))
                    node_covar = np.asarray(np.concatenate([node_covar, zeros], axis=0), dtype="float32")
                    re_node_covar = node_covar[re_indices, :]
                    node_covar = node_covar[indices, :]

                    sf = np.expand_dims(self.size_factors[key][idx_nodes], axis=1)
                    diff = self.max_nodes - sf.shape[0]
                    zeros = np.zeros((diff, sf.shape[1]))
                    sf = np.asarray(np.concatenate([sf, zeros], axis=0), dtype="float32")
                    re_sf = sf[re_indices, :]
                    sf = sf[indices, :]

                    g = np.zeros((self.n_domains,), dtype="int32")
                    g[self.domains[key]] = 1

                    if self.vi_model:
                        kl_dummy = np.zeros((self.n_eval_nodes_per_graph,), dtype="float32")
                        yield (h_1, sf, h_0, h_0_full, a, a_full, node_covar, g), (h_1, kl_dummy), (
                            re_h_1,
                            re_sf,
                            re_h_0,
                            h_0_full,
                            re_a,
                            a_full,
                            re_node_covar,
                            g,
                        ), (re_h_1, kl_dummy)
                    else:
                        yield (h_1, sf, h_0, h_0_full, a, a_full, node_covar, g), h_1, (
                            re_h_1,
                            re_sf,
                            re_h_0,
                            h_0_full,
                            re_a,
                            a_full,
                            re_node_covar,
                            g,
                        ), re_h_1

        output_signature = self._get_output_signature(resampled=True)

        dataset = tf.data.Dataset.from_generator(generator=generator, output_signature=output_signature)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch)
        return dataset


class EstimatorNoGraph(Estimator):
    """EstimatorNoGraph class for baseline models."""

    def init_model(self, **kwargs):
        """Initialize EstimatorNoGraph.

        Parameters
        ----------
        kwargs
            Arbitrary keyword arguments.
        """
        pass

    def _get_output_signature(self, resampled: bool = False):
        """Get output signatures.

        Parameters
        ----------
        resampled : bool
            Whether dataset is resampled or not.

        Returns
        -------
        output_signature
        """
        h_1 = tf.TensorSpec(
            shape=(self.n_eval_nodes_per_graph, self.n_features_1), dtype=tf.float32
        )  # input node features
        sf = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, 1), dtype=tf.float32)  # input node size factors
        node_covar = tf.TensorSpec(
            shape=(self.n_eval_nodes_per_graph, self.n_node_covariates), dtype=tf.float32
        )  # node-level covariates
        domain = tf.TensorSpec(shape=(self.n_domains,), dtype=tf.int32)  # domain
        reconstruction = tf.TensorSpec(
            shape=(self.n_eval_nodes_per_graph, self.n_features_1), dtype=tf.float32
        )  # node features to reconstruct
        kl_dummy = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph,), dtype=tf.float32)  # dummy for kl loss

        if self.vi_model:
            if resampled:
                output_signature = (
                    (h_1, sf, node_covar, domain),
                    (reconstruction, kl_dummy),
                    (h_1, sf, node_covar, domain),
                    (reconstruction, kl_dummy),  # shapes for resampled output
                )
            else:
                output_signature = ((h_1, sf, node_covar, domain), (reconstruction, kl_dummy))
        else:
            if resampled:
                output_signature = (
                    (h_1, sf, node_covar, domain),
                    reconstruction,
                    (h_1, sf, node_covar, domain),
                    reconstruction,  # shapes for resampled output
                )
            else:
                output_signature = ((h_1, sf, node_covar, domain), reconstruction)
        return output_signature

    def _get_dataset(
        self,
        image_keys: List[str],
        nodes_idx: Dict[str, np.ndarray],
        batch_size: int,
        shuffle_buffer_size: Optional[int],
        train: bool = True,
        seed: Optional[int] = None,
        prefetch: int = 100,
        reinit_n_eval: Optional[int] = None,
    ):
        """Prepare a dataset.

        Parameters
        ----------
        image_keys : np.array
            Image keys in partition.
        nodes_idx : dict, str
            Dictionary of nodes per image in partition.
        batch_size : int
            Batch size.
        shuffle_buffer_size : int, optional
            Shuffle buffer size.
        train : bool
            Whether dataset is used for training or not (influences shuffling of nodes).
        seed : int, optional
            Random seed.
        prefetch: int
            Prefetch of dataset.
        reinit_n_eval : int, optional
            Used if model is reinitialized to different number of nodes per graph.

        Returns
        -------
        A tensorflow dataset.
        """
        np.random.seed(seed)
        if reinit_n_eval is not None and reinit_n_eval != self.n_eval_nodes_per_graph:
            print(
                "ATTENTION: specifying reinit_n_eval will change class argument n_eval_nodes_per_graph "
                "from %i to %i" % (self.n_eval_nodes_per_graph, reinit_n_eval)
            )
            self.n_eval_nodes_per_graph = reinit_n_eval

        def generator():
            for key in image_keys:
                if nodes_idx[key].size == 0:  # needed for images where no nodes are selected
                    continue
                idx_nodes = np.arange(0, self.a[key].shape[0])

                if train:
                    index_list = [
                        np.asarray(
                            np.random.choice(
                                a=nodes_idx[key],
                                size=self.n_eval_nodes_per_graph,
                                replace=True,
                            ),
                            dtype=np.int32,
                        )
                    ]
                else:
                    # dropping
                    index_list = [
                        np.asarray(
                            nodes_idx[key][self.n_eval_nodes_per_graph * i : self.n_eval_nodes_per_graph * (i + 1)],
                            dtype=np.int32,
                        )
                        for i in range(len(nodes_idx[key]) // self.n_eval_nodes_per_graph)
                    ]

                for indices in index_list:
                    h_1 = self.h_1[key][idx_nodes]
                    diff = self.max_nodes - h_1.shape[0]

                    zeros = np.zeros((diff, h_1.shape[1]))
                    h_1 = np.asarray(np.concatenate((h_1, zeros), axis=0), dtype="float32")
                    h_1 = h_1[indices]
                    if self.log_transform:
                        h_1 = np.log(h_1 + 1.0)

                    node_covar = self.node_covar[key][idx_nodes]
                    diff = self.max_nodes - node_covar.shape[0]
                    zeros = np.zeros((diff, node_covar.shape[1]))
                    node_covar = np.asarray(np.concatenate([node_covar, zeros], axis=0), dtype="float32")
                    node_covar = node_covar[indices]

                    sf = np.expand_dims(self.size_factors[key][idx_nodes], axis=1)
                    diff = self.max_nodes - sf.shape[0]
                    zeros = np.zeros((diff, sf.shape[1]))
                    sf = np.asarray(np.concatenate([sf, zeros], axis=0), dtype="float32")
                    sf = sf[indices, :]

                    g = np.zeros((self.n_domains,), dtype="int32")
                    g[self.domains[key]] = 1

                    if self.vi_model:
                        kl_dummy = np.zeros((self.n_eval_nodes_per_graph,), dtype="float32")
                        yield (h_1, sf, node_covar, g), (h_1, kl_dummy)
                    else:
                        yield (h_1, sf, node_covar, g), h_1

        output_signature = self._get_output_signature(resampled=False)

        dataset = tf.data.Dataset.from_generator(generator=generator, output_signature=output_signature)
        if train:
            if shuffle_buffer_size is not None:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=None, reshuffle_each_iteration=True)
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch)
        return dataset

    def _get_resampled_dataset(
        self,
        image_keys: np.ndarray,
        nodes_idx: dict,
        batch_size: int,
        seed: Optional[int] = None,
        prefetch: int = 100,
        reinit_n_eval: Optional[int] = None,
    ):
        """Evaluate model based on resampled dataset for posterior resampling.

        node_1 + domain_1 -> encoder -> z_1 + domain_2 -> decoder -> reconstruction_2.

        Parameters
        ----------
        image_keys: np.array
            Image keys in partition.
        nodes_idx : dict
            Dictionary of nodes per image in partition.
        batch_size : int
            Batch size.
        seed : int, optional
            Seed.
        prefetch : int
            Prefetch.
        reinit_n_eval : int, optional
            Used if model is reinitialized to different number of nodes per graph.

        Returns
        -------
        A tensorflow dataset.
        """
        np.random.seed(seed)
        if reinit_n_eval is not None:
            print(
                "ATTENTION: specifying reinit_n_eval will change class argument n_eval_nodes_per_graph "
                "from %i to %i" % (self.n_eval_nodes_per_graph, reinit_n_eval)
            )
            self.n_eval_nodes_per_graph = reinit_n_eval

        def generator():
            for key in image_keys:
                if nodes_idx[key].size == 0:  # needed for images where no nodes are selected
                    continue
                idx_nodes = np.arange(0, self.a[key].shape[0])

                index_list = [
                    np.asarray(
                        nodes_idx[key][self.n_eval_nodes_per_graph * i : self.n_eval_nodes_per_graph * (i + 1)],
                        dtype=np.int32,
                    )
                    for i in range(len(nodes_idx[key]) // self.n_eval_nodes_per_graph)
                ]
                resampled_index_list = [
                    np.asarray(
                        np.random.choice(
                            a=nodes_idx[key],
                            size=self.n_eval_nodes_per_graph,
                            replace=True,
                        ),
                        dtype=np.int32,
                    )
                    for i in range(len(nodes_idx[key]) // self.n_eval_nodes_per_graph)
                ]

                for i, indices in enumerate(index_list):
                    re_indices = resampled_index_list[i]

                    h_1 = self.h_1[key][idx_nodes]
                    diff = self.max_nodes - h_1.shape[0]
                    zeros = np.zeros((diff, h_1.shape[1]))
                    h_1 = np.asarray(np.concatenate((h_1, zeros), axis=0), dtype="float32")
                    re_h_1 = h_1[re_indices]
                    h_1 = h_1[indices]
                    if self.log_transform:
                        h_1 = np.log(h_1 + 1.0)
                        re_h_1 = np.log(re_h_1 + 1.0)

                    node_covar = self.node_covar[key][idx_nodes]
                    diff = self.max_nodes - node_covar.shape[0]
                    zeros = np.zeros((diff, node_covar.shape[1]))
                    node_covar = np.asarray(np.concatenate([node_covar, zeros], axis=0), dtype="float32")
                    re_node_covar = node_covar[re_indices]
                    node_covar = node_covar[indices]

                    sf = np.expand_dims(self.size_factors[key][idx_nodes], axis=1)
                    diff = self.max_nodes - sf.shape[0]
                    zeros = np.zeros((diff, sf.shape[1]))
                    sf = np.asarray(np.concatenate([sf, zeros], axis=0), dtype="float32")
                    re_sf = sf[re_indices, :]
                    sf = sf[indices, :]

                    g = np.zeros((self.n_domains,), dtype="int32")
                    g[self.domains[key]] = 1

                    if self.vi_model:
                        kl_dummy = np.zeros((self.n_eval_nodes_per_graph,), dtype="float32")
                        yield (h_1, sf, node_covar, g), (h_1, kl_dummy), (re_h_1, re_sf, re_node_covar, g), (
                            re_h_1,
                            kl_dummy,
                        )
                    else:
                        yield (h_1, sf, node_covar, g), h_1, (re_h_1, re_sf, re_node_covar, g), re_h_1

        output_signature = self._get_output_signature(resampled=True)

        dataset = tf.data.Dataset.from_generator(generator=generator, output_signature=output_signature)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch)
        return dataset
