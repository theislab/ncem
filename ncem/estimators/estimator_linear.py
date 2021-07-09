from typing import Optional, Union

import numpy as np
import tensorflow as tf
from patsy import dmatrix
from scipy.sparse import coo_matrix

from ncem.estimators import Estimator
from ncem.models import ModelLinear


class EstimatorLinear(Estimator):
    """Estimator class for linear models. Subclass of Estimator."""

    def __init__(
        self,
        log_transform: bool = False,
    ):
        """Initialize a EstimatorLinear object.

        Parameters
        ----------
        log_transform : bool
            Whether to log transform h_1.
        """
        super(EstimatorLinear, self).__init__()
        self.model_type = "linear"
        self.adj_type = "full"
        self.log_transform = log_transform
        self.metrics = {"np": [], "tf": []}
        self.n_eval_nodes_per_graph = None

    def _get_output_signature(self, resampled: bool = False):
        pass

    def _get_dataset(
        self,
        image_keys: np.array,
        nodes_idx: Union[dict, str],
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
        nodes_idx: dict, str
            Dictionary of nodes per image in partition.
        batch_size: int
            Batch size.
        shuffle_buffer_size : int, optional
            Shuffle buffer size.
        train : bool
            Whether dataset is used for training or not (influences shuffling of nodes).
        seed : int, optional
            Random seed.
        prefetch : int
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

                    h_0 = self.h_0[key][idx_nodes]
                    diff = self.max_nodes - h_0.shape[0]
                    zeros = np.zeros((diff, h_0.shape[1]), dtype="float32")
                    h_0_full = np.asarray(np.concatenate((h_0, zeros), axis=0), dtype="float32")
                    h_0 = h_0_full[indices]

                    a = self.a[key][idx_nodes, :][:, idx_nodes]
                    a = a[indices, :]
                    coo = a.tocoo()
                    a_shape = np.asarray((self.n_eval_nodes_per_graph, self.max_nodes), dtype="int64")
                    a = coo_matrix((coo.data, (coo.row, coo.col)), a_shape)

                    # Assemble design matrix components for interactions:
                    source = (a.toarray() > 0.0).astype("int").dot(h_0_full)  # n_eval_nodes_per_graph x node types
                    # discretize source types
                    source = np.asarray(source > 0, dtype="int")

                    data = {"target": h_0, "source": source}
                    target = np.asarray(dmatrix("target-1", data))
                    source = np.asarray(dmatrix("source-1", data))

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

                    yield (target, source, sf, node_covar, g), h_1

        dataset = tf.data.Dataset.from_generator(
            generator=generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, self.n_features_0), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, self.n_features_0), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, self.n_node_covariates), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.n_domains,), dtype=tf.int32),
                ),
                tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, self.n_features_1), dtype=tf.float32),
            ),
        )
        if train:
            if shuffle_buffer_size is not None:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=None, reshuffle_each_iteration=True)
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch)
        return dataset

    def _get_resampled_dataset(
        self,
        image_keys: np.array,
        nodes_idx: dict,
        batch_size: int,
        seed: Optional[int] = None,
        prefetch: int = 100,
    ):
        """Evaluate model based on resampled dataset for posterior resampling.

        node_1 + domain_1 -> encoder -> z_1 + domain_2 -> decoder -> reconstruction_2.

        Parameters
        ----------
        image_keys : np.array
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

    def init_model(
        self,
        optimizer: str = "adam",
        learning_rate: float = 0.0001,
        n_eval_nodes_per_graph: int = 32,
        l2_coef: float = 0.0,
        l1_coef: float = 0.0,
        use_source_type: bool = True,
        use_domain: bool = False,
        scale_node_size: bool = False,
        output_layer: str = "linear",
        **kwargs
    ):
        """Initialize a ModelInteractions object.

        Parameters
        ----------
        optimizer : str
            Optimizer.
        learning_rate : float
            Learning rate.
        l2_coef : float
            l2 regularization coefficient.
        l1_coef : float
            l1 regularization coefficient.
        n_eval_nodes_per_graph : int
            Number of nodes per graph.
        use_source_type : bool
            Whether to use source type.
        use_domain : bool
            Whether to use domain information.
        scale_node_size : bool
            Whether to scale output layer by node sizes.
        output_layer : str
            Output layer.
        kwargs
            Arbitrary keyword arguments.
        """
        self.n_eval_nodes_per_graph = n_eval_nodes_per_graph
        self.model = ModelLinear(
            input_shapes=(
                self.n_features_0,  # target_dim
                self.n_features_1,  # out_node_feature_dim
                self.n_features_0,  # source_dim
                self.n_eval_nodes_per_graph,  # in_node_dim
                self.n_node_covariates,  # categ_condition_dim
                self.n_domains,  # domain_dim
            ),
            l1_coef=l1_coef,
            l2_coef=l2_coef,
            use_source_type=use_source_type,
            use_domain=use_domain,
            scale_node_size=scale_node_size,
            output_layer=output_layer,
        )
        optimizer = tf.keras.optimizers.get(optimizer)
        tf.keras.backend.set_value(optimizer.lr, learning_rate)
        self._compile_model(optimizer=optimizer, output_layer=output_layer)
