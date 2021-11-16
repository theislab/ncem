from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ncem.estimators.base_estimator import Estimator


class EstimatorNeighborhood(Estimator):
    """EstimatorGraph class for spatial models of the nieghborhood only (not full graph)."""

    n_features_in: int
    _n_neighbors_padded: Union[int, None]
    h0_in: bool
    features: list
    target_feature_names: list
    neighbor_feature_names: list
    idx_target_features: np.ndarray
    idx_neighbor_features: np.ndarray

    def __init__(self):
        super(EstimatorNeighborhood, self).__init__()
        self._n_neighbors_padded = None

    def set_input_features(self, h0_in=True, target_feature_names=None, neighbor_feature_names=None):
        """
        Need to run this before compiling model.

        Returns:
        """
        self.h0_in = h0_in
        if self.h0_in:
            assert target_feature_names is None
            assert neighbor_feature_names is None
            self.n_features_in = self.n_features_0
        else:
            features = self.data.var_names.tolist()
            self.features = features
            self.target_feature_names = target_feature_names
            self.neighbor_feature_names = neighbor_feature_names
            self.idx_target_features = np.array([features.index(x) for x in target_feature_names])
            self.idx_neighbor_features = np.array([features.index(x) for x in neighbor_feature_names])
            assert len(self.idx_target_features) == len(self.idx_neighbor_features)
            #assert len(set(self.idx_target_features.tolist()).intersection(set(self.idx_neighbor_features.tolist()))) == 0
            self.n_features_in = len(self.idx_target_features)

    @property
    def n_neighbors_padded(self):
        if self._n_neighbors_padded is None:
            self._n_neighbors_padded = int(np.max(np.asarray([
                np.max(np.asarray(np.sum(a, axis=1)).flatten()) for a in self.a.values()
            ])))
        return self._n_neighbors_padded

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
        # target node features
        h_targets = tf.TensorSpec(
            shape=(self.n_eval_nodes_per_graph, self.n_features_in), dtype=tf.float32
        )
        # neighbor node features
        h_neighbors = tf.TensorSpec(
            shape=(self.n_eval_nodes_per_graph, self.n_neighbors_padded, self.n_features_in), dtype=tf.float32
        )
        sf = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, 1), dtype=tf.float32)  # input node size factors
        # node-level covariates
        node_covar = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, self.n_node_covariates), dtype=tf.float32)
        # adjacency matrix
        a = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, self.n_neighbors_padded), dtype=tf.float32)
        # domain
        domain = tf.TensorSpec(shape=(self.n_domains,), dtype=tf.int32)
        # node features to reconstruct
        reconstruction = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, self.n_features_1), dtype=tf.float32)
        # dummy for kl loss
        kl_dummy = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph,), dtype=tf.float32)

        if self.vi_model:
            if resampled:
                output_signature = (
                    (h_targets, h_neighbors, sf, a, node_covar, domain),
                    (reconstruction, kl_dummy),
                    (h_targets, h_neighbors, sf, a, node_covar, domain),
                    (reconstruction, kl_dummy),
                )
            else:
                output_signature = ((h_targets, h_neighbors, sf, a, node_covar, domain),
                                    (reconstruction, kl_dummy))
        else:
            if resampled:
                output_signature = (
                    (h_targets, h_neighbors, sf, a, node_covar, domain),
                    reconstruction,
                    (h_targets, h_neighbors, sf, a, node_covar, domain),
                    reconstruction,
                )
            else:
                output_signature = ((h_targets, h_neighbors, sf, a, node_covar, domain),
                                    reconstruction)
        # print(output_signature)
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
                            nodes_idx[key][self.n_eval_nodes_per_graph * i: self.n_eval_nodes_per_graph * (i + 1)],
                            dtype=np.int32,
                        )
                        for i in range(len(nodes_idx[key]) // self.n_eval_nodes_per_graph)
                    ]

                for indices in index_list:
                    h_out = self.h_1[key][idx_nodes[indices], :]
                    if self.h0_in:
                        h_targets = self.h_0[key][idx_nodes[indices], :]
                    else:
                        h_targets = self.h_1[key][idx_nodes[indices], :][:, self.idx_target_features]
                    h_neighbors = []
                    a_neighborhood = np.zeros((self.n_eval_nodes_per_graph, self.n_neighbors_padded), "float32")
                    for i, j in enumerate(idx_nodes[indices]):
                        a_j = np.asarray(self.a[key][j, :].todense()).flatten()
                        idx_neighbors = np.where(a_j > 0.)[0]
                        if self.h0_in:
                            h_neighbors_j = self.h_0[key][idx_neighbors, :]
                        else:
                            h_neighbors_j = self.h_1[key][idx_neighbors, :][:, self.idx_neighbor_features]
                        h_neighbors_j = np.expand_dims(h_neighbors_j, axis=0)
                        # Pad neighborhoods:
                        diff = self.n_neighbors_padded - h_neighbors_j.shape[1]
                        zeros = np.zeros((1, diff, h_neighbors_j.shape[2]), dtype="float32")
                        h_neighbors_j = np.concatenate([h_neighbors_j, zeros], axis=1)
                        h_neighbors.append(h_neighbors_j)
                        a_neighborhood[i, :len(idx_neighbors)] = a_j[idx_neighbors]
                    h_neighbors = np.concatenate(h_neighbors, axis=0)
                    if self.log_transform:
                        h_targets = np.log(h_targets + 1.0)
                        h_neighbors = np.log(h_neighbors + 1.0)

                    node_covar = self.node_covar[key][idx_nodes][indices, :]
                    sf = np.expand_dims(self.size_factors[key][idx_nodes][indices], axis=1)

                    g = np.zeros((self.n_domains,), dtype="int32")
                    g[self.domains[key]] = 1

                    if self.vi_model:
                        kl_dummy = np.zeros((self.n_eval_nodes_per_graph,), dtype="float32")
                        yield (h_targets, h_neighbors, sf, a_neighborhood, node_covar, g), (h_out, kl_dummy)
                    else:
                        yield (h_targets, h_neighbors, sf, a_neighborhood, node_covar, g), h_out

        output_signature = self._get_output_signature(resampled=False)

        dataset = tf.data.Dataset.from_generator(generator=generator, output_signature=output_signature)
        if train:
            if shuffle_buffer_size is not None:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=None, reshuffle_each_iteration=True)
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch)
        return dataset
