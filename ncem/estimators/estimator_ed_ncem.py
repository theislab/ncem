import abc
import tensorflow as tf
from typing import Tuple, Union

from ncem.estimators import Estimator, EstimatorGraph, EstimatorNeighborhood
from ncem.estimators.base_estimator import transfer_layers
from ncem.models import ModelEDncem, ModelEd2Ncem
from ncem.models.layers.output_layers import IDENTIFIER_OUTPUT_LAYER


class EstimatorEDncemBase(Estimator, abc.ABC):
    model: Union[ModelEDncem, ModelEd2Ncem]

    def predict_embedding_any(self, img_keys, node_idx, batch_size: int = 1):
        """Predict embedding on any given data set.

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
        transfer_layers(model1=self.model.training_model, model2=self.model.encoder)
        return self.model.encoder.predict(ds)

    def get_decoding_weights(self):
        layer_name_out = [x.name for x in self.model.training_model.layers if IDENTIFIER_OUTPUT_LAYER in x.name]
        assert len(layer_name_out) == 1, "there should only be one output layer for this operation"
        w = self.model.training_model.get_layer(name=layer_name_out[0]).get_weights()
        return w


class EstimatorEDncem(EstimatorGraph, EstimatorEDncemBase):
    """Estimator class for encoder-decoder NCEM models. Subclass of EstimatorGraph."""

    def __init__(
        self,
        cond_type: str = "gcn",
        use_type_cond: bool = True,
        log_transform: bool = False,
    ):
        """Initialize a EstimatorEDncem object.

        Parameters
        ----------
        cond_type : str
            Max, ind or gcn, graph layer used in conditional.
        use_type_cond : bool
            Whether to use the categorical cell type label in conditional.
        log_transform : bool
            Whether to log transform h_1.

        Raises
        ------
        ValueError
            If `cond_type` is not recognized.
        """
        super(EstimatorEDncem, self).__init__()
        self.model_type = "ed_ncem"
        if cond_type == "gcn":
            self.adj_type = "scaled"
        elif cond_type == "max":
            self.adj_type = "full"
        else:
            raise ValueError("cond_type %s not recognized" % cond_type)
        self.cond_type = cond_type
        self.use_type_cond = use_type_cond
        self.log_transform = log_transform
        self.metrics = {"np": [], "tf": []}
        self.n_eval_nodes_per_graph = None
        self.cond_depth = None

    def init_model(
        self,
        optimizer: str = "adam",
        learning_rate: float = 0.0001,
        latent_dim: int = 10,
        dropout_rate: float = 0.1,
        l2_coef: float = 0.0,
        l1_coef: float = 0.0,
        enc_intermediate_dim: int = 128,
        enc_depth: int = 2,
        dec_intermediate_dim: int = 128,
        dec_depth: int = 2,
        cond_depth: int = 1,
        cond_dim: int = 8,
        cond_dropout_rate: float = 0.1,
        cond_activation: str = "relu",
        cond_l2_reg: float = 0.0,
        cond_use_bias: bool = False,
        n_eval_nodes_per_graph: int = 32,
        use_domain: bool = False,
        scale_node_size: bool = True,
        beta: float = 0.01,
        max_beta: float = 1.0,
        pre_warm_up: int = 0,
        output_layer: str = "gaussian",
        **kwargs
    ):
        """Initialize a ModelEDncem object.

        Parameters
        ----------
        optimizer : str
            Optimizer.
        learning_rate : float
            Learning rate.
        latent_dim : int
            Latent dimension.
        dropout_rate : float
            Dropout.
        l2_coef : float
            l2 regularization coefficient.
        l1_coef : float
            l1 regularization coefficient.
        enc_intermediate_dim : int
            Encoder intermediate dimension.
        enc_depth : int
            Encoder depth.
        dec_intermediate_dim : int
            Decoder intermediate dimension.
        dec_depth : int
            Decoder depth.
        cond_depth : int
            Graph conditional depth.
        cond_dim : int
            Graph conditional dimension.
        cond_dropout_rate : float
            Graph conditional dropout rate.
        cond_activation : str
            Graph conditional activation.
        cond_l2_reg : float
            Graph conditional l2 regularization coefficient.
        cond_use_bias : bool
            Graph conditional use bias.
        n_eval_nodes_per_graph : int
            Number of nodes per graph.
        use_domain : bool
            Whether to use domain information.
        scale_node_size : bool
            Whether to scale output layer by node sizes.
        beta : float
            Beta used in BetaScheduler.
        max_beta : float
            Maximal beta used in BetaScheduler.
        pre_warm_up : int
            Number of epochs in pre warm up.
        output_layer : str
            Output layer.
        kwargs
            Arbitrary keyword arguments.
        """
        self.n_eval_nodes_per_graph = n_eval_nodes_per_graph
        self.model = ModelEDncem(
            input_shapes=(
                self.n_features_0,
                self.n_features_1,
                self.max_nodes,
                self.n_eval_nodes_per_graph,
                self.n_node_covariates,
                self.n_domains,
            ),
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            l2_coef=l2_coef,
            l1_coef=l1_coef,
            enc_intermediate_dim=enc_intermediate_dim,
            enc_depth=enc_depth,
            dec_intermediate_dim=dec_intermediate_dim,
            dec_depth=dec_depth,
            cond_type=self.cond_type,
            cond_depth=cond_depth,
            cond_dim=cond_dim,
            cond_dropout_rate=cond_dropout_rate,
            cond_activation=cond_activation,
            cond_l2_reg=cond_l2_reg,
            cond_use_bias=cond_use_bias,
            use_domain=use_domain,
            use_type_cond=self.use_type_cond,
            scale_node_size=scale_node_size,
            output_layer=output_layer,
        )
        optimizer = tf.keras.optimizers.get(optimizer)
        tf.keras.backend.set_value(optimizer.lr, learning_rate)
        self.cond_depth = cond_depth
        self.beta = beta
        self.max_beta = max_beta
        self.pre_warm_up = pre_warm_up
        self._compile_model(optimizer=optimizer, output_layer=output_layer)


class EstimatorEdNcemNeighborhood(EstimatorNeighborhood, EstimatorEDncemBase):
    """Estimator class for encoder-decoder NCEM models with single graph layer. Subclass of EstimatorNeighborhood."""

    def __init__(
        self,
        cond_type: str,
        use_type_cond: bool = True,
        log_transform: bool = False,
    ):
        """Initialize a EstimatorEDncem object.

        Parameters
        ----------
        cond_type : str
            Max, ind or gcn, graph layer used in conditional.
        use_type_cond : bool
            Whether to use the categorical cell type label in conditional.
        log_transform : bool
            Whether to log transform h_1.

        Raises
        ------
        ValueError
            If `cond_type` is not recognized.
        """
        super(EstimatorEdNcemNeighborhood, self).__init__()
        self.model_type = "ed_ncem"
        if cond_type in ["gat", "lr_gat", "max", "gcn", "none"]:
            self.adj_type = "full"
        else:
            raise ValueError("cond_type %s not recognized" % cond_type)
        self.cond_type = cond_type
        self.use_type_cond = use_type_cond
        self.log_transform = log_transform
        self.metrics = {"np": [], "tf": []}
        self.n_eval_nodes_per_graph = None

    def init_model(
        self,
        optimizer: str,
        learning_rate: float,
        latent_dim: Tuple[int],
        dropout_rate: float,
        l2_coef: float,
        l1_coef: float,
        n_eval_nodes_per_graph: int,
        use_domain: bool,
        scale_node_size: bool,
        output_layer: str,
        dec_intermediate_dim: int,
        dec_n_hidden: int,
        dec_dropout_rate: float,
        dec_l1_coef: float,
        dec_l2_coef: float,
        dec_use_batch_norm: bool,
        **kwargs
    ):
        self.n_eval_nodes_per_graph = n_eval_nodes_per_graph
        self.model = ModelEd2Ncem(
            input_shapes=(
                self.n_features_in,
                self.n_features_1,
                self.n_eval_nodes_per_graph,
                self.n_neighbors_padded,
                self.n_node_covariates,
                self.n_domains,
            ),
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            l2_coef=l2_coef,
            l1_coef=l1_coef,
            use_domain=use_domain,
            use_type_cond=self.use_type_cond,
            scale_node_size=scale_node_size,
            output_layer=output_layer,
            cond_type=self.cond_type,
            dec_intermediate_dim=dec_intermediate_dim,
            dec_n_hidden=dec_n_hidden,
            dec_dropout_rate=dec_dropout_rate,
            dec_l1_coef=dec_l1_coef,
            dec_l2_coef=dec_l2_coef,
            dec_use_batch_norm=dec_use_batch_norm,
        )
        optimizer = tf.keras.optimizers.get(optimizer)
        tf.keras.backend.set_value(optimizer.lr, learning_rate)
        self._compile_model(optimizer=optimizer, output_layer=output_layer)
