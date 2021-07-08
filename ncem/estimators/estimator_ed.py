import tensorflow as tf

from ncem.estimators import EstimatorNoGraph
from ncem.models import ModelED


class EstimatorED(EstimatorNoGraph):
    """Estimator class for encoder-decoder models. Subclass of EstimatorNoGraph."""

    def __init__(
        self,
        use_type_cond: bool = True,
        log_transform: bool = False,
    ):
        """Initialize a EstimatorED object.

        Parameters
        ----------
        use_type_cond : bool
            whether to use the categorical cell type label in conditional.
        log_transform : bool
            Whether to log transform h_1.
        """
        super(EstimatorED, self).__init__()
        self.adj_type = "none"
        self.model_type = "ed"
        self.use_type_cond = use_type_cond
        self.log_transform = log_transform
        self.metrics = {"np": [], "tf": []}
        self.n_eval_nodes_per_graph = None

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
        n_eval_nodes_per_graph: int = 32,
        use_domain: bool = False,
        scale_node_size: bool = True,
        beta: float = 0.01,
        max_beta: float = 1.0,
        pre_warm_up: int = 0,
        output_layer: str = "gaussian",
        **kwargs
    ):
        """Initialize a ModelED object.

        Parameters
        ----------
        optimizer : str
            Optimizer.
        learning_rate : float
            Learning rate.
        latent_dim : int
            Latent dimension.
        dropout_rate : float
            Dropout rate.
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
        self.model = ModelED(
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
            use_domain=use_domain,
            use_type_cond=self.use_type_cond,
            scale_node_size=scale_node_size,
            output_layer=output_layer,
        )
        optimizer = tf.keras.optimizers.get(optimizer)
        tf.keras.backend.set_value(optimizer.lr, learning_rate)
        self.beta = beta
        self.max_beta = max_beta
        self.pre_warm_up = pre_warm_up
        self._compile_model(optimizer=optimizer, output_layer=output_layer)
