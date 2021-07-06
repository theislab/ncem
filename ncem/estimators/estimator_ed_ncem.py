import tensorflow as tf

from ncem.estimators import EstimatorGraph
from ncem.models import ModelEDncem


class EstimatorEDncem(EstimatorGraph):
    """Estimator class for encoder-decoder NCEM models. Subclass of EstimatorGraph.
    Attributes:
        model_type (str):
        adj_type (str):
        cond_type (str):
        use_type_cond (bool):
        log_transform (bool):
        metrics (dict):
        n_eval_nodes_per_graph:
    """

    def __init__(
        self,
        cond_type: str = "gcn",
        use_type_cond: bool = True,
        log_transform: bool = False,
    ):
        """Initializes a EstimatorEDncem object.

        Args:
            cond_type (int): max, ind or gcn, graph layer used in conditional.
            use_type_cond (bool): whether to use the categorical cell type label in conditional.
            log_transform (bool): Whether to log transform h_1.
        Raises:
            ValueError: If `cond_type` is not recognized.
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
        """Initializes a ModelEDncem object.

        Args:
            optimizer (str): Optimizer.
            learning_rate (float): Learning rate.
            latent_dim (int): Latent dimension.
            dropout_rate (float): Dropout rate.
            l2_coef (float): l2 regularization coefficient.
            l1_coef (float): l1 regularization coefficient.
            enc_intermediate_dim (int): Encoder intermediate dimension.
            enc_depth (int): Encoder depth.
            dec_intermediate_dim (int): Decoder intermediate dimension.
            dec_depth (int): Decoder depth.
            cond_depth (int): Graph conditional depth.
            cond_dim (int): Graph conditional dimension.
            cond_dropout_rate (float): Graph conditional dropout rate.
            cond_activation: Graph conditional activation.
            cond_l2_reg (float): Graph conditional l2 regularization coefficient.
            cond_use_bias (bool): Graph conditional use bias.
            n_eval_nodes_per_graph (int): Number of nodes per graph.
            use_domain (bool): Whether to use domain information.
            scale_node_size (bool) Whether to scale output layer by node sizes.
            beta (float): Beta used in BetaScheduler.
            max_beta (float): Maximal beta used in BetaScheduler.
            pre_warm_up (int): Number of epochs in pre warm up.
            output_layer (str): Output layer.
            **kwargs: Arbitrary keyword arguments.

        Attributes:
            n_eval_nodes_per_graph (int):
            model:
            cond_depth (int): Graph conditional depth.
            beta (float): Beta used in BetaScheduler.
            max_beta (float): Maximal beta used in BetaScheduler.
            pre_warm_up (int): Number of epochs in pre warm up.
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
