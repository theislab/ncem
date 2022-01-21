import tensorflow as tf

from ncem.models.layers import LinearConstDispOutput, LinearOutput


class ModelLinear:
    """Model class for linear model, baseline and spatial model.

    Attributes:
        args (dict):
        training_model:

    Raises:
        ValueError: If `output_layer` is not recognized.
    """

    def __init__(
        self,
        input_shapes,
        l2_coef: float = 0.0,
        l1_coef: float = 0.0,
        use_proportions: bool = False,
        scale_node_size: bool = False,
        output_layer: str = "linear",
        **kwargs
    ):
        """Initialize linear model.

        Parameters
        ----------
        input_shapes
            input_shapes.
        l2_coef : float
            l2 regularization coefficient.
        l1_coef : float
            l1 regularization coefficient.
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

        Raises
        ------
        ValueError
            If `cond_type` or `output_layer` is not recognized.
        """
        super().__init__()
        self.args = {
            "input_shapes": input_shapes,
            "l2_coef": l2_coef,
            "l1_coef": l1_coef,
            "use_proportions": use_proportions,
            "scale_node_size": scale_node_size,
            "output_layer": output_layer,
        }
        in_node_dim = input_shapes[0]
        feature_dim = input_shapes[1]
        cell_dim = input_shapes[2]

        input_spot = tf.keras.Input(shape=(in_node_dim, feature_dim), name="spot_expression")  # spot expression
        input_celltype = tf.keras.Input(shape=(in_node_dim, cell_dim), name="source")
        proportions = tf.keras.Input(shape=(in_node_dim, cell_dim), name="proportions")  # proportions in spot

        # node size - reconstruction: Input Tensor - shape=(None, N, 1)  - ToDo?
        input_node_size = tf.keras.Input(shape=(in_node_dim, 1), name="node_size_reconstruct")

        if use_proportions:
            x = tf.concat([input_spot, input_celltype, proportions], axis=-1, name="input_concat")
        else:
            x = tf.concat([input_spot, input_celltype], axis=-1, name="input_concat")
        x = tf.reshape(x, [-1, x.shape[-1]], name="input_reshape")  # bs * n x (neighbour_embedding + categ_cond)

        output = tf.keras.layers.Dense(
            units=feature_dim,
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_coef, l2=l2_coef),
            name="LinearLayer",
        )(
            x
        )  # bs * n x genes
        output = tf.reshape(output, [-1, in_node_dim, feature_dim], name="output_reshape")  # bs x n x genes

        if output_layer == "linear":
            output = LinearOutput(use_node_scale=scale_node_size, name="LinearOutput")((output, input_node_size))
        elif output_layer == "linear_const_disp":
            output = LinearConstDispOutput(use_node_scale=scale_node_size, name="LinearOutput")(
                (output, input_node_size)
            )
        else:
            raise ValueError("tried to access a non-supported output layer %s" % output_layer)

        output_concat = tf.keras.layers.Concatenate(axis=2, name="reconstruction")(output)

        self.training_model = tf.keras.Model(
            inputs=[input_spot, input_celltype, proportions, input_node_size],
            outputs=output_concat,
            name="linear_deconvolution_model",
        )
