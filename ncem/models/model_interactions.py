from typing import Union

import tensorflow as tf

from ncem.models.layers import (DenseInteractions, LinearConstDispOutput,
                                LinearOutput)


class ModelInteractions:
    """Model class for interaction model, baseline and spatial model."""

    def __init__(
        self,
        input_shapes,
        l2_coef: Union[float, None] = 0.0,
        l1_coef: Union[float, None] = 0.0,
        use_interactions: bool = False,
        use_domain: bool = False,
        scale_node_size: bool = False,
        output_layer: str = "linear",
        **kwargs
    ):
        """Initialize interaction model.

        Parameters
        ----------
        input_shapes
            input_shapes.
        l2_coef : float
            l2 regularization coefficient.
        l1_coef : float
            l1 regularization coefficient.
        use_interactions : bool
            Whether to use interactions.
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
            "use_interactions": use_interactions,
            "use_domain": use_domain,
            "scale_node_size": scale_node_size,
            "output_layer": output_layer,
        }
        target_dim = input_shapes[0]
        out_node_feature_dim = input_shapes[1]
        interaction_dim = input_shapes[2]
        in_node_dim = input_shapes[3]
        categ_condition_dim = input_shapes[4]
        domain_dim = input_shapes[5]

        input_target = tf.keras.Input(shape=(in_node_dim, target_dim), name="target")
        input_interaction = tf.keras.Input(shape=(in_node_dim, interaction_dim), name="interaction", sparse=True)
        # node size - reconstruction: Input Tensor - shape=(None, N, 1)
        input_node_size = tf.keras.Input(shape=(in_node_dim, 1), name="node_size_reconstruct")
        # Categorical predictors: Input Tensor - shape=(None, N, P)
        input_categ_condition = tf.keras.Input(shape=(in_node_dim, categ_condition_dim), name="categorical_predictor")
        # domain information of graph - shape=(None, 1)
        input_g = tf.keras.layers.Input(shape=(domain_dim,), name="input_da_group", dtype="int32")

        if use_domain:
            categ_condition = tf.concat(
                [
                    input_categ_condition,
                    tf.tile(tf.expand_dims(tf.cast(input_g, dtype="float32"), axis=-2), [1, in_node_dim, 1]),
                ],
                axis=-1,
            )
        else:
            categ_condition = input_categ_condition

        if use_interactions:
            interactions = DenseInteractions(in_node_dim=in_node_dim, interaction_dim=interaction_dim)(
                input_interaction
            )
            x = tf.concat([input_target, interactions, categ_condition], axis=-1, name="input_concat")
        else:
            x = tf.concat([input_target, categ_condition], axis=-1, name="input_concat")
        x = tf.reshape(x, [-1, x.shape[-1]], name="input_reshape")  # bs * n x (neighbour_embedding + categ_cond)

        output = tf.keras.layers.Dense(
            units=out_node_feature_dim,
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_coef, l2=l2_coef),
            name="LinearLayer",
        )(
            x
        )  # bs * n x genes
        output = tf.reshape(output, [-1, in_node_dim, out_node_feature_dim], name="output_reshape")  # bs x n x genes

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
            inputs=[input_target, input_interaction, input_node_size, input_categ_condition, input_g],
            outputs=output_concat,
            name="interaction_linear_model",
        )
