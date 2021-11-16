from typing import Union

import tensorflow as tf

from ncem.models.layers import (Decoder, Encoder, GCNLayer, MaxLayer, get_out)


class ModelEDncem:
    """Model class for NCEM encoder-decoder with graph layer IND (MAX) or GCN."""

    def __init__(
        self,
        input_shapes,
        latent_dim: int = 10,
        dropout_rate: float = 0.1,
        l2_coef: float = 0.0,
        l1_coef: float = 0.0,
        enc_intermediate_dim: int = 128,
        enc_depth: int = 2,
        dec_intermediate_dim: int = 128,
        dec_depth: int = 2,
        cond_type: str = "gcn",
        cond_depth: int = 1,
        cond_dim: int = 8,
        cond_dropout_rate: float = 0.1,
        cond_activation: Union[str, tf.keras.layers.Layer] = "relu",
        cond_l2_reg: float = 0.0,
        cond_use_bias: bool = True,
        use_domain: bool = False,
        use_type_cond: bool = False,
        scale_node_size: bool = False,
        output_layer: str = "gaussian",
        **kwargs,
    ):
        """Initialize encoder-decoder NCEM model.

        Parameters
        ----------
        input_shapes
            Input shapes.
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
        cond_type : str
            Graph conditional type.
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
        use_domain : bool
            Whether to use domain information.
        use_type_cond : bool
            whether to use the categorical cell type label in conditional.
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
            "latent_dim": latent_dim,
            "dropout_rate": dropout_rate,
            "l2_coef": l2_coef,
            "l1_coef": l1_coef,
            "enc_intermediate_dim": enc_intermediate_dim,
            "enc_depth": enc_depth,
            "dec_intermediate_dim": dec_intermediate_dim,
            "dec_depth": dec_depth,
            "cond_type": cond_type,
            "cond_depth": cond_depth,
            "cond_dim": cond_dim,
            "cond_dropout_rate": cond_dropout_rate,
            "cond_activation": cond_activation
            if not isinstance(cond_activation, tf.keras.layers.Layer)
            else cond_activation.name,
            "cond_l2_reg": cond_l2_reg,
            "cond_use_bias": cond_use_bias,
            "use_domain": use_domain,
            "use_type_cond": use_type_cond,
            "scale_node_size": scale_node_size,
            "output_layer": output_layer,
        }
        in_node_feature_dim = input_shapes[0]
        out_node_feature_dim = input_shapes[1]
        graph_dim = input_shapes[2]
        in_node_dim = input_shapes[3]
        categ_condition_dim = input_shapes[4]
        domain_dim = input_shapes[5]

        # node features - reconstruction: Input Tensor - shape=(None, N, F)
        input_x_reconstruct = tf.keras.Input(
            shape=(in_node_dim, out_node_feature_dim), name="node_features_reconstruct"
        )
        # node size - reconstruction: Input Tensor - shape=(None, N, 1)
        input_node_size = tf.keras.Input(shape=(in_node_dim, 1), name="node_size_reconstruct")
        # node features - node representation of other nodes: Input Tensor - shape=(None, N, F)
        input_x_cond = tf.keras.Input(shape=(in_node_dim, in_node_feature_dim), name="node_features_cond")
        # node features - node representation of other nodes: Input Tensor - shape=(None, N, F)
        input_x_cond_full = tf.keras.Input(shape=(graph_dim, in_node_feature_dim), name="node_features_cond_full")
        # adj_matrices - A: Input Tensor - shape=(None, N, N)
        input_a = tf.keras.Input(shape=(in_node_dim, graph_dim), name="adjacency_matrix", sparse=True)
        # full adj_matrices - A: Input Tensor - shape=(None, N, N)
        input_afull = tf.keras.Input(shape=(graph_dim, graph_dim), name="adjacency_matrix_full", sparse=True)
        # Categorical predictors: Input Tensor - shape=(None, N, P)
        input_categ_condition = tf.keras.Input(shape=(in_node_dim, categ_condition_dim), name="categorical_predictor")
        # domain information of graph - shape=(None, 1)
        input_g = tf.keras.layers.Input(shape=(domain_dim,), name="input_da_group", dtype="int32")

        # Decoder inputs:
        input_latent_sampling = tf.keras.Input(shape=(in_node_dim, latent_dim), name="z_sampling")
        latent_sampling_reshaped = tf.reshape(input_latent_sampling, [-1, latent_dim])

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

        if cond_depth > 1:
            print("using multi layer graph model")
        x_neighbour_embedding = input_x_cond_full
        if cond_type == "gcn" or cond_type == "gcn_full":
            for i in range(cond_depth - 1):
                cond_layer = GCNLayer(
                    output_dim=cond_dim,
                    dropout_rate=cond_dropout_rate,
                    activation=cond_activation,
                    l2_reg=cond_l2_reg,
                    use_bias=cond_use_bias,
                    name=f"conditional_layer_stack_{i}",
                )
                x_neighbour_embedding = cond_layer([x_neighbour_embedding, input_afull])
            # In last forward pass, only retain nodes that are to be decoded: not using the full adjacency matrix:
            cond_layer = GCNLayer(
                output_dim=cond_dim,
                dropout_rate=cond_dropout_rate,
                activation=cond_activation,
                l2_reg=cond_l2_reg,
                use_bias=cond_use_bias,
                name=f"conditional_layer_stack_{cond_depth}",
            )
            x_neighbour_embedding = cond_layer([x_neighbour_embedding, input_a])
        elif cond_type == "max":
            for i in range(cond_depth - 1):
                cond_layer = MaxLayer(name=f"conditional_layer_stack_{i}")
                x_neighbour_embedding = cond_layer([x_neighbour_embedding, input_afull])
            # In last forward pass, only retain nodes that are to be decoded: not using the full adjacency matrix:
            cond_layer = MaxLayer(name=f"conditional_layer_stack_{cond_depth}")
            x_neighbour_embedding = cond_layer([x_neighbour_embedding, input_a])
        else:
            raise ValueError("tried to access a non-supported conditional layer %s" % cond_type)

        inputs_encoder = (x_neighbour_embedding, categ_condition)

        self.encoder_model = Encoder(
            latent_dim=latent_dim,
            intermediate_dim=enc_intermediate_dim,
            dropout_rate=dropout_rate,
            n_hidden=enc_depth,
            l1_coef=l1_coef,
            l2_coef=l2_coef,
            probabilistic=False,
            use_type_cond=use_type_cond,
        )
        output_encoder = self.encoder_model(inputs_encoder)

        z, z_mean, z_log_var = output_encoder

        self.decoder_model = Decoder(
            intermediate_dim=dec_intermediate_dim,
            dropout_rate=dropout_rate,
            n_hidden=dec_depth,
            l1_coef=l1_coef,
            l2_coef=l2_coef,
            use_type_cond=use_type_cond,
        )
        output_decoder = self.decoder_model((z, categ_condition))
        sampling_decoder = self.decoder_model((latent_sampling_reshaped, categ_condition))

        output_decoder_layer = get_out(
            output_layer=output_layer, out_feature_dim=out_node_feature_dim, scale_node_size=scale_node_size
        )((output_decoder, input_node_size))
        output_sampling_decoder = get_out(
            output_layer=output_layer, out_feature_dim=out_node_feature_dim, scale_node_size=scale_node_size,
            name='sampling'
        )((sampling_decoder, input_node_size))

        output_decoder_concat = tf.keras.layers.Concatenate(axis=2, name="reconstruction")(output_decoder_layer)
        output_sampling_concat = tf.keras.layers.Concatenate(axis=2, name="reconstruction")(output_sampling_decoder)

        self.encoder = tf.keras.Model(
            inputs=[
                input_x_reconstruct,
                input_x_cond,
                input_x_cond_full,
                input_a,
                input_afull,
                input_categ_condition,
                input_g,
            ],
            outputs=output_encoder,
            name="encoder_ncem",
        )
        self.decoder = tf.keras.Model(
            inputs=[
                input_latent_sampling,
                input_node_size,
                input_x_cond,
                input_x_cond_full,
                input_a,
                input_afull,
                input_categ_condition,
                input_g,
            ],
            outputs=output_sampling_concat,
            name="decoder_ncem",
        )
        self.training_model = tf.keras.Model(
            inputs=[
                input_x_reconstruct,
                input_node_size,
                input_x_cond,
                input_x_cond_full,
                input_a,
                input_afull,
                input_categ_condition,
                input_g,
            ],
            outputs=output_decoder_concat,
            name="ed_ncem",
        )
