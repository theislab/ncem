from typing import Union

import numpy as np
import tensorflow as tf

from ncem.models.layers import (CondDecoder, CondEncoder, GCNLayer, MaxLayer, get_out, PreprocInput, SamplingPrior)


class ModelCVAEncem:
    """Model class for NCEM conditional variational autoencoder with graph layer IND (MAX) or GCN."""

    def __init__(
        self,
        input_shapes,
        latent_dim: int = 10,
        intermediate_dim_enc: int = 128,
        intermediate_dim_dec: int = 128,
        depth_enc: int = 1,
        depth_dec: int = 1,
        dropout_rate: float = 0.1,
        l2_coef: float = 0.0,
        l1_coef: float = 0.0,
        cond_type: str = "gcn",
        cond_depth: int = 1,
        cond_dim: int = 8,
        cond_dropout_rate: float = 0.1,
        cond_activation: Union[str, tf.keras.layers.Layer] = "relu",
        cond_l2_reg: float = 0.0,
        cond_use_bias: bool = True,
        use_domain: bool = False,
        scale_node_size: bool = False,
        use_type_cond: bool = True,
        use_batch_norm: bool = False,
        transform_input: bool = False,
        output_layer: str = "gaussian",
        **kwargs,
    ):
        """Initialize conditional variational autoencoder NCEM model.

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
        intermediate_dim_enc : int
            Encoder intermediate dimension.
        depth_enc : int
            Encoder depth.
        intermediate_dim_dec : int
            Decoder intermediate dimension.
        depth_dec : int
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
        use_batch_norm : bool
            Whether to use batch normalization.
        use_type_cond : bool
            whether to use the categorical cell type label in conditional.
        scale_node_size : bool
            Whether to scale output layer by node sizes.
        transform_input : bool
            Whether to transform input.
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
            "intermediate_dim_enc": intermediate_dim_enc,
            "intermediate_dim_dec": intermediate_dim_dec,
            "depth_enc": depth_enc,
            "depth_dec": depth_dec,
            "dropout_rate": dropout_rate,
            "l2_coef": l2_coef,
            "l1_coef": l1_coef,
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

        # N - maximum number of nodes in graph
        # M - number of nodes in graph evaluated in each batch
        # F - number of node features in reconstruction space
        # P - number of node covariate dimensions
        # node_features - reconstruction: Input Tensor - shape=(None, M, F)
        input_x_reconstruct = tf.keras.Input(
            shape=(in_node_dim, out_node_feature_dim), name="node_features_reconstruct"
        )
        # node size - reconstruction: Input Tensor - shape=(None, M, 1)
        input_node_size = tf.keras.Input(shape=(in_node_dim, 1), name="node_size_reconstruct")
        # node_features - conditions: Input Tensor - shape=(None, M, F)
        input_x_cond = tf.keras.Input(shape=(in_node_dim, in_node_feature_dim), name="node_features_cond")
        # node features - node representation of other nodes: Input Tensor - shape=(None, N, F)
        input_x_cond_full = tf.keras.Input(shape=(graph_dim, in_node_feature_dim), name="node_features_cond_full")
        # adj_matrices - A: Input Tensor - shape=(None, M, N)
        input_a = tf.keras.Input(shape=(in_node_dim, graph_dim), name="adjacency_matrix", sparse=True)
        # full adj_matrices - A: Input Tensor - shape=(None, N, N)
        input_afull = tf.keras.Input(shape=(graph_dim, graph_dim), name="adjacency_matrix_full", sparse=True)
        # Categorical predictors: Input Tensor - shape=(None, M, P)
        input_categ_condition = tf.keras.Input(shape=(in_node_dim, categ_condition_dim), name="categorical_predictor")
        # domain information of graph - shape=(None, 1)
        input_g = tf.keras.layers.Input(shape=(domain_dim,), name="input_da_group", dtype="int32")

        # Decoder inputs:
        # 1) Sample in mode:
        latent_sampling1 = SamplingPrior(width=latent_dim)(input_x_reconstruct)
        latent_sampling_reshaped1 = tf.reshape(latent_sampling1, [-1, latent_dim])
        # 2) Sample in data intput:
        input_latent_sampling2 = tf.keras.Input(shape=(in_node_dim, latent_dim), name="z_sampling")
        latent_sampling_reshaped2 = tf.reshape(input_latent_sampling2, [-1, latent_dim])

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
        if cond_type == "gcn":
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

        x = PreprocInput()(input_x_reconstruct) if transform_input else input_x_reconstruct

        self.encoder_model = CondEncoder(
            latent_dim=latent_dim,
            intermediate_dim=intermediate_dim_enc,
            dropout_rate=dropout_rate,
            n_hidden=depth_enc,
            l1_coef=l1_coef,
            l2_coef=l2_coef,
            use_type_cond=use_type_cond,
            use_batch_norm=use_batch_norm,
            probabilistic=True,
        )
        output_encoder = self.encoder_model((x, x_neighbour_embedding, categ_condition))

        # output_encoder
        z, z_mean, z_log_var = output_encoder
        latent_space = tf.keras.layers.Concatenate(axis=1, name="bottleneck")([z, z_mean, z_log_var])
        latent_space_sampling = tf.zeros_like(latent_space, name="bottleneck")
        latent_space2 = tf.keras.layers.Concatenate(axis=1, name="bottleneck")(
            [  # immitate latent_space tensor
                tf.zeros_like(input_latent_sampling2),
                tf.zeros_like(input_latent_sampling2),
                tf.zeros_like(input_latent_sampling2),
            ]
        )

        self.decoder_model = CondDecoder(
            intermediate_dim=intermediate_dim_dec,
            dropout_rate=dropout_rate,
            n_hidden=depth_dec,
            l1_coef=l1_coef,
            l2_coef=l2_coef,
            use_type_cond=use_type_cond,
            use_batch_norm=use_batch_norm,
        )
        output_decoder = self.decoder_model((z, x_neighbour_embedding, categ_condition))
        sampling_decoder1 = self.decoder_model((latent_sampling_reshaped1, x_neighbour_embedding, categ_condition))
        sampling_decoder2 = self.decoder_model((latent_sampling_reshaped2, x_neighbour_embedding, categ_condition))

        output_decoder_layer = get_out(
            output_layer=output_layer, out_feature_dim=out_node_feature_dim, scale_node_size=scale_node_size
        )((output_decoder, input_node_size))
        output_sampling_decoder1 = get_out(
            output_layer=output_layer, out_feature_dim=out_node_feature_dim, scale_node_size=scale_node_size,
            name='sampling1'
        )((sampling_decoder1, input_node_size))
        output_sampling_decoder2 = get_out(
            output_layer=output_layer, out_feature_dim=out_node_feature_dim, scale_node_size=scale_node_size,
            name='sampling2'
        )((sampling_decoder2, input_node_size))

        output_decoder_concat = tf.keras.layers.Concatenate(axis=2, name="reconstruction")(output_decoder_layer)
        output_sampling_concat1 = tf.keras.layers.Concatenate(axis=2, name="reconstruction")(output_sampling_decoder1)
        output_sampling_concat2 = tf.keras.layers.Concatenate(axis=2, name="reconstruction")(output_sampling_decoder2)

        self.graph_condition_model = tf.keras.Model(
            inputs=[
                input_x_reconstruct,
                input_x_cond,
                input_x_cond_full,
                input_a,
                input_afull,
                input_categ_condition,
                input_g,
            ],
            outputs=x_neighbour_embedding,
            name="graph_conditional",
        )
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
        self.decoder_sampling = tf.keras.Model(
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
            outputs=[output_sampling_concat1, latent_space_sampling],
            name="decoder_sampling_ncem",
        )
        self.decoder = tf.keras.Model(
            inputs=[
                input_latent_sampling2,
                input_node_size,
                input_x_cond,
                input_x_cond_full,
                input_a,
                input_afull,
                input_categ_condition,
                input_g,
            ],
            outputs=[output_sampling_concat2, latent_space2],
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
            outputs=[output_decoder_concat, latent_space],
            name="cvae_ncem",
        )

        # Add non-scaled ELBO to model as metric (ie no annealing or beta-VAE scaling):
        log2pi = tf.math.log(2.0 * np.pi)
        logqz_x = -0.5 * tf.reduce_mean(tf.square(z - z_mean) * tf.exp(-z_log_var) + z_log_var + log2pi)
        logpz = -0.5 * tf.reduce_mean(tf.square(z) + log2pi)
        d_kl = logqz_x - logpz
        loc, scale = output_decoder_layer
        if output_layer in ["gaussian", "gaussian_const_disp"]:
            neg_ll = tf.math.log(tf.sqrt(2 * np.math.pi) * scale) + 0.5 * tf.math.square(
                loc - input_x_reconstruct
            ) / tf.math.square(scale)
        elif output_layer in ["nb", "nb_const_disp", "nb_shared_disp"]:
            eta_loc = tf.math.log(loc)
            eta_scale = tf.math.log(scale)

            log_r_plus_mu = tf.math.log(scale + loc)

            ll = tf.math.lgamma(scale + input_x_reconstruct)
            ll = ll - tf.math.lgamma(input_x_reconstruct + tf.ones_like(input_x_reconstruct))
            ll = ll - tf.math.lgamma(scale)
            ll = (
                ll
                + tf.multiply(input_x_reconstruct, eta_loc - log_r_plus_mu)
                + tf.multiply(scale, eta_scale - log_r_plus_mu)
            )

            neg_ll = -tf.clip_by_value(ll, -300, 300, "log_probs")
        else:
            neg_ll = None
        neg_ll = tf.reduce_mean(tf.reduce_sum(neg_ll, axis=-1))
        self.training_model.add_metric(neg_ll + d_kl, name="elbo", aggregation="mean")
