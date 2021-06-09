import numpy as np
import tensorflow as tf

from ncem.models.layers import (Decoder, Encoder, GaussianOutput,
                                NegBinConstDispOutput, NegBinOutput,
                                NegBinSharedDispOutput, SamplingPrior)


class ModelLVM:
    def __init__(
        self,
        input_shapes,
        latent_dim: int = 10,
        dropout_rate: float = 0.1,
        l2_coef: float = 0.0,
        l1_coef: float = 0.0,
        enc_intermediate_dim: int = 128,
        enc_depth: int = 2,
        bottleneck_activation: str = "linear",
        dec_intermediate_dim: int = 128,
        dec_depth: int = 2,
        use_domain: bool = False,
        use_type_cond: bool = True,
        scale_node_size: bool = False,
        output_layer: str = "gaussian",
        probabilistic: bool = True,
    ):
        super().__init__()
        self.args = {
            "input_shapes": input_shapes,
            "latent_dim": latent_dim,
            "dropout_rate": dropout_rate,
            "l2_coef": l2_coef,
            "l1_coef": l1_coef,
            "enc_intermediate_dim": enc_intermediate_dim,
            "enc_depth": enc_depth,
            "bottleneck_activation": bottleneck_activation,
            "dec_intermediate_dim": dec_intermediate_dim,
            "dec_depth": dec_depth,
            "use_domain": use_domain,
            "use_type_cond": use_type_cond,
            "scale_node_size": scale_node_size,
            "output_layer": output_layer,
            "probabilistic": probabilistic,
        }
        in_node_feature_dim = input_shapes[0]
        out_node_feature_dim = input_shapes[1]
        graph_dim = input_shapes[2]
        in_node_dim = input_shapes[3]
        categ_condition_dim = input_shapes[4]
        domain_dim = input_shapes[5]

        # node_features - H: Input Tensor - shape=(None, N, F)
        input_x = tf.keras.Input(shape=(in_node_dim, out_node_feature_dim), name="node_features")
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

        # Decoder inputs:
        # 1) Sample in mode:
        if probabilistic:
            # This does not make sense in a model in which the bottleneck activation is not forced to be close to a
            # unit gaussian, ie with a non-constrained bottleneck.
            latent_sampling1 = SamplingPrior(width=latent_dim)(input_x)
            latent_sampling_reshaped1 = tf.reshape(latent_sampling1, [-1, latent_dim])
        # 2) Sample in data intput:
        input_latent_sampling2 = tf.keras.Input(shape=(in_node_dim, latent_dim), name="z_sampling")
        latent_sampling_reshaped2 = tf.reshape(input_latent_sampling2, [-1, latent_dim])

        inputs_encoder = (categ_condition, tf.zeros_like(categ_condition))
        self.encoder_model = Encoder(
            latent_dim=latent_dim,
            intermediate_dim=enc_intermediate_dim,
            dropout_rate=dropout_rate,
            n_hidden=enc_depth,
            l1_coef=l1_coef,
            l2_coef=l2_coef,
            bottleneck_activation=bottleneck_activation,
            probabilistic=probabilistic,
            use_type_cond=False,  # type condition is in direct input to encoder, see above
        )
        output_encoder = self.encoder_model(inputs_encoder)

        z, z_mean, z_log_var = output_encoder
        if probabilistic:
            latent_space = tf.keras.layers.Concatenate(axis=1, name="bottleneck")([z, z_mean, z_log_var])
            latent_space_sampling = tf.zeros_like(latent_space, name="bottleneck")

        self.decoder_model = Decoder(
            intermediate_dim=dec_intermediate_dim,
            dropout_rate=dropout_rate,
            n_hidden=dec_depth,
            l1_coef=l1_coef,
            l2_coef=l2_coef,
            use_type_cond=use_type_cond,
        )
        output_decoder = self.decoder_model((z, categ_condition))
        if probabilistic:
            sampling_decoder1 = self.decoder_model((latent_sampling_reshaped1, categ_condition))
        sampling_decoder2 = self.decoder_model((latent_sampling_reshaped2, categ_condition))

        if output_layer == "gaussian":
            output_decoder_layer = GaussianOutput(
                original_dim=out_node_feature_dim,
                ncells_selected=in_node_dim,
                use_node_scale=scale_node_size,
                name="GaussianOutput_decoder",
            )((output_decoder, input_node_size))
            if probabilistic:
                output_sampling_decoder1 = GaussianOutput(
                    original_dim=out_node_feature_dim,
                    ncells_selected=in_node_dim,
                    use_node_scale=scale_node_size,
                    name="GaussianOutput_sampling",
                )((sampling_decoder1, input_node_size))
            output_sampling_decoder2 = GaussianOutput(
                original_dim=out_node_feature_dim,
                ncells_selected=in_node_dim,
                use_node_scale=scale_node_size,
                name="GaussianOutput_sampling",
            )((sampling_decoder2, input_node_size))
        elif output_layer == "nb":
            output_decoder_layer = NegBinOutput(
                original_dim=out_node_feature_dim,
                ncells_selected=in_node_dim,
                use_node_scale=scale_node_size,
                name="NegBinOutput_decoder",
            )((output_decoder, input_node_size))
            if probabilistic:
                output_sampling_decoder1 = NegBinOutput(
                    original_dim=out_node_feature_dim,
                    ncells_selected=in_node_dim,
                    use_node_scale=scale_node_size,
                    name="NegBinOutput_sampling",
                )((sampling_decoder1, input_node_size))
            output_sampling_decoder2 = NegBinOutput(
                original_dim=out_node_feature_dim,
                ncells_selected=in_node_dim,
                use_node_scale=scale_node_size,
                name="NegBinOutput_sampling",
            )((sampling_decoder2, input_node_size))
        elif output_layer == "nb_shared_disp":
            output_decoder_layer = NegBinSharedDispOutput(
                original_dim=out_node_feature_dim,
                ncells_selected=in_node_dim,
                use_node_scale=scale_node_size,
                name="NegBinSharedDispOutput_decoder",
            )((output_decoder, input_node_size))
            if probabilistic:
                output_sampling_decoder1 = NegBinSharedDispOutput(
                    original_dim=out_node_feature_dim,
                    ncells_selected=in_node_dim,
                    use_node_scale=scale_node_size,
                    name="NegBinSharedDispOutput_sampling",
                )((sampling_decoder1, input_node_size))
            output_sampling_decoder2 = NegBinSharedDispOutput(
                original_dim=out_node_feature_dim,
                ncells_selected=in_node_dim,
                use_node_scale=scale_node_size,
                name="NegBinSharedDispOutput_sampling",
            )((sampling_decoder2, input_node_size))
        elif output_layer == "nb_const_disp":
            output_decoder_layer = NegBinConstDispOutput(
                original_dim=out_node_feature_dim,
                ncells_selected=in_node_dim,
                use_node_scale=scale_node_size,
                name="NegBinConstDispOutput_decoder",
            )((output_decoder, input_node_size))
            if probabilistic:
                output_sampling_decoder1 = NegBinConstDispOutput(
                    original_dim=out_node_feature_dim,
                    ncells_selected=in_node_dim,
                    use_node_scale=scale_node_size,
                    name="NegBinConstDispOutput_sampling",
                )((sampling_decoder1, input_node_size))
            output_sampling_decoder2 = NegBinConstDispOutput(
                original_dim=out_node_feature_dim,
                ncells_selected=in_node_dim,
                use_node_scale=scale_node_size,
                name="NegBinConstDispOutput_sampling",
            )((sampling_decoder2, input_node_size))
        else:
            raise ValueError("tried to access a non-supported output layer %s" % output_layer)

        output_decoder_concat = tf.keras.layers.Concatenate(axis=2, name="reconstruction")(output_decoder_layer)
        if probabilistic:
            output_sampling_concat1 = tf.keras.layers.Concatenate(axis=2, name="reconstruction")(
                output_sampling_decoder1
            )
        output_sampling_concat2 = tf.keras.layers.Concatenate(axis=2, name="reconstruction")(output_sampling_decoder2)

        self.encoder = tf.keras.Model(
            inputs=[input_x, input_categ_condition, input_g], outputs=output_encoder, name="encoder"
        )
        if probabilistic:
            self.decoder_sampling = tf.keras.Model(
                inputs=[input_x, input_node_size, input_categ_condition, input_g],
                outputs=[output_sampling_concat1, latent_space_sampling],
                name="decoder-sampling",
            )
            self.decoder = tf.keras.Model(
                inputs=[input_latent_sampling2, input_node_size, input_categ_condition, input_g],
                outputs=[output_sampling_concat2, latent_space_sampling],
                name="decoder",
            )
            self.training_model = tf.keras.Model(
                inputs=[input_x, input_node_size, input_categ_condition, input_g],
                outputs=[output_decoder_concat, latent_space],
                name="lvm",
            )
        else:
            self.decoder_sampling = None
            self.decoder = tf.keras.Model(
                inputs=[input_latent_sampling2, input_node_size, input_categ_condition, input_g],
                outputs=output_sampling_concat2,
                name="decoder",
            )
            self.training_model = tf.keras.Model(
                inputs=[input_x, input_node_size, input_categ_condition, input_g],
                outputs=output_decoder_concat,
                name="lvm",
            )

        if probabilistic:
            # Add non-scaled ELBO to model as metric (ie no annealing or beta-VAE scaling):
            log2pi = tf.math.log(2.0 * np.pi)
            logqz_x = -0.5 * tf.reduce_mean(
                tf.reduce_sum(tf.square(z - z_mean) * tf.exp(-z_log_var) + z_log_var + log2pi, axis=-1)
            )
            logpz = -0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z) + log2pi, axis=-1))
            d_kl = logqz_x - logpz

            loc, scale = output_decoder_layer
            if output_layer == "gaussian":
                neg_ll = tf.math.log(tf.sqrt(2 * np.math.pi) * scale) + 0.5 * tf.math.square(
                    loc - input_x
                ) / tf.math.square(scale)
            elif output_layer == "nb" or output_layer == "nb_const_disp" or output_layer == "nb_shared_disp":
                eta_loc = tf.math.log(loc)
                eta_scale = tf.math.log(scale)

                log_r_plus_mu = tf.math.log(scale + loc)

                ll = tf.math.lgamma(scale + input_x)
                ll = ll - tf.math.lgamma(input_x + tf.ones_like(input_x))
                ll = ll - tf.math.lgamma(scale)
                ll = ll + tf.multiply(input_x, eta_loc - log_r_plus_mu) + tf.multiply(scale, eta_scale - log_r_plus_mu)

                neg_ll = -tf.clip_by_value(ll, -300, 300, "log_probs")
            neg_ll = tf.reduce_mean(tf.reduce_sum(neg_ll, axis=-1))
            self.training_model.add_metric(neg_ll + d_kl, name="elbo", aggregation="mean")
