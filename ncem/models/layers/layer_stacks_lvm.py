import tensorflow as tf


class SamplingPrior(tf.keras.layers.Layer):
    """Uses input tensor to sample from bottleneck of defined size."""

    def __init__(self, width, name: str = "sampling_layer", **kwargs):
        """Initialize SamplingPrior custom layer.

        Parameters
        ----------
        width
            Width.
        name : str
            Layer name.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.width = width

    def call(self, inputs, **kwargs):
        """Call SamplingPrior layer.

        Parameters
        ----------
        inputs
            Inputs.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        epsilon
            epsilon.
        """
        x = inputs
        batch = tf.shape(x)[0]  # batch_size
        node_dim = tf.shape(x)[1]  # number of nodes per graph
        epsilon = tf.random.normal(
            shape=(batch, node_dim, self.width), mean=0.0, stddev=1.0, dtype=x.dtype
        )  # shape: (batch size, nodes per graph, bottleneck size)
        return epsilon


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""

    def __init__(self, name: str = "sampling_layer", **kwargs):
        """Initialize SamplingPrior custom layer.

        Parameters
        ----------
        name : str
            Layer name.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        """Call Sampling layer.

        Reparameterization trick by sampling from an isotropic unit Gaussian.

        Parameters
        ----------
        inputs
            mean and log of variance of Q(z|X).
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        z
            sampled latent vector.
        """
        z_mean, z_log_var = inputs
        # z_mean - shape: (batch_size, latent_dim), z_log_var - shape: (batch_size, latent_dim)
        batch = tf.shape(z_mean)[0]  # batch_size
        dim = tf.shape(z_mean)[1]  # latent_dim
        # by default, random_normal has mean = 0 and std = 1
        epsilon = tf.random.normal(
            shape=(batch, dim), mean=0.0, stddev=1.0, dtype=z_mean.dtype
        )  # shape: (batch_size, latent_dim)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon  # shape: (batch_size, latent_dim)


class Encoder(tf.keras.layers.Layer):
    """Maps input to embedding space."""

    def __init__(
        self,
        latent_dim,
        intermediate_dim,
        dropout_rate,
        n_hidden,
        l1_coef: float,
        l2_coef: float,
        use_type_cond: bool = True,
        use_batch_norm: bool = False,
        probabilistic: bool = True,
        name="encoder",
        **kwargs
    ):
        """Initialize Encoder custom layer.

        Parameters
        ----------
        latent_dim : int
            Latent dimension.
        intermediate_dim : int
            Intermediate dimension.
        dropout_rate : float
            Dropout rate.
        n_hidden : int
            Number of hidden layers.
        l1_coef : float
            l1 regularization coefficient.
        l2_coef : float
            l2 regularization coefficient.
        use_type_cond : bool
            Whether to use type conditional.
        use_batch_norm : bool
            Whether to use batch normalization.
        probabilistic : bool
            Whether sampling is done or not.
        name : str
            Layer name.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout_rate
        self.n_hidden = n_hidden
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.use_type_cond = use_type_cond
        self.use_batch_norm = use_batch_norm
        self.probabilistic = probabilistic

        self.kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.l1_coef, l2=self.l2_coef)

        self.fwd_pass = None
        self.dense_mean = None
        self.dense_log_var = None
        self.sampling = None

    def get_config(self):
        """Get config Encoder layer.

        Returns
        -------
        config
        """
        config = super().get_config().copy()
        config.update(
            {
                "encoder_latent_dim": self.latent_dim,
                "encoder_intermediate_dim": self.intermediate_dim,
                "encoder_dropout_rate": self.dropout_rate,
                "encoder_n_hidden": self.n_hidden,
                "encoder_l1_coef": self.l1_coef,
                "encoder_l2_coef": self.l2_coef,
                "encoder_use_type_cond": self.use_type_cond,
            }
        )
        return config

    def build(self, input_shapes):
        """Build Encoder layer.

        Parameters
        ----------
        input_shapes
            Input shapes.
        """
        self.fwd_pass = []
        for i in range(0, self.n_hidden):
            self.fwd_pass.append(
                tf.keras.layers.Dense(
                    units=self.intermediate_dim,
                    activation="tanh",
                    kernel_regularizer=self.kernel_regularizer,
                    name="encoder_layer_" + str(i + 1),
                )
            )
            if i < self.n_hidden - 1 and self.use_batch_norm:
                self.fwd_pass.append(tf.keras.layers.BatchNormalization(center=True, scale=False))
            if self.dropout_rate is not None:
                self.fwd_pass.append(tf.keras.layers.Dropout(self.dropout_rate, noise_shape=None, seed=None))

        # final layer
        self.dense_mean = tf.keras.layers.Dense(
            units=self.latent_dim, activation="linear"
        )  # last dense layer if not if self.probabilistic
        if self.probabilistic:
            self.dense_log_var = tf.keras.layers.Dense(units=self.latent_dim, activation="linear")
            self.sampling = Sampling()

    def call(self, inputs, **kwargs):
        """Call Encoder layer.

        Parameters
        ----------
        inputs
            Inputs.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        z
        z_mean
        z_var_log
        """
        h, c = inputs
        # h - shape: (batch_size, N, F), c - shape: (batch_size, N, P)
        if self.use_type_cond:
            x = tf.concat([h, c], axis=-1, name="encoder_concat_fwdpass")  # shape: (batch_size, N, F+P)
        else:
            x = h  # shape: (batch_size, N, F)
        x = tf.reshape(
            x, [-1, x.shape[-1]], name="encoder_reshape_x_fwpass"
        )  # shape: (batch_size*N, F) OR (batch_size*N, F+P)
        for layer in self.fwd_pass:
            x = layer(x, **kwargs)

        # final layer
        z_mean = self.dense_mean(x)
        if self.probabilistic:
            z_log_var = self.dense_log_var(x, **kwargs)
            z = self.sampling((z_mean, z_log_var), **kwargs)
            return z, z_mean, z_log_var
        else:
            return z_mean, z_mean, z_mean  # stick to output format from probabilistic model


class Decoder(tf.keras.layers.Layer):
    """Maps latent space sample back to output."""

    def __init__(
        self,
        intermediate_dim: int,
        dropout_rate: float,
        n_hidden: int,
        l1_coef: float,
        l2_coef: float,
        use_type_cond: bool = True,
        use_batch_norm: bool = False,
        name: str = "decoder",
        **kwargs
    ):
        """Initialize Decoder custom layer.

        Parameters
        ----------
        intermediate_dim : int
            Intermediate dimension.
        dropout_rate : float
            Dropout rate.
        n_hidden : int
            Number of hidden layers.
        l1_coef : float
            l1 regularization coefficient.
        l2_coef : float
            l2 regularization coefficient.
        use_type_cond : bool
            Whether to use type conditional.
        use_batch_norm : bool
            Whether to use batch normalization.
        name : str
            Layer name.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(name=name, **kwargs)

        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout_rate
        self.n_hidden = n_hidden
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.use_type_cond = use_type_cond
        self.use_batch_norm = use_batch_norm

        self.kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.l1_coef, l2=self.l2_coef)
        self.fwd_pass = None

    def get_config(self):
        """Get config Decoder layer.

        Returns
        -------
        config
        """
        config = super().get_config().copy()
        config.update(
            {
                "decoder_intermediate_dim": self.intermediate_dim,
                "decoder_dropout_rate": self.dropout_rate,
                "decoder_n_hidden": self.n_hidden,
                "decoder_l1_coef": self.l1_coef,
                "decoder_l2_coef": self.l2_coef,
                "decoder_use_type_cond": self.use_type_cond,
                "decoder_use_batch_norm": self.use_batch_norm,
            }
        )
        return config

    def build(self, input_shapes):
        """Build Decoder layer.

        Parameters
        ----------
        input_shapes
            Input shapes.
        """
        self.fwd_pass = []
        for i in range(0, self.n_hidden):
            self.fwd_pass.append(
                tf.keras.layers.Dense(
                    units=self.intermediate_dim,
                    activation="tanh",
                    kernel_regularizer=self.kernel_regularizer,
                    name="decoder_Layer_" + str(i + 1),
                )
            )
            if self.use_batch_norm:
                self.fwd_pass.append(tf.keras.layers.BatchNormalization(center=True, scale=False))
            if self.dropout_rate is not None:
                self.fwd_pass.append(tf.keras.layers.Dropout(self.dropout_rate, noise_shape=None, seed=None))

    def call(self, inputs, **kwargs):
        """Call Decoder layer.

        Parameters
        ----------
        inputs
            Inputs.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        x
            output of Decoder layer
        """
        z, c = inputs
        in_node_dim = c.shape[1]
        # z - shape: (batch_size*N, latent_dim), c - shape: (batch_size, N, P)
        if self.use_type_cond:
            c = tf.reshape(c, [-1, c.shape[-1]])  # shape: (batch_size*N, P)
            x = tf.concat([z, c], axis=-1, name="decoder_concat_fwdpass")  # shape: (batch_size*N, latent_dim+P)
        else:
            x = z  # shape: (batch_size*N, latent_dim)

        if len(self.fwd_pass) > 0:
            for layer in self.fwd_pass:
                x = layer(x, **kwargs)
            x = tf.reshape(x, [-1, in_node_dim, self.intermediate_dim], name="decoder_reshape_x")
        else:  # empty decoder
            x = tf.reshape(x, [-1, in_node_dim, x.shape[1]], name="decoder_reshape_x")
        return x


class CondEncoder(tf.keras.layers.Layer):
    """Maps input to embedding space."""

    def __init__(
        self,
        latent_dim: int,
        intermediate_dim: int,
        dropout_rate: float,
        n_hidden: int,
        l1_coef: float,
        l2_coef: float,
        use_graph_conditional: bool = True,
        use_type_cond: bool = True,
        use_batch_norm: bool = False,
        probabilistic: bool = True,
        name: str = "cond_encoder",
        **kwargs
    ):
        """Initialize CondEncoder custom layer.

        Parameters
        ----------
        latent_dim : int
            Latent dimension.
        intermediate_dim : int
            Intermediate dimension.
        dropout_rate : float
            Dropout rate.
        n_hidden : int
            Number of hidden layers.
        l1_coef : float
            l1 regularization coefficient.
        l2_coef : float
            l2 regularization coefficient.
        use_graph_conditional : bool
            Whether to use graph conditional.
        use_type_cond : bool
            Whether to use type conditional.
        use_batch_norm : bool
            Whether to use batch normalization.
        probabilistic : bool
            Whether sampling is done or not.
        name : str
            Layer name.
        kwargs
            Arbitrary keyword arguments.
        """
        super(CondEncoder, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout_rate
        self.n_hidden = n_hidden
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.use_graph_conditional = use_graph_conditional
        self.use_type_cond = use_type_cond
        self.use_batch_norm = use_batch_norm
        self.probabilistic = probabilistic

        self.fwd_pass = None
        self.dense_mean = None
        self.dense_log_var = None
        self.sampling = None

    def get_config(self):
        """Get config CondEncoder layer.

        Returns
        -------
        config
        """
        config = super().get_config().copy()
        config.update(
            {
                "encoder_latent_dim": self.latent_dim,
                "encoder_intermediate_dim": self.intermediate_dim,
                "encoder_dropout_rate": self.dropout_rate,
                "encoder_n_hidden": self.n_hidden,
                "encoder_l1_coef": self.l1_coef,
                "encoder_l2_coef": self.l2_coef,
                "encoder_use_graph_conditional": self.use_graph_conditional,
                "encoder_use_type_cond": self.use_type_cond,
                "encoder_use_batch_norm": self.use_batch_norm,
            }
        )
        return config

    def build(self, input_shapes):
        """Build CondEncoder layer.

        Parameters
        ----------
        input_shapes
            Input shapes.
        """
        self.fwd_pass = []
        for i in range(0, self.n_hidden):
            self.fwd_pass.append(
                tf.keras.layers.Dense(
                    units=self.intermediate_dim,
                    activation="tanh",
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_coef, l2=self.l2_coef),
                    name="encoder_layer_" + str(i + 1),
                )
            )
            if i < self.n_hidden - 1 and self.use_batch_norm:
                self.fwd_pass.append(tf.keras.layers.BatchNormalization(center=True, scale=False))
            if self.dropout_rate is not None:
                self.fwd_pass.append(tf.keras.layers.Dropout(self.dropout_rate, noise_shape=None, seed=None))

        # final layer
        self.dense_mean = tf.keras.layers.Dense(
            units=self.latent_dim, activation="linear"
        )  # last dense layer if not if self.probabilistic
        if self.probabilistic:
            self.dense_log_var = tf.keras.layers.Dense(units=self.latent_dim, activation="linear")
            self.sampling = Sampling()

    # Put encoder network in the context of input data and graph data in the latent layer.
    def call(self, inputs, **kwargs):
        """Call CondEncoder layer.

        Parameters
        ----------
        inputs
            Inputs.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        z
        z_mean
        z_log_var
        """
        h_reconstruct, neighbour_embedding, c = inputs
        # h_reconstruct - shape: (batch_size, N, F), h_cond - shape: (batch_size, N, F)
        # a - shape: (batch_size, N, N), c - shape: (batch_size, N, P)

        if self.use_graph_conditional:
            c = tf.concat(
                [neighbour_embedding, c], axis=2, name="encoder_concat_c"
            )  # shape: (batch_size, N, cond_dim+P)
            # with cond_dim being the output dim of the conditional layer stack

        c = tf.reshape(c, [-1, c.shape[-1]])  # shape: (batch_size*N, cond_dim+P) OR (batch_size*N, cond_dim)
        h_reconstruct = tf.reshape(h_reconstruct, [-1, h_reconstruct.shape[-1]])  # shape: (batch_size*N, F)

        if self.use_type_cond:
            x = tf.concat([h_reconstruct, c], axis=-1, name="encoder_concat_c_h")  # shape: (batch_size*N, F+cond_dim+P)
        else:
            x = h_reconstruct

        for layer in self.fwd_pass:
            x = layer(x, **kwargs)

        # final layer
        z_mean = self.dense_mean(x)  # shape: (batch_size*N, latent_dim)
        if self.probabilistic:
            z_log_var = self.dense_log_var(x, **kwargs)  # shape: (batch_size*N, latent_dim)
            z = self.sampling((z_mean, z_log_var), **kwargs)  # shape: (batch_size*N, latent_dim)
            return z, z_mean, z_log_var
        else:
            return z_mean, z_mean, z_mean  # stick to output format from probabilistic model


class CondDecoder(tf.keras.layers.Layer):
    """Maps latent space sample back to output."""

    def __init__(
        self,
        intermediate_dim: int,
        dropout_rate: float,
        n_hidden: int,
        l1_coef: float,
        l2_coef: float,
        use_graph_conditional: bool = True,
        use_type_cond: bool = True,
        use_batch_norm: bool = False,
        name: str = "cond_decoder",
        **kwargs
    ):
        """Initialize Decoder custom layer.

        Parameters
        ----------
        intermediate_dim : int
            Intermediate dimension.
        dropout_rate : float
            Dropout rate.
        n_hidden : int
            Number of hidden layers.
        l1_coef : float
            l1 regularization coefficient.
        l2_coef : float
            l2 regularization coefficient.
        use_graph_conditional : bool
            Whether to use graph conditional.
        use_type_cond : bool
            Whether to use type conditional.
        use_batch_norm : bool
            Whether to use batch normalization.
        name : str
            Layer name.
        kwargs
            Arbitrary keyword arguments.
        """
        super(CondDecoder, self).__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout_rate
        self.n_hidden = n_hidden
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.use_graph_conditional = use_graph_conditional
        self.use_type_cond = use_type_cond
        self.use_batch_norm = use_batch_norm

        self.fwd_pass = None

    def get_config(self):
        """Get config of CondDecoder.

        Returns
        -------
        config
        """
        config = super().get_config().copy()
        config.update(
            {
                "decoder_intermediate_dim": self.intermediate_dim,
                "decoder_dropout_rate": self.dropout_rate,
                "decoder_n_hidden": self.n_hidden,
                "decoder_l1_coef": self.l1_coef,
                "decoder_l2_coef": self.l2_coef,
                "decoder_use_graph_conditional": self.use_graph_conditional,
                "decoder_use_type_cond": self.use_type_cond,
                "decoder_use_batch_norm": self.use_batch_norm,
            }
        )
        return config

    def build(self, input_shapes):
        """Build CondDecoder layer.

        Parameters
        ----------
        input_shapes
            Input shapes.
        """
        self.fwd_pass = []
        for i in range(0, self.n_hidden):
            self.fwd_pass.append(
                tf.keras.layers.Dense(
                    units=self.intermediate_dim,
                    activation="tanh",
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_coef, l2=self.l2_coef),
                    name="decoder_layer_" + str(i + 1),
                )
            )
            if self.use_batch_norm:
                self.fwd_pass.append(tf.keras.layers.BatchNormalization(center=True, scale=False))
            if self.dropout_rate is not None:
                self.fwd_pass.append(tf.keras.layers.Dropout(self.dropout_rate, noise_shape=None, seed=None))

    def call(self, inputs, **kwargs):
        """Call CondDecoder layer.

        Parameters
        ----------
        inputs
            Inputs.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        x
            output of CondDecoder layer
        """
        z, neighbour_embedding, c = inputs
        in_node_dim = c.shape[1]
        # z - shape: (batch_size*N, latent_dim),
        # neighbour_embedding - shape: (batch_size, N, F)
        # c - shape: (batch_size, N, P)

        if self.use_graph_conditional:
            c = tf.concat(
                [neighbour_embedding, c], axis=2, name="encoder_concat_c"
            )  # shape: (batch_size, N, cond_dim+P)
            # with cond_dim being the output dim of the conditional layer stack

        if self.use_type_cond:
            c = tf.reshape(c, [-1, c.shape[-1]])  # shape: (batch_size*N, cond_dim+P) OR (batch_size*N, cond_dim)
            x = tf.concat(
                [z, c], axis=-1, name="decoder_concat_fwdpass"
            )  # shape: (batch_size*N, latent_dim+cond_dim+P)
        else:
            x = z

        if len(self.fwd_pass) > 0:
            for layer in self.fwd_pass:
                x = layer(x, **kwargs)
            x = tf.reshape(x, [-1, in_node_dim, self.intermediate_dim], name="decoder_reshape_x")
            # shape: (batch_size, N, latent_dim+cond_dim+P)
        else:  # empty decoder
            x = tf.reshape(x, [-1, in_node_dim, x.shape[1]], name="decoder_reshape_x")
        return x
