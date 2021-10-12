import tensorflow as tf


class SingleMaxLayer(tf.keras.layers.Layer):

    def call(self, inputs, **kwargs):
        x = inputs[0]
        y = tf.where(x > 0.5, x=tf.divide(x, x), y=tf.multiply(x, tf.constant(0.0, dtype=tf.float32)))
        return y


class SingleGcnLayer(tf.keras.layers.Layer):

    def __init__(self, input_dim, latent_dim, dropout_rate, activation, l2_reg, use_bias: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.activation = tf.keras.activations.get(activation)
        self.l2_reg = l2_reg
        self.use_bias = use_bias

        self.kernel = None
        self.bias = None
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.input_dim, self.latent_dim),
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
        )
        # Layer bias
        if self.use_bias:
            self.bias = self.add_weight(name="bias", shape=(self.output_dim,))

    def call(self, inputs, **kwargs):
        targets = inputs[0]  # (batch, target nodes, features)
        neighbors = inputs[1]  # (batch, target nodes, padded neighbor nodes, features)
        neighborhood = tf.concat([tf.expand_dims(targets, axis=-2), neighbors], axis=-2)
        a = inputs[2]  # (batch, target nodes, padded neighbor nodes)

        # (batch, target nodes, padded neighbor nodes, latent)
        y = tf.matmul(neighborhood, self.kernel, axes=1)
        y = tf.reduce_sum(y, axis=-2)  # (batch, target nodes, latent)
        # Normalise neighborhood size in embedding:
        factor = tf.reduce_sum(a, axis=-1, keepdims=True)  # (batch, target nodes, 1)
        factor = tf.where(factor > 0.5, x=factor, y=tf.ones_like(factor))
        y = y / factor
        y = tf.keras.layers.Dropout(self.dropout_rate)(y)

        if self.use_bias:
            y = tf.add(y, self.bias)
        if self.activation is not None:
            y = self.activation(y)

        return y


class SingleLrGatLayer(tf.keras.layers.Layer):

    def __init__(self, lr_dim, out_dim, dropout_rate, l2_reg, **kwargs):
        """Initialize GCNLayer.

        Parameters
        ----------
        out_dim
            Output dimension.
        dropout_rate
            Dropout rate.
        activation
            Activation.
        l2_reg
            l2 regularization coefficient.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.lr_dim = lr_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        # Layer kernel
        self.kernel_l = self.add_weight(
            name="kernel_l",
            shape=(1, 1, self.lr_dim),
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
        )
        self.bias_l = self.add_weight(name="bias_l", shape=(1, 1, self.lr_dim,))
        self.kernel_r = self.add_weight(
            name="kernel_r",
            shape=(1, 1, self.lr_dim),
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
        )
        self.bias_r = self.add_weight(name="bias_r", shape=(1, 1, 1, self.lr_dim,))

    def call(self, inputs, **kwargs):
        targets_receptor = inputs[0]  # (batch, target nodes, lr)
        neighbors_ligand = inputs[1]  # (batch, target nodes, padded neighbor nodes, lr)
        a = inputs[2]  # (batch, target nodes, padded neighbor nodes)

        targets_receptor = targets_receptor * self.kernel_r + self.bias_r  # (batch, target nodes, lr)
        neighbors_ligand = neighbors_ligand * self.kernel_l + self.bias_l  # (batch, target nodes, padded neighbor nodes, lr)
        targets_receptor = tf.expand_dims(targets_receptor, axis=-2)  # (batch, target nodes, 1, lr)
        weights = targets_receptor * neighbors_ligand
        # Mask embeddings to neighbors
        a = tf.expand_dims(a, axis=-1)  # (batch, target nodes, padded neighbor nodes, 1)
        weights = weights * a
        weights = tf.reduce_sum(weights, axis=2)  # (batch, target nodes, lr)
        y = tf.math.sigmoid(weights)

        return y


class SingleGatLayer(tf.keras.layers.Layer):

    def __init__(self, in_dim, out_dim, dropout_rate, l2_reg, **kwargs):
        """Initialize GatLayer.

        Parameters
        ----------
        out_dim
            Output dimension.
        dropout_rate
            Dropout rate.
        activation
            Activation.
        l2_reg
            l2 regularization coefficient.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        # Layer kernel
        self.kernel_target = self.add_weight(
            name="kernel_l",
            shape=(1, 1, self.in_dim),
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
        )
        self.bias_target = self.add_weight(name="bias_l", shape=(1, 1, self.in_dim,))
        self.kernel_neighbor = self.add_weight(
            name="kernel_r",
            shape=(1, 1, self.in_dim),
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
        )
        self.bias_neighbor = self.add_weight(name="bias_r", shape=(1, 1, 1, self.in_dim,))

    def call(self, inputs, **kwargs):
        targets = inputs[0]  # (batch, target nodes, features)
        neighbors = inputs[1]  # (batch, target nodes, padded neighbor nodes, features)
        a = inputs[2]  # (batch, target nodes, padded neighbor nodes)

        targets = targets * self.kernel_neighbor + self.bias_neighbor  # (batch, target nodes, lr)
        neighbors = neighbors * self.kernel_target + self.bias_target  # (batch, target nodes, padded neighbor nodes, lr)
        targets = tf.expand_dims(targets, axis=-2)  # (batch, target nodes, 1, lr)
        weights = targets * neighbors
        # Mask embeddings to neighbors
        a = tf.expand_dims(a, axis=-1)  # (batch, target nodes, padded neighbor nodes, 1)
        weights = weights * a
        weights = tf.reduce_sum(weights, axis=2)  # (batch, target nodes, lr)
        y = tf.math.sigmoid(weights)

        return y
