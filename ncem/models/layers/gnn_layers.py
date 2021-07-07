import tensorflow as tf

from ncem.utils.sparse import sparse_dense_matmult_batch


class MaxLayer(tf.keras.layers.Layer):
    """Initialize MaxLayer."""

    def __init__(self, **kwargs):
        """Initialize MaxLayer.

        Parameters
        ----------
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """Call MAX/IND layer.

        Parameters
        ----------
        inputs
            Inputs.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        output
            output of MAX/IND layer
        """
        x = inputs[0]
        a = inputs[1]

        if isinstance(a, tf.SparseTensor):
            t = sparse_dense_matmult_batch(a, x)
        else:
            t = tf.matmul(a, x)

        output = tf.where(t > 0.5, x=tf.divide(t, t), y=tf.multiply(t, tf.constant(0.0, dtype=tf.float32)))
        return output


class GCNLayer(tf.keras.layers.Layer):
    """Initialize GCNLayer."""

    def __init__(self, output_dim, dropout_rate, activation, l2_reg, use_bias: bool = False, **kwargs):
        """Initialize GCNLayer.

        Parameters
        ----------
        output_dim
            Output dimension.
        dropout_rate
            Dropout rate.
        activation
            Activation.
        l2_reg
            l2 regularization coefficient.
        use_bias : bool
            Use bias.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation = tf.keras.activations.get(activation)
        self.l2_reg = l2_reg
        self.use_bias = use_bias

        self.kernel = None
        self.bias = None

    def get_config(self):
        """Get config GCN layer.

        Returns
        -------
        config
        """
        config = super().get_config().copy()
        config.update(
            {
                "output_dim": self.output_dim,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "l2_reg": self.l2_reg,
                "use_bias": self.use_bias,
            }
        )
        return config

    def build(self, input_shapes):
        """Build GCN layer.

        Parameters
        ----------
        input_shapes
            Input shapes.
        """
        input_shape = input_shapes[0]
        # Layer kernel
        self.kernel = self.add_weight(
            name="kernel",
            shape=(int(input_shape[2]), self.output_dim),
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
        )
        # Layer bias
        if self.use_bias:
            self.bias = self.add_weight(name="bias", shape=(self.output_dim,))

    def call(self, inputs, **kwargs):
        """GCN layer call function.

        Parameters
        ----------
        inputs
            Inputs.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        output
            output of GCN layer
        """
        x = inputs[0]
        a = inputs[1]

        if isinstance(a, tf.SparseTensor):
            t = sparse_dense_matmult_batch(a, x)
        else:
            t = tf.matmul(a, x)
        output = tf.tensordot(t, self.kernel, axes=1)
        output = tf.keras.layers.Dropout(self.dropout_rate)(output)

        if self.use_bias:
            output = tf.add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output
