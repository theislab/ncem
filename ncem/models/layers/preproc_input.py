import tensorflow as tf


class PreprocInput(tf.keras.layers.Layer):
    """PreprocInput layer."""

    def __init__(self, magnitude: float = 100.0, name: str = "preproc_input", **kwargs):
        """Transform last dimension.

        Parameters
        ----------
        magnitude : float
            magnitude.
        name : str
            Layer name.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.magnitude = magnitude

    def call(self, inputs, **kwargs):
        """Call PreprocInput layer.

        Parameters
        ----------
        inputs
            Inputs.git
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        inputs
        """
        inputs = inputs / tf.reduce_sum(inputs, axis=-1, keepdims=True) * self.magnitude
        inputs = tf.math.log(inputs + 1)
        return inputs


class DenseInteractions(tf.keras.layers.Layer):
    """DenseInteractions layer."""

    def __init__(self, in_node_dim, interaction_dim, name: str = "denseinteraction", **kwargs):
        """Initialize DenseInteractions.

        Parameters
        ----------
        in_node_dim
            Node dimension.
        interaction_dim
            Interactions dimension.
        name : str
            Layer name.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.in_node_dim = in_node_dim
        self.interaction_dim = interaction_dim

    def call(self, inputs, **kwargs):
        """Call DenseInteractions layer.

        Parameters
        ----------
        inputs
            Inputs.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        interactions
        """
        interaction = tf.sparse.to_dense(inputs)
        interaction = tf.reshape(interaction, shape=[-1, self.in_node_dim, self.interaction_dim])
        return interaction
