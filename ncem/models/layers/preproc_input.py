import tensorflow as tf


class PreprocInput(tf.keras.layers.Layer):
    def __init__(self, magnitude: float = 100.0, name="preproc_input", **kwargs):
        """
        Transform last dimension.

        :param magnitude:
        :param name:
        :param kwargs:
        """
        super().__init__(name=name, **kwargs)
        self.magnitude = magnitude

    def call(self, inputs, **kwargs):
        inputs = inputs / tf.reduce_sum(inputs, axis=-1, keepdims=True) * self.magnitude
        inputs = tf.math.log(inputs + 1)
        return inputs


class NodeDegrees(tf.keras.layers.Layer):
    def __init__(self, in_node_dim, name="nodedegrees", **kwargs):
        super().__init__(name=name, **kwargs)
        self.in_node_dim = in_node_dim

    def call(self, inputs, **kwargs):
        node_degrees = tf.sparse.reduce_sum(inputs, axis=-1)
        node_degrees = tf.reshape(node_degrees, shape=[-1, self.in_node_dim, 1])
        return node_degrees


class DenseInteractions(tf.keras.layers.Layer):
    def __init__(self, in_node_dim, interaction_dim, name="denseinteraction", **kwargs):
        super().__init__(name=name, **kwargs)
        self.in_node_dim = in_node_dim
        self.interaction_dim = interaction_dim

    def call(self, inputs, **kwargs):
        interaction = tf.sparse.to_dense(inputs)
        interaction = tf.reshape(interaction, shape=[-1, self.in_node_dim, self.interaction_dim])
        return interaction
