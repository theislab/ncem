import tensorflow as tf

IDENTIFIER_OUTPUT_LAYER = "Output"


def get_out(output_layer: str, out_feature_dim, scale_node_size, name: str = 'decoder'):
    if output_layer == "gaussian":
        output_decoder_layer = GaussianOutput(
            original_dim=out_feature_dim,
            use_node_scale=scale_node_size,
            name=f"Gaussian{IDENTIFIER_OUTPUT_LAYER}_{name}",
        )
    elif output_layer == "nb":
        output_decoder_layer = NegBinOutput(
            original_dim=out_feature_dim,
            use_node_scale=scale_node_size,
            name=f"NegBin{IDENTIFIER_OUTPUT_LAYER}_{name}",
        )
    elif output_layer == "nb_shared_disp":
        output_decoder_layer = NegBinSharedDispOutput(
            original_dim=out_feature_dim,
            use_node_scale=scale_node_size,
            name=f"NegBinSharedDisp{IDENTIFIER_OUTPUT_LAYER}_{name}",
        )
    elif output_layer == "nb_const_disp":
        output_decoder_layer = NegBinConstDispOutput(
            original_dim=out_feature_dim,
            use_node_scale=scale_node_size,
            name=f"NegBinConstDisp{IDENTIFIER_OUTPUT_LAYER}_{name}",
        )
    else:
        raise ValueError("tried to access a non-supported output layer %s" % output_layer)
    return output_decoder_layer


class LinearOutput(tf.keras.layers.Layer):
    """Linear output layer."""

    def __init__(self, use_node_scale: bool = False, name: str = "linear_output", **kwargs):
        """Initialize LinearOutput.

        Parameters
        ----------
        use_node_scale : bool
            Use node scale.
        name : str
            Layer name.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.use_node_scale = use_node_scale
        self.var_bias = None

    def get_config(self):
        """Get config LinearOutput.

        Returns
        -------
        config
        """
        config = super().get_config().copy()
        config.update({"original_dim": self.original_dim, "use_node_scale": self.use_node_scale})
        return config

    def build(self, input_shapes):
        """Build LinearOutput layer.

        Parameters
        ----------
        input_shapes
            Input shapes.
        """
        genes_dim = input_shapes[0][-1]
        self.var_bias = self.add_weight("var_bias", shape=[1, genes_dim], initializer="zeros")

    def call(self, inputs, **kwargs):
        """Call LinearOutput layer.

        Parameters
        ----------
        inputs
            Inputs.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        eta_loc
        eta_scale
        """
        bound = 60.0
        mean, sf = inputs

        var = self.var_bias

        if self.use_node_scale:
            mean = mean * tf.clip_by_value(sf, tf.exp(-bound), tf.exp(bound), "decoder_sf_clip")
        var = tf.zeros_like(mean) + var  # broadcast

        # clip to log of largest values supported by log operation
        mean_clip = tf.clip_by_value(mean, -tf.exp(bound), tf.exp(bound), "decoder_clip")
        var_clip = tf.clip_by_value(var, -bound, bound, "decoder_clip")

        # exp_mean = mean_clip + sf
        eta_loc = mean_clip
        eta_scale = tf.exp(var_clip)

        return [eta_loc, eta_scale]


class LinearConstDispOutput(tf.keras.layers.Layer):
    """Linear output layer with constant dispersion."""

    def __init__(self, use_node_scale: bool = False, name: str = "linear_const_disp_output", **kwargs):
        """Initialize LinearConstDispOutput.

        Parameters
        ----------
        use_node_scale : bool
            Use node scale.
        name : str
            Layer name.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.use_node_scale = use_node_scale
        self.var_bias = None

    def get_config(self):
        """Get config LinearConstDispOutput.

        Returns
        -------
        config
        """
        config = super().get_config().copy()
        config.update({"original_dim": self.original_dim, "use_node_scale": self.use_node_scale})
        return config

    def call(self, inputs, **kwargs):
        """Call LinearConstDispOutput layer.

        Parameters
        ----------
        inputs
            Inputs.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        eta_loc
        eta_scale
        """
        bound = 60.0
        mean, sf = inputs

        var = tf.zeros_like(mean)

        if self.use_node_scale:
            mean = mean * tf.clip_by_value(sf, tf.exp(-bound), tf.exp(bound), "decoder_sf_clip")
        var = tf.zeros_like(mean) + var

        # clip to log of largest values supported by log operation
        mean_clip = tf.clip_by_value(mean, -tf.exp(bound), tf.exp(bound), "decoder_clip")
        var_clip = tf.clip_by_value(var, -bound, bound, "decoder_clip")

        # exp_mean = mean_clip + sf
        eta_loc = mean_clip
        eta_scale = tf.exp(var_clip)

        return [eta_loc, eta_scale]


class GaussianOutput(tf.keras.layers.Layer):
    """Log normal likelihood output layer."""

    def __init__(self, original_dim=None, use_node_scale: bool = False, name: str = "gaussian_output", **kwargs):
        """Initialize GaussianOutput.

        Parameters
        ----------
        original_dim
            original dimension.
        use_node_scale : bool
            Use node scale.
        name : str
            Layer name.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(name=name, **kwargs)

        self.original_dim = original_dim
        self.intermediate_dim = None
        self.use_node_scale = use_node_scale
        self.means = None
        self.var_bias = None

    def get_config(self):
        """Get config GaussianOutput.

        Returns
        -------
        config
        """
        config = super().get_config().copy()
        config.update({"original_dim": self.original_dim, "use_node_scale": self.use_node_scale})
        return config

    def build(self, input_shapes):
        """Build GaussianOutput layer.

        Parameters
        ----------
        input_shapes
            Input shapes.
        """
        input_shape = input_shapes[0]
        self.intermediate_dim = input_shape[2]

        self.means = tf.keras.layers.Dense(units=self.original_dim, use_bias=True, activation="linear")
        self.var_bias = self.add_weight("var_bias", shape=[1, self.original_dim], initializer="zeros")

    def call(self, inputs, **kwargs):
        """Call GaussianOutput layer.

        Parameters
        ----------
        inputs
            Inputs.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        eta_loc
        eta_scale
        """
        bound = 60.0

        activation, sf = inputs
        in_node_dim = activation.shape[1]

        activation = tf.reshape(activation, [-1, self.intermediate_dim], name="output_layer_reshape_activation_fwdpass")

        mean = self.means(activation)
        var = self.var_bias

        mean = tf.reshape(mean, [-1, in_node_dim, self.original_dim], name="output_layer_reshape_mean")
        var = tf.zeros_like(mean) + var  # broadcast
        if self.use_node_scale:
            mean = mean * tf.clip_by_value(sf, tf.exp(-bound), tf.exp(bound), "decoder_sf_clip")

        # clip to log of largest values supported by log operation
        mean_clip = tf.clip_by_value(mean, -tf.exp(bound), tf.exp(bound), "decoder_clip")
        var_clip = tf.clip_by_value(var, -bound, bound, "decoder_clip")

        # exp_mean = mean_clip + sf
        eta_loc = mean_clip
        eta_scale = tf.exp(var_clip)

        return [eta_loc, eta_scale]


class GaussianConstDispOutput(tf.keras.layers.Layer):
    """Log normal likelihood output layer."""

    def __init__(self, original_dim=None, use_node_scale: bool = False, name: str = "gaussian_output", **kwargs):
        """Initialize GaussianConstDispOutput.

        Parameters
        ----------
        original_dim
            original dimension.
        use_node_scale : bool
            Use node scale.
        name : str
            Layer name.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(name=name, **kwargs)

        self.original_dim = original_dim
        self.intermediate_dim = None
        self.use_node_scale = use_node_scale
        self.means = None

    def get_config(self):
        """Get config GaussianConstDispOutput.

        Returns
        -------
        config
        """
        config = super().get_config().copy()
        config.update({"original_dim": self.original_dim, "use_node_scale": self.use_node_scale})
        return config

    def build(self, input_shapes):
        """Build GaussianConstDispOutput layer.

        Parameters
        ----------
        input_shapes
            Input shapes.
        """
        input_shape = input_shapes[0]
        self.intermediate_dim = input_shape[2]

        self.means = tf.keras.layers.Dense(units=self.original_dim, use_bias=True, activation="linear")

    def call(self, inputs, **kwargs):
        """Call GaussianConstDispOutput layer.

        Parameters
        ----------
        inputs
            Inputs.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        eta_loc
        eta_scale
        """
        bound = 60.0

        activation, sf = inputs
        in_node_dim = activation.shape[1]
        activation = tf.reshape(activation, [-1, self.intermediate_dim], name="output_layer_reshape_activation_fwdpass")

        mean = self.means(activation)
        mean = tf.reshape(mean, [-1, in_node_dim, self.original_dim], name="output_layer_reshape_mean")
        var = tf.zeros_like(mean)
        if self.use_node_scale:
            mean = mean * tf.clip_by_value(sf, tf.exp(-bound), tf.exp(bound), "decoder_sf_clip")

        # clip to log of largest values supported by log operation
        mean_clip = tf.clip_by_value(mean, -tf.exp(bound), tf.exp(bound), "decoder_clip")
        var_clip = tf.clip_by_value(var, -bound, bound, "decoder_clip")

        # exp_mean = mean_clip + sf
        eta_loc = mean_clip
        eta_scale = tf.exp(var_clip)

        return [eta_loc, eta_scale]


class NegBinOutput(tf.keras.layers.Layer):
    """Negative binomial output layer."""

    def __init__(self, original_dim=None, use_node_scale: bool = False, name: str = "neg_bin_output", **kwargs):
        """Initialize NegBinOutput.

        Parameters
        ----------
        original_dim
            original dimension.
        use_node_scale : bool
            Use node scale.
        name : str
            Layer name.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(name=name, **kwargs)

        self.original_dim = original_dim
        self.intermediate_dim = None
        self.use_node_scale = use_node_scale
        self.means = None
        self.var = None

    def get_config(self):
        """Get config NegBinOutput.

        Returns
        -------
        config
        """
        config = super().get_config().copy()
        config.update({"original_dim": self.original_dim, "use_node_scale": self.use_node_scale})
        return config

    def build(self, input_shapes):
        """Build NegBinOutput layer.

        Parameters
        ----------
        input_shapes
            Input shapes.
        """
        input_shape = input_shapes[0]
        self.intermediate_dim = input_shape[2]

        self.means = tf.keras.layers.Dense(units=self.original_dim, use_bias=True, activation="linear")
        self.var = tf.keras.layers.Dense(units=self.original_dim, use_bias=True, activation="linear")

    def call(self, inputs, **kwargs):
        """Call NegBinOutput.

        Parameters
        ----------
        inputs
            Inputs.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        exp_mean
        exp_var
        """
        bound = 60.0

        activation, sf = inputs
        activation = tf.reshape(activation, [-1, self.intermediate_dim], name="output_layer_reshape_activation_fwdpass")

        mean = self.means(activation)
        var = self.var(activation)
        if self.use_node_scale:
            mean = mean + tf.math.log(tf.clip_by_value(sf, tf.exp(-bound), tf.exp(bound), "decoder_sf_clip"))

        # clip to log of largest values supported by log operation
        mean_clip = tf.clip_by_value(mean, -bound, bound, "decoder_clip")
        var_clip = tf.clip_by_value(var, -bound, bound, "decoder_clip")

        exp_mean = tf.exp(mean_clip)
        exp_var = tf.exp(var_clip)

        return [exp_mean, exp_var]


class NegBinSharedDispOutput(tf.keras.layers.Layer):
    """Negative binomial output layer with dispersion shared over features."""

    def __init__(
        self, original_dim=None, use_node_scale: bool = False, name: str = "neg_bin_shared_disp_output", **kwargs
    ):
        """Initialize NegBinSharedDispOutput.

        Parameters
        ----------
        original_dim
            original dimension.
        use_node_scale : bool
            Use node scale.
        name : str
            Layer name.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(name=name, **kwargs)

        self.original_dim = original_dim
        self.intermediate_dim = None
        self.use_node_scale = use_node_scale
        self.means = None
        self.var = None
        self.var_bias = None

    def get_config(self):
        """Get config NegBinSharedDispOutput.

        Returns
        -------
        config
        """
        config = super().get_config().copy()
        config.update({"original_dim": self.original_dim, "use_node_scale": self.use_node_scale})
        return config

    def build(self, input_shapes):
        """Build NegBinSharedDispOutput layer.

        Parameters
        ----------
        input_shapes
            Input shapes.
        """
        input_shape = input_shapes[0]
        self.intermediate_dim = input_shape[2]

        self.means = tf.keras.layers.Dense(units=self.original_dim, use_bias=True, activation="linear")
        self.var_bias = self.add_weight("var_bias", shape=[1, self.original_dim], initializer="zeros")

    def call(self, inputs, **kwargs):
        """Call NegBinSharedDispOutput layer.

        Parameters
        ----------
        inputs
            Inputs.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        exp_mean
        exp_var
        """
        bound = 60.0

        activation, sf = inputs
        in_node_dim = activation.shape[1]
        activation = tf.reshape(activation, [-1, self.intermediate_dim], name="output_layer_reshape_activation_fwdpass")

        mean = self.means(activation)
        var = self.var_bias

        mean = tf.reshape(mean, [-1, in_node_dim, self.original_dim], name="output_layer_reshape_mean")
        if self.use_node_scale:
            mean = mean + tf.math.log(tf.clip_by_value(sf, tf.exp(-bound), tf.exp(bound), "decoder_sf_clip"))
        var = tf.zeros_like(mean) + var  # broadcast

        # clip to log of largest values supported by log operation
        mean_clip = tf.clip_by_value(mean, -bound, bound, "decoder_clip")
        var_clip = tf.clip_by_value(var, -bound, bound, "decoder_clip")

        exp_mean = tf.exp(mean_clip)
        exp_var = tf.exp(var_clip)

        return [exp_mean, exp_var]


class NegBinConstDispOutput(tf.keras.layers.Layer):
    """Negative binomial output layer with constant dispersion."""

    def __init__(
        self, original_dim=None, use_node_scale: bool = False, name: str = "neg_bin_const_disp_output", **kwargs
    ):
        """Initialize NegBinConstDispOutput.

        Parameters
        ----------
        original_dim
            original dimension.
        use_node_scale : bool
            Use node scale.
        name : str
            Layer name.
        kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(name=name, **kwargs)

        self.original_dim = original_dim
        self.intermediate_dim = None
        self.use_node_scale = use_node_scale
        self.means = None
        self.var = None

    def get_config(self):
        """Get config NegBinConstDispOutput.

        Returns
        -------
        config
        """
        config = super().get_config().copy()
        config.update({"original_dim": self.original_dim, "use_node_scale": self.use_node_scale})
        return config

    def build(self, input_shapes):
        """Build NegBinConstDispOutput layer.

        Parameters
        ----------
        input_shapes
            Input shapes.
        """
        input_shape = input_shapes[0]
        self.intermediate_dim = input_shape[2]

        self.means = tf.keras.layers.Dense(units=self.original_dim, use_bias=True, activation="linear")

    def call(self, inputs, **kwargs):
        """Call NegBinConstDispOutput layer.

        Parameters
        ----------
        inputs
            Inputs.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        exp_mean
        exp_var
        """
        bound = 60.0

        activation, sf = inputs
        in_node_dim = activation.shape[1]
        activation = tf.reshape(activation, [-1, self.intermediate_dim], name="output_layer_reshape_activation_fwdpass")

        mean = self.means(activation)
        var = tf.zeros_like(mean)

        mean = tf.reshape(mean, [-1, in_node_dim, self.original_dim], name="output_layer_reshape_mean")
        if self.use_node_scale:
            mean = mean + tf.math.log(tf.clip_by_value(sf, tf.exp(-bound), tf.exp(bound), "decoder_sf_clip"))
        var = tf.zeros_like(mean) + var  # broadcast

        # clip to log of largest values supported by log operation
        mean_clip = tf.clip_by_value(mean, -bound, bound, "decoder_clip")
        var_clip = tf.clip_by_value(var, -bound, bound, "decoder_clip")

        exp_mean = tf.exp(mean_clip)
        exp_var = tf.exp(var_clip)

        return [exp_mean, exp_var]
