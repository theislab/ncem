import math

import numpy as np
import tensorflow as tf


class NegBinLoss(tf.keras.losses.Loss):
    """Custom negative binomial loss."""

    def call(self, y_true, y_pred):
        """Implement the negative log likelihood loss as reconstruction loss.

        Parameters
        ----------
        y_true
            y_true.
        y_pred
            y_pred.

        Returns
        -------
        neg_ll
            negative log likelihood loss as reconstruction loss.
        """
        x = y_true
        loc, scale = tf.split(y_pred, num_or_size_splits=2, axis=2)

        eta_loc = tf.math.log(loc)
        eta_scale = tf.math.log(scale)

        log_r_plus_mu = tf.math.log(scale + loc)

        ll = tf.math.lgamma(scale + x)
        ll = ll - tf.math.lgamma(x + tf.ones_like(x))
        ll = ll - tf.math.lgamma(scale)
        ll = ll + tf.multiply(x, eta_loc - log_r_plus_mu) + tf.multiply(scale, eta_scale - log_r_plus_mu)

        ll = tf.clip_by_value(ll, -300, 300, "log_probs")
        neg_ll = (tf.reduce_sum(-ll, axis=-1),)  # sum across output features
        return neg_ll


class GaussianLoss(tf.keras.losses.Loss):
    """Custom gaussian loss."""

    def call(self, y_true, y_pred):
        """Implement Gaussian loss as reconstruction loss.

        Parameters
        ----------
        y_true
            y_true.
        y_pred
            y_pred.

        Returns
        -------
        neg_ll
            Gaussian loss as reconstruction loss.
        """
        y_pred, sd = tf.split(y_pred, num_or_size_splits=2, axis=2, name="gaussian_loss_split")
        # sd = 1.  # change also in metric if this is changed

        neg_ll = tf.math.log(tf.sqrt(2 * math.pi) * sd) + 0.5 * tf.math.square(y_pred - y_true) / tf.math.square(sd)
        neg_ll = tf.reduce_sum(neg_ll, axis=-1)  # sum across output features
        return neg_ll


class KLLoss(tf.keras.losses.Loss):
    """Custom gaussian loss."""

    def __init__(self, beta: float = 1.0, max_beta: float = 1.0, pre_warm_up: int = 0):
        """Initialize Kullback Leibler divergence.

        Parameters
        ----------
        beta : float
            Beta.
        max_beta : float
            Maximal beta.
        pre_warm_up : int
            Pre warm up.
        """
        super().__init__()
        self.beta = tf.Variable(beta, dtype=tf.float32, trainable=False)
        self.max_beta = max_beta
        self.pre_warm_up = pre_warm_up

    def call(self, y_true, y_pred):
        """Call Kullback-Leibler divergence.

        Parameters
        ----------
        y_true
            y_true.
        y_pred
            y_pred.

        Returns
        -------
        kl_loss
            Kullback-Leibler divergence.
        """
        z, z_mean, z_log_var = tf.split(y_pred, num_or_size_splits=3, axis=1)
        log2pi = tf.math.log(2.0 * np.pi)
        logqz_x = -0.5 * tf.reduce_sum(tf.square(z - z_mean) * tf.exp(-z_log_var) + z_log_var + log2pi, axis=-1)
        logpz = -0.5 * tf.reduce_sum(tf.square(z) + log2pi, axis=-1)
        kl_loss = logqz_x - logpz
        return self.beta * kl_loss
