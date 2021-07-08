import math

import tensorflow as tf
import tensorflow.experimental.numpy as tnp


def custom_mae(y_true, y_pred):
    """Compute custom mean absolute error metric.

    Parameters
    ----------
    y_true
        y_true.
    y_pred
        y_pred.

    Returns
    -------
    mae
    """
    y_pred, _ = tf.split(y_pred, num_or_size_splits=2, axis=2)
    abs = tf.abs(y_true - y_pred)
    mean = tf.reduce_mean(abs)
    return mean


def custom_mse(y_true, y_pred):
    """Compute custom mean squared error metric.

    Parameters
    ----------
    y_true
        y_true.
    y_pred
        y_pred.

    Returns
    -------
    mse
    """
    y_pred, _ = tf.split(y_pred, num_or_size_splits=2, axis=2)
    se = tf.square(y_true - y_pred)
    se = tf.reduce_mean(se, axis=-1)
    return se


def custom_mean_sd(y_true, y_pred):
    """Compute custom mean standard deviation metric.

    Parameters
    ----------
    y_true
        y_true.
    y_pred
        y_pred.

    Returns
    -------
    sd
    """
    _, sd = tf.split(y_pred, num_or_size_splits=2, axis=2)
    sd = tf.reduce_mean(sd)
    return sd


def logp1_custom_mse(y_true, y_pred):
    """Compute custom logp1 mean squared error metric.

    Parameters
    ----------
    y_true
        y_true.
    y_pred
        y_pred.

    Returns
    -------
    logp1_mse
    """
    y_pred, _ = tf.split(y_pred, num_or_size_splits=2, axis=2)
    y_true = tf.math.log(y_true + 1.0)
    y_pred = tf.math.log(y_pred + 1.0)
    se = tf.square(y_true - y_pred)
    se = tf.reduce_mean(se, axis=-1)
    return se


def custom_mse_scaled(y_true, y_pred):
    """Compute custom mean squared error scaled metric.

    Parameters
    ----------
    y_true
        y_true.
    y_pred
        y_pred.

    Returns
    -------
    mse_scaled
    """
    y_pred, sd = tf.split(y_pred, num_or_size_splits=2, axis=2)
    se = tf.square(y_true - y_pred) / tf.square(sd)
    se = tf.reduce_mean(se, axis=-1)
    return se


def gaussian_reconstruction_loss(y_true, y_pred):
    """Compute custom gaussian reconstruction loss.

    Parameters
    ----------
    y_true
        y_true.
    y_pred
        y_pred.

    Returns
    -------
    neg_ll
    """
    y_pred, sd = tf.split(y_pred, num_or_size_splits=2, axis=2)
    neg_ll = tf.math.log(tf.sqrt(2 * math.pi) * sd) + 0.5 * tf.math.square(y_pred - y_true) / tf.math.square(sd)
    neg_ll = tf.reduce_sum(neg_ll, axis=-1)  # sum across output features
    return neg_ll


def nb_reconstruction_loss(y_true, y_pred):
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


def custom_kl(y_true, y_pred):
    """Compute custom kullback-leibler divergence.

    Parameters
    ----------
    y_true
        y_true.
    y_pred
        y_pred.

    Returns
    -------
    kl_loss
    """
    z, z_mean, z_log_var = tf.split(y_pred, num_or_size_splits=3, axis=1)
    log2pi = tf.math.log(2.0 * math.pi)
    logqz_x = -0.5 * tf.reduce_sum(tf.square(z - z_mean) * tf.exp(-z_log_var) + z_log_var + log2pi, axis=-1)
    logpz = -0.5 * tf.reduce_sum(tf.square(z) + log2pi, axis=-1)
    kl_loss = logqz_x - logpz
    return kl_loss


def r_squared(y_true, y_pred):
    """Compute custom r squared.

    Parameters
    ----------
    y_true
        y_true.
    y_pred
        y_pred.

    Returns
    -------
    r2
    """
    y_pred, _ = tf.split(y_pred, num_or_size_splits=2, axis=2)
    residual = tf.reduce_sum(tf.square(y_true - y_pred))
    total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    return r2


def r_squared_linreg(y_true, y_pred):
    """Compute custom r squared (follows the scipy linear regression implementation of R2).

    Parameters
    ----------
    y_true
        y_true.
    y_pred
        y_pred.

    Returns
    -------
    r2
    """
    y_pred, _ = tf.split(y_pred, num_or_size_splits=2, axis=2)
    x = y_true
    y = y_pred
    # means
    xmean = tnp.mean(x)
    ymean = tnp.mean(y)
    # covariance
    ssxm = tnp.mean(tnp.square(x - xmean))
    ssym = tnp.mean(tnp.square(y - ymean))
    ssxym = tnp.mean((x - xmean) * (y - ymean))
    xmym = ssxm * ssym

    # Helper functions for tf.cond
    def f0():
        return tnp.zeros(shape=(), dtype=tf.float32)

    def f1():
        return tnp.ones(shape=(), dtype=tf.float32)

    def r():
        return ssxym / tnp.sqrt(xmym)  # formula for r

    def r2():
        return r ** 2  # formula for r_squared

    # R-value
    # If the denominator was going to be 0, r = 0.0
    r = tf.cond(tnp.not_equal(xmym, tnp.zeros(shape=(), dtype=tf.float32)), r, f0)
    # Test for numerical error propagation (make sure -1 < r < 1)
    r_squared = tf.cond(tnp.greater(tnp.abs(r), tnp.ones(shape=(), dtype=tf.float32)), f1, r2)
    return r_squared


def logp1_r_squared(y_true, y_pred):
    """Compute custom logp1 r squared.

    Parameters
    ----------
    y_true
        y_true.
    y_pred
        y_pred.

    Returns
    -------
    r2
    """
    y_pred, _ = tf.split(y_pred, num_or_size_splits=2, axis=2)
    y_true = tf.math.log(y_true + 1.0)
    y_pred = tf.math.log(y_pred + 1.0)
    residual = tf.reduce_sum(tf.square(y_true - y_pred))
    total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    return r2


def logp1_r_squared_linreg(y_true, y_pred):
    """Compute custom logp1 r squared ((follows the scipy linear regression implementation of R2).

    Parameters
    ----------
    y_true
        y_true.
    y_pred
        y_pred.

    Returns
    -------
    r2
    """
    y_pred, _ = tf.split(y_pred, num_or_size_splits=2, axis=2)

    x = tf.math.log(y_true + 1.0)
    y = tf.math.log(y_pred + 1.0)

    # means
    xmean = tnp.mean(x)
    ymean = tnp.mean(y)

    ssxm = tnp.mean(tnp.square(x - xmean))
    ssym = tnp.mean(tnp.square(y - ymean))
    ssxym = tnp.mean((x - xmean) * (y - ymean))

    # R-value
    r = ssxym / tnp.sqrt(ssxm * ssym)
    return r ** 2
