import numpy as np
import tensorflow as tf


class BetaScheduler(tf.keras.callbacks.Callback):
    """Beta parameter scheduler."""

    def __init__(self, verbose: int = 1):
        """Initialize BetaScheduler.

        Parameters
        ----------
        verbose : int
            0 quiet, 1 update messages.
        """
        super(BetaScheduler, self).__init__()
        self.verbose = verbose
        self.beta = None

    def on_epoch_begin(self, epoch, logs=None):
        """Create callback function on_epoch_begin for for BetaScheduler.

        Parameters
        ----------
        epoch
            Epoch.
        logs
            Logs.

        Raises
        ------
        ValueError
            If `beta`, `max_beta` or `pre_warm_up` not found in attribute.
        """
        if not hasattr(self.model.loss[1], "beta"):
            raise ValueError('Model must have a "beta" attribute.')
        if not hasattr(self.model.loss[1], "max_beta"):
            raise ValueError('Model must have a "max_beta" attribute.')
        if not hasattr(self.model.loss[1], "pre_warm_up"):
            raise ValueError('Model must have a "pre_warm_up" attribute.')
        if epoch == 0:
            self.beta = tf.keras.backend.get_value(self.model.loss[1].beta)

        value = np.maximum(
            np.minimum(self.model.loss[1].max_beta, (epoch + 1 - self.model.loss[1].pre_warm_up) * self.beta), 0.0
        )
        tf.keras.backend.set_value(self.model.loss[1].beta, value)
        if self.verbose > 0:
            print(
                "\nEpoch %05d (starting %05d): BetaScheduler setting beta to %s / %f."
                % (epoch + 1, self.model.loss[1].pre_warm_up, value, self.model.loss[1].max_beta)
            )

    def on_epoch_end(self, epoch, logs=None):
        """Create callback function on_epoch_begin for for BetaScheduler.

        Parameters
        ----------
        epoch
            Epoch.
        logs
            Logs.
        """
        logs = logs or {}
        logs["beta"] = tf.keras.backend.get_value(self.model.loss[1].beta)
