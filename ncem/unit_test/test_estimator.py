import unittest

import ncem.api as ncem


class TestEstimator(unittest.TestCase):
    data_path_zhang = "/Users/anna.schaar/phd/datasets/zhang/"
    data_path_jarosch = "/Users/anna.schaar/phd/datasets/busch/"
    data_path_hartmann = "/Users/anna.schaar/phd/datasets/hartmann/"
    data_path_schuerch = "/Users/anna.schaar/phd/datasets/schuerch/buffer/"

    def get_estimator(
        self,
        model: str,
        data_origin: str = "zhang",
    ):
        node_label_space_id = "type"
        if model == "linear":
            self.est = ncem.train.EstimatorLinear()
            node_feature_space_id = "type"
        else:
            assert False

        if data_origin == "zhang":
            radius = 100
            data_path = self.data_path_zhang
        elif data_origin == "hartmann":
            radius = 100
            data_path = self.data_path_hartmann
        else:
            assert False

        self.est.get_data(
            data_origin=data_origin,
            data_path=data_path,
            radius=radius,
            node_label_space_id=node_label_space_id,
            node_feature_space_id=node_feature_space_id,
            merge_node_types_predefined=True,
        )

    def _test_train(self, model: str, data_origin: str = "zhang"):
        self.get_estimator(model=model, data_origin=data_origin)

        if model == 'linear':
            kwargs = {
                "use_domain": True,
                "learning_rate": 1e-2
            }
            train_kwargs = {}
        else:
            assert False
        self._model_kwargs = kwargs
        self.est.init_model(**kwargs)
        self.est.split_data_node(
            validation_split=0.5,
            test_split=0.5
        )

        if data_origin == 'hartmann':
            batch_size = None
        else:
            batch_size = 16

        if batch_size is None:
            bs = len(list(self.est.complete_img_keys))
            shuffle_buffer_size = None  # None
        else:
            bs = batch_size
            shuffle_buffer_size = int(2)
        self.est.train(
            epochs=5,
            max_steps_per_epoch=2,
            batch_size=bs,
            validation_batch_size=6,
            max_validation_steps=1,
            shuffle_buffer_size=shuffle_buffer_size,
            log_dir=None,
            **train_kwargs
        )

    def test_linear(self):
        self._test_train(model="linear", data_origin="hartmann")
