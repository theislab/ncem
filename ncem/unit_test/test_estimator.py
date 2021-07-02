import unittest

import ncem.api as ncem


class TestEstimator(unittest.TestCase):
    base_path = "/Users/anna.schaar/phd/datasets/"
    data_path_zhang = base_path + "zhang/"
    data_path_jarosch = base_path + "busch/"
    data_path_hartmann = base_path + "hartmann/"
    data_path_schuerch = base_path + "buffer/"

    def get_estimator(
        self,
        model: str,
        data_origin: str = "zhang",
    ):
        node_label_space_id = "type"
        node_feature_space_id = "standard"
        if model in ["linear_baseline", "linear"]:
            self.est = ncem.train.EstimatorLinear()
        elif model in ["interactions_baseline", "interactions"]:
            self.est = ncem.train.EstimatorInteractions()
        elif model == "ed":
            self.est = ncem.train.EstimatorED()
        elif model == "ed_ncem_max":
            self.est = ncem.train.EstimatorEDncem(cond_type="max")
        elif model == "ed_ncem_gcn":
            self.est = ncem.train.EstimatorEDncem(cond_type="gcn")
        elif model == "cvae":
            self.est = ncem.train.EstimatorCVAE()
        elif model == "cvae_ncem_max":
            self.est = ncem.train.EstimatorCVAEncem(cond_type="max")
        elif model == "cvae_ncem_gcn":
            self.est = ncem.train.EstimatorCVAEncem(cond_type="gcn")
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

        if model == "linear":
            kwargs = {"use_source_type": True, "use_domain": True, "learning_rate": 1e-2}
            train_kwargs = {}
        elif model == "linear_baseline":
            kwargs = {"use_source_type": False, "use_domain": True, "learning_rate": 1e-2}
            train_kwargs = {}
        elif model == "interactions":
            kwargs = {"use_interactions": True, "use_domain": True, "learning_rate": 1e-2}
            train_kwargs = {}
        elif model == "interactions_baseline":
            kwargs = {"use_interactions": False, "use_domain": True, "learning_rate": 1e-2}
            train_kwargs = {}
        elif model == "ed":
            kwargs = {
                "depth_enc": 1,
                "depth_dec": 0,
                "use_domain": True,
                "use_bias": True,
                "learning_rate": 1e-2,
                "beta": 0.1,
            }
            train_kwargs = {}
        elif model in ["ed_ncem_max", "ed_ncem_gcn"]:
            kwargs = {
                "depth_enc": 1,
                "depth_dec": 0,
                "cond_depth": 1,
                "use_domain": True,
                "use_bias": True,
                "learning_rate": 1e-2,
                "beta": 0.1,
            }
            train_kwargs = {}
        elif model == "cvae":
            kwargs = {
                "depth_enc": 1,
                "depth_dec": 1,
                "use_domain": True,
                "use_bias": True,
                "learning_rate": 1e-2,
                "beta": 0.1,
            }
            train_kwargs = {}
        elif model in ["cvae_ncem_max", "cvae_ncem_gcn"]:
            kwargs = {
                "depth_enc": 1,
                "depth_dec": 1,
                "cond_depth": 1,
                "use_domain": True,
                "use_bias": True,
                "learning_rate": 1e-2,
                "beta": 0.1,
            }
            train_kwargs = {}
        else:
            assert False
        self._model_kwargs = kwargs
        self.est.init_model(**kwargs)
        self.est.split_data_node(validation_split=0.5, test_split=0.5)

        if data_origin == "hartmann":
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
        self.est.model.training_model.summary()

    def test_linear(self):
        # self._test_train(model='linear_baseline', data_origin='hartmann')
        # self._test_train(model="linear", data_origin="hartmann")
        # self._test_train(model='interactions_baseline', data_origin='hartmann')
        self._test_train(model="interactions", data_origin="hartmann")

    def test_ed(self):
        # self._test_train(model="ed", data_origin="hartmann")
        self._test_train(model="ed_ncem_max", data_origin="hartmann")
        # self._test_train(model="ed_ncem_gcn", data_origin="hartmann")

    def test_cvae(self):
        # self._test_train(model="cvae", data_origin="hartmann")
        self._test_train(model="cvae_ncem_max", data_origin="hartmann")
        # self._test_train(model="cvae_ncem_gcn", data_origin="hartmann")
