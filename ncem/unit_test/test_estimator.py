import unittest
from typing import Tuple, Union

import numpy as np
import scipy.sparse

import ncem.api as ncem
import ncem.data as data
from ncem.models.custom_callbacks import BetaScheduler


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

    def test_linear(self):
        self._test_train(model="linear", data_origin="zhang")
