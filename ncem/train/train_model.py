import os
import pickle
from typing import Union

from ncem.estimators import (EstimatorCVAE, EstimatorCVAEncem, EstimatorED,
                             EstimatorEDncem, EstimatorEdNcemNeighborhood, EstimatorInteractions,
                             EstimatorLinear, EstimatorDeconvolution)


class TrainModel:
    estimator: Union[
        EstimatorLinear, EstimatorInteractions, EstimatorED, EstimatorEDncem, EstimatorCVAE, EstimatorCVAEncem,
        EstimatorDeconvolution
    ]

    @staticmethod
    def _try_save(fn: str, obj):
        with open(fn, "wb") as f:
            pickle.dump(obj=obj, file=f)

    def _save_model(self, fn: str, save_weights: bool = True):
        if save_weights:
            self.estimator.model.training_model.save_weights(fn + "_model_weights.tf")
        self._try_save(fn + "_model_args.pickle", self.estimator.model.args)

    def _save_history(self, fn: str):
        self._try_save(fn + "_history.pickle", self.estimator.history)

    def _save_hyperparam(self, fn: str):
        self._try_save(fn + "_hyperparam.pickle", self.estimator.train_hyperparam)

    def _save_predictions(self, fn: str):
        prediction = self.estimator.predict()
        self._try_save(fn + "_prediction.pickle", prediction)

    def _save_evaluation(self, fn: str):
        evaluation_val = self.estimator.evaluate_any(
            img_keys=self.estimator.img_keys_eval, node_idx=self.estimator.nodes_idx_eval
        )
        evaluation_train = self.estimator.evaluate_any(
            img_keys=self.estimator.img_keys_train, node_idx=self.estimator.nodes_idx_train
        )
        evaluation_all = self.estimator.evaluate_any(
            img_keys=self.estimator.img_keys_all, node_idx=self.estimator.nodes_idx_all
        )
        evaluations = {"train": evaluation_train, "val": evaluation_val, "all": evaluation_all}
        if len(self.estimator.img_keys_test) > 0:
            evaluation_test = self.estimator.evaluate_any(
                img_keys=self.estimator.img_keys_test, node_idx=self.estimator.nodes_idx_test
            )
            evaluations["test"] = evaluation_test
        self._try_save(fn + "_evaluation.pickle", evaluations)

    def _save_indices(self, fn: str):
        if not os.path.isfile(fn + "_indices.pickle"):
            indices = {
                "all": self.estimator.complete_img_keys,
                "test": self.estimator.img_keys_test,
                "train": self.estimator.img_keys_train,
                "val": self.estimator.img_keys_eval,
                "test_nodes": self.estimator.nodes_idx_test,
                "train_nodes": self.estimator.nodes_idx_train,
                "val_nodes": self.estimator.nodes_idx_eval,
            }
            self._try_save(fn + "_indices.pickle", indices)

    def _save_data_input(self, fn: str):
        if not os.path.isfile(fn + "_datainput.pickle"):
            info = {"img_to_patient_dict": self.estimator.img_to_patient_dict}
            self._try_save(fn + "_datainput.pickle", info)

    def save_time(self, fn: str, duration):
        with open(fn + "_time.pickle", "wb") as f:
            pickle.dump(obj=duration, file=f)

    def _save_specific(self, fn):
        # save some model specific interesting stuff (override in specific Train classes)
        pass

    def _save_evaluation_per_node_type(self, fn: str):
        split_per_node_type, evaluation_per_node_type = self.estimator.evaluate_per_node_type()
        self._try_save(fn + "_ntindices.pickle", split_per_node_type)
        self._try_save(fn + "_ntevaluation.pickle", evaluation_per_node_type)

    def save(self, fn: str, save_weights: bool = True):
        """
        Save weights and summary statistics.
        """
        assert self.estimator is not None, "initialize estimator first"
        self._save_model(fn=fn, save_weights=save_weights)

        self._save_history(fn=fn)
        self._save_hyperparam(fn=fn)
        self._save_specific(fn=fn)
        self._save_evaluation(fn=fn)

        self._save_indices(fn=fn)
        self._save_data_input(fn=fn)


class TrainModelLinear(TrainModel):
    estimator: EstimatorLinear

    def init_estim(self, **kwargs):
        self.estimator = EstimatorLinear(**kwargs)

    def _save_specific(self, fn):
        self._save_evaluation_per_node_type(fn=fn)


class TrainModelLinearDeconvolution(TrainModel):
    estimator: EstimatorDeconvolution

    def init_estim(self, **kwargs):
        self.estimator = EstimatorDeconvolution(**kwargs)

    def _save_specific(self, fn):
        self._save_evaluation_per_node_type(fn=fn)


class TrainModelInteractions(TrainModel):
    estimator: EstimatorInteractions

    def init_estim(self, **kwargs):
        self.estimator = EstimatorInteractions(**kwargs)

    def _save_specific(self, fn):
        self._save_evaluation_per_node_type(fn=fn)


class TrainModelED(TrainModel):
    estimator: EstimatorED

    def init_estim(self, **kwargs):
        self.estimator = EstimatorED(**kwargs)

    def _save_specific(self, fn):
        self._save_evaluation_per_node_type(fn=fn)


class TrainModelEDncemBase:

    estimator: Union[EstimatorEDncem, EstimatorEdNcemNeighborhood]

    def _save_embedding(self, fn):
        splits = {
            "eval": {"img_keys": self.estimator.img_keys_eval, "node_idx": self.estimator.nodes_idx_eval},
            "test": {"img_keys": self.estimator.img_keys_test, "node_idx": self.estimator.nodes_idx_test},
            "train": {"img_keys": self.estimator.img_keys_train, "node_idx": self.estimator.nodes_idx_train},
        }
        embeddings = {}
        for k, v in splits.items():
            embeddings[k] = self.estimator.predict_embedding_any(**v)
        with open(fn + "_embeddings.pickle", "wb") as f:
            pickle.dump(obj=embeddings, file=f)

    def _save_output_weights(self, fn):
        w = self.estimator.get_decoding_weights()
        with open(fn + "_output_weights.pickle", "wb") as f:
            pickle.dump(obj=w, file=f)


class TrainModelEDncem(TrainModel, TrainModelEDncemBase):
    estimator: EstimatorEDncem

    def init_estim(self, **kwargs):
        self.estimator = EstimatorEDncem(**kwargs)

    def _save_specific(self, fn):
        self._save_evaluation_per_node_type(fn=fn)


class TrainModelEdSingleNcem(TrainModel, TrainModelEDncemBase):
    estimator: EstimatorEdNcemNeighborhood

    def init_estim(self, **kwargs):
        self.estimator = EstimatorEdNcemNeighborhood(**kwargs)

    def _save_lr_feature_names(self, fn):
        feature_names = {
            "features": self.estimator.features,
            "target_feature_names": self.estimator.target_feature_names,
            "neighbor_feature_names": self.estimator.neighbor_feature_names,
        }
        self._try_save(fn + "_lr_features.pickle", feature_names)

    def _save_specific(self, fn):
        self._save_evaluation_per_node_type(fn=fn)
        self._save_lr_feature_names(fn=fn)
        self._save_embedding(fn=fn)
        self._save_output_weights(fn=fn)


class TrainModelCVAEBase(TrainModel):
    estimator: Union[EstimatorCVAE, EstimatorCVAEncem]

    def _save_evaluation_posterior_sampling(self, fn: str):
        (
            evaluation_val,
            true_val,
            pred_val,
            z_val,
            z_mean_val,
            z_log_var_val,
        ) = self.estimator.evaluate_any_posterior_sampling(
            img_keys=self.estimator.img_keys_eval, node_idx=self.estimator.nodes_idx_eval
        )
        (
            evaluation_train,
            true_train,
            pred_train,
            z_train,
            z_mean_train,
            z_log_var_train,
        ) = self.estimator.evaluate_any_posterior_sampling(
            img_keys=self.estimator.img_keys_train, node_idx=self.estimator.nodes_idx_train
        )
        (
            evaluation_all,
            true_all,
            pred_all,
            z_all,
            z_mean_all,
            z_log_var_all,
        ) = self.estimator.evaluate_any_posterior_sampling(
            img_keys=self.estimator.img_keys_all, node_idx=self.estimator.nodes_idx_all
        )
        evaluations = {"train": evaluation_train, "val": evaluation_val, "all": evaluation_all}
        predictions = {
            "train": (true_train, pred_train),
            "val": (true_val, pred_val),
            "all": (true_all, pred_all),
        }
        latent_space = {
            "train": (z_train, z_mean_train, z_log_var_train),
            "val": (z_val, z_mean_val, z_log_var_val),
            "all": (z_all, z_mean_all, z_log_var_all),
        }
        if len(self.estimator.img_keys_test) > 0:
            (
                evaluation_test,
                true_test,
                pred_test,
                z_test,
                z_mean_test,
                z_log_var_test,
            ) = self.estimator.evaluate_any_posterior_sampling(
                img_keys=self.estimator.img_keys_test, node_idx=self.estimator.nodes_idx_test
            )
            evaluations["test"] = evaluation_test
            predictions["test"] = (true_test, pred_test)
            latent_space["test"] = (z_test, z_mean_test, z_log_var_test)
        self._try_save(fn + "_evaluation_posterior_sampling.pickle", evaluations)
        self._try_save(fn + "_prediction_posterior_sampling.pickle", predictions)
        self._try_save(fn + "_latent_space_posterior_sampling.pickle", latent_space)


class TrainModelCVAE(TrainModelCVAEBase):
    estimator: EstimatorCVAE

    def init_estim(self, **kwargs):
        self.estimator = EstimatorCVAE(**kwargs)

    def _save_specific(self, fn):
        self._save_evaluation_posterior_sampling(fn=fn)
        self._save_evaluation_per_node_type(fn=fn)


class TrainModelCVAEncem(TrainModelCVAEBase):
    estimator: EstimatorCVAEncem

    def init_estim(self, **kwargs):
        self.estimator = EstimatorCVAEncem(**kwargs)

    def _save_specific(self, fn):
        self._save_evaluation_posterior_sampling(fn=fn)
        self._save_evaluation_per_node_type(fn=fn)
        self._try_save(fn + "_pretrain_history.pickle", self.estimator.pretrain_history)
