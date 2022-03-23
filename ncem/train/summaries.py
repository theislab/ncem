import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from tqdm import tqdm
from matplotlib.ticker import FormatStrFormatter


class GridSearchContainer:
    """GridSearchContainer class."""

    runparams: dict
    run_ids_clean: dict
    source_gs: dict
    cv_ids: dict

    target_cell_runparams: dict
    target_cell_evals: dict
    target_cell_indices: dict

    def __init__(self, source_path, gs_ids, lateral_resolution):
        """Initialize GridSearchContainer.

        Parameters
        ----------
        source_path
            Source path.
        gs_ids
            Grid search identifiers.
        lateral_resolution
            Lateral resolution.
        """
        self.source_path = source_path
        if isinstance(gs_ids, str):
            gs_ids = [gs_ids]
        self.gs_ids = gs_ids
        self.lateral_resolution = lateral_resolution

        self.evals = {}
        self.evals_posterior_sampling = {}
        self.histories = {}

        self.summary_table = None
        self.runparams_table = None
        self.target_cell_table = None
        self.best_model_hyperparam_table = None
        self.target_cell_frequencies = None

    @property
    def cv_keys(self) -> List[str]:
        """Return keys of cross-validation used in dictionaries in this class.

        Returns
        -------
        list of string keys
        """
        return np.unique(self.summary_table["cv"].values).tolist()

    def load_gs(
        self,
        expected_pickle: Optional[list] = None,
        add_posterior_sampling_model: bool = False,
        report_unsuccessful_runs: bool = False,
    ):
        """Load all metrics from grid search output files.

        Core results are save in self.summary_table.

        Parameters
        ----------
        expected_pickle : list, optional
            Expected pickle files.
        add_posterior_sampling_model : bool
            Whether to add posterior sampling model as seperate model to summary_table.
        report_unsuccessful_runs : bool
            Whether to print reporting statements in out stream.

        Raises
        ------
        ValueError
            If no complete runs found.
        """
        if expected_pickle is None:
            expected_pickle = ["evaluation", "history", "hyperparam", "model_args", "time"]

        if add_posterior_sampling_model:
            expected_pickle += ["evaluation_posterior_sampling"]
        self.summary_table = []
        self.runparams_table = []
        self.runparams = {}
        self.run_ids_clean = {}

        self.evals = {}
        self.evals_posterior_sampling = {}
        self.histories = {}
        self.source_gs = {}
        self.cv_ids = {}
        with tqdm(total=len(self.gs_ids)) as pbar:
            for gs_id in self.gs_ids:
                # Collect runs that belong to grid search by looping over file names in directory.
                indir = self.source_path + gs_id + "/results/"
                # runs_ids are the unique hyper-parameter settings, which are again subsetted by cross-validation.
                # These ids are present in all files names but are only collected from the *model.tf file names here.

                run_ids = np.sort(
                    np.unique(
                        [
                            "_".join(".".join(x.split(".")[:-1]).split("_")[:-1])
                            for x in os.listdir(indir)
                            if x.split("_")[-1].split(".")[0] == "time"
                        ]
                    )
                )
                cv_ids = np.sort(np.unique([x.split("_")[-1] for x in run_ids]))  # identifiers of cross-validation splits
                run_ids = np.sort(
                    np.unique(["_".join(x.split("_")[:-1]) for x in run_ids])  # identifiers of hyper-parameters settings
                )
                run_ids_clean = []  # only IDs of completed runs (all files present)
                for r in run_ids:
                    complete_run = True
                    for cv in cv_ids:
                        # Check pickled files:
                        for end in expected_pickle:
                            fn = r + "_" + cv + "_" + end + ".pickle"
                            if not os.path.isfile(indir + fn):
                                if report_unsuccessful_runs:
                                    print("File %r missing" % fn)
                                complete_run = False
                    # Check run parameter files (one per cross-validation set):
                    fn = r + "_runparams.pickle"
                    if not os.path.isfile(indir + fn):
                        if report_unsuccessful_runs:
                            print("File %r missing" % fn)
                        complete_run = False
                    if not complete_run:
                        if report_unsuccessful_runs:
                            print("Run %r not successful" % r + "_" + cv)
                    else:
                        run_ids_clean.append(r)
                # Load results and settings from completed runs:
                evals = (
                    {}
                )  # Dictionary over runs with dictionary over cross-validations with results from model evaluation.
                runparams = {}  # Dictionary over runs with model settings.
                histories = {}
                for x in run_ids_clean:
                    # Load model settings (these are shared across all partitions).
                    fn_runparams = indir + x + "_runparams.pickle"
                    with open(fn_runparams, "rb") as f:
                        runparams[x] = pickle.load(f)
                    evals[x] = {}
                    for cv in cv_ids:
                        fn_eval = indir + x + "_" + cv + "_evaluation.pickle"
                        with open(fn_eval, "rb") as f:
                            evals[x][cv] = pickle.load(f)
                    histories[x] = {}
                    for cv in cv_ids:
                        fn_eval = indir + x + "_" + cv + "_history.pickle"
                        with open(fn_eval, "rb") as f:
                            histories[x][cv] = pickle.load(f)
                self.runparams[gs_id] = runparams
                self.run_ids_clean[gs_id] = run_ids_clean
                self.evals[gs_id] = evals
                self.histories[gs_id] = histories
                self.cv_ids[gs_id] = cv_ids
                for run_id in run_ids_clean:
                    self.source_gs[run_id] = gs_id
                if len(run_ids_clean) == 0:
                    raise ValueError("no complete runs found")

                # Summarise all metrics in a single table with rows for each runs and CV partition.
                params_list = list(
                    {
                        "data_set": [runparams[x]["data_set"] for x in run_ids_clean],
                        "model_class": [runparams[x]["model_class"] for x in run_ids_clean],
                        "cond_type": [runparams[x]["cond_type"] for x in run_ids_clean]
                        if "cond_type" in list(runparams[x].keys())
                        else "none",
                        "gs_id": [runparams[x]["gs_id"] for x in run_ids_clean],
                        "model_id": [runparams[x]["model_id"] for x in run_ids_clean],
                        #"split_mode": [runparams[x]["split_mode"] for x in run_ids_clean],
                        "radius": [runparams[x]["radius"] for x in run_ids_clean] if "radius" in list(runparams[x].keys()) else [runparams[x]["max_dist"] for x in run_ids_clean],
                        "n_rings": [runparams[x]["n_rings"] for x in run_ids_clean] if "n_rings" in list(runparams[x].keys()) else "none",
                        "graph_covar_selection": [runparams[x]["graph_covar_selection"] for x in run_ids_clean],
                        #"node_label_space_id": [runparams[x]["node_label_space_id"] for x in run_ids_clean],
                        #"node_feature_space_id": [runparams[x]["node_feature_space_id"] for x in run_ids_clean],
                        #"feature_transformation": [runparams[x]["feature_transformation"] for x in run_ids_clean],
                        "use_covar_node_position": [runparams[x]["use_covar_node_position"] for x in run_ids_clean],
                        "use_covar_node_label": [runparams[x]["use_covar_node_label"] for x in run_ids_clean],
                        "use_covar_graph_covar": [runparams[x]["use_covar_graph_covar"] for x in run_ids_clean],
                        "target_cell_type": [runparams[x]["target_cell_type"] for x in run_ids_clean]
                        if "target_cell_type" in list(runparams[x].keys())
                        else "none",
                        "optimizer": [runparams[x]["optimizer"] for x in run_ids_clean],
                        "learning_rate": [runparams[x]["learning_rate"] for x in run_ids_clean],
                        "intermediate_dim_enc": [runparams[x]["intermediate_dim_enc"] for x in run_ids_clean]
                        if "intermediate_dim_enc" in list(runparams[x].keys())
                        else "none",
                        "intermediate_dim_dec": [runparams[x]["intermediate_dim_dec"] for x in run_ids_clean]
                        if "intermediate_dim_dec" in list(runparams[x].keys())
                        else "none",
                        "latent_dim": [runparams[x]["latent_dim"] for x in run_ids_clean]
                        if "latent_dim" in list(runparams[x].keys())
                        else "none",
                        "depth_enc": [runparams[x]["depth_enc"] for x in run_ids_clean]
                        if "depth_enc" in list(runparams[x].keys())
                        else "none",
                        "depth_dec": [runparams[x]["depth_dec"] for x in run_ids_clean]
                        if "depth_dec" in list(runparams[x].keys())
                        else "none",
                        "dropout_rate": [runparams[x]["dropout_rate"] for x in run_ids_clean]
                        if "dropout_rate" in list(runparams[x].keys())
                        else "none",
                        "l2_coef": [runparams[x]["l2_coef"] for x in run_ids_clean]
                        if "l2_coef" in list(runparams[x].keys())
                        else "none",
                        "l1_coef": [runparams[x]["l1_coef"] for x in run_ids_clean]
                        if "l1_coef" in list(runparams[x].keys())
                        else "none",
                        "pretrain_decoder": [runparams[x]["pretrain_decoder"] for x in run_ids_clean]
                        if "pretrain_decoder" in list(runparams[x].keys())
                        else "none",
                        "aggressive": [runparams[x]["aggressive"] for x in run_ids_clean]
                        if "aggressive" in list(runparams[x].keys())
                        else "none",
                        "cond_depth": [str(runparams[x]["cond_depth"]) for x in run_ids_clean]
                        if "cond_depth" in list(runparams[x].keys())
                        else "none",
                        "cond_dim": [str(runparams[x]["cond_dim"]) for x in run_ids_clean]
                        if "cond_dim" in list(runparams[x].keys())
                        else "none",
                        "cond_dropout_rate": [str(runparams[x]["cond_dropout_rate"]) for x in run_ids_clean]
                        if "cond_dropout_rate" in list(runparams[x].keys())
                        else "none",
                        "cond_activation": [str(runparams[x]["cond_activation"]) for x in run_ids_clean]
                        if "cond_activation" in list(runparams[x].keys())
                        else "none",
                        "cond_l2_reg": [str(runparams[x]["cond_l2_reg"]) for x in run_ids_clean]
                        if "cond_l2_reg" in list(runparams[x].keys())
                        else "none",
                        "cond_use_bias": [str(runparams[x]["cond_use_bias"]) for x in run_ids_clean]
                        if "cond_bias" in list(runparams[x].keys())
                        else "none",
                        "use_domain": [runparams[x]["use_domain"] for x in run_ids_clean],
                        "domain_type": [runparams[x]["domain_type"] for x in run_ids_clean],
                        "use_batch_norm": [runparams[x]["use_batch_norm"] for x in run_ids_clean] if "use_batch_norm" in list(runparams[x].keys())
                        else "none",
                        "use_type_cond": [runparams[x]["use_type_cond"] for x in run_ids_clean]
                        if "use_type_cond" in list(runparams[x].keys())
                        else "none",
                        "scale_node_size": [runparams[x]["scale_node_size"] for x in run_ids_clean],
                        "transform_input": [runparams[x]["transform_input"] for x in run_ids_clean] if "transform_input" in list(runparams[x].keys())
                        else "none",
                        "output_layer": [runparams[x]["output_layer"] for x in run_ids_clean],
                        "log_transform": [runparams[x]["log_transform"] for x in run_ids_clean]
                        if "log_transform" in list(runparams[x].keys())
                        else False,
                        "segmentation_robustness_node_fraction": [runparams[x]["segmentation_robustness_node_fraction"] for x in run_ids_clean]
                        if "segmentation_robustness_node_fraction" in list(runparams[x].keys())
                        else False,
                        "segmentation_robustness_overflow_fraction": [runparams[x]["segmentation_robustness_overflow_fraction"] for x in run_ids_clean]
                        if "segmentation_robustness_overflow_fraction" in list(runparams[x].keys())
                        else False,
                        "resimulate_nodes_w_depdency": [runparams[x]["resimulate_nodes_w_depdency"] for x in run_ids_clean]
                        if "resimulate_nodes_w_depdency" in list(runparams[x].keys())
                        else False,
                        "resimulate_nodes_sparsity_rate": [runparams[x]["resimulate_nodes_sparsity_rate"] for x in run_ids_clean]
                        if "resimulate_nodes_sparsity_rate" in list(runparams[x].keys())
                        else False,
                        "epochs": [runparams[x]["epochs"] for x in run_ids_clean],
                        "batch_size": [runparams[x]["batch_size"] for x in run_ids_clean],
                        "n_eval_nodes_per_graph": [runparams[x]["n_eval_nodes_per_graph"] for x in run_ids_clean],
                        "run_id_params": run_ids_clean,
                    }.items()
                )
                self.runparams_table.append(pd.DataFrame(dict(params_list)))

                self.summary_table.append(
                    pd.concat(
                        [
                            pd.DataFrame(
                                dict(
                                    list(params_list)
                                    + list(
                                        {
                                            "cv": cv,
                                            "run_id": run_ids_clean,
                                            "model": "_".join(gs_id.split("/")[-1].split("_")[1:-1]),
                                            "forward_pass_eval": True,
                                        }.items()
                                    )
                                    + list(
                                        dict(
                                            [
                                                (
                                                    "train_" + m.replace("reconstruction_", "").replace("logp1_", ""),
                                                    [evals[x][cv]["train"][m] for x in run_ids_clean],
                                                )
                                                for m in list(evals[run_ids_clean[0]][cv]["train"].keys())
                                            ]
                                        ).items()
                                    )
                                    + list(
                                        dict(
                                            [
                                                (
                                                    "val_" + m.replace("reconstruction_", "").replace("logp1_", ""),
                                                    [evals[x][cv]["val"][m] for x in run_ids_clean],
                                                )
                                                for m in list(evals[run_ids_clean[0]][cv]["val"].keys())
                                            ]
                                        ).items()
                                    )
                                    + list(
                                        dict(
                                            [
                                                (
                                                    "test_" + m.replace("reconstruction_", "").replace("logp1_", ""),
                                                    [evals[x][cv]["test"][m] for x in run_ids_clean],
                                                )
                                                for m in list(evals[run_ids_clean[0]][cv]["test"].keys())
                                            ]
                                        ).items()
                                    )
                                    + list(
                                        dict(
                                            [
                                                (
                                                    "all_" + m.replace("reconstruction_", "").replace("logp1_", ""),
                                                    [evals[x][cv]["all"][m] for x in run_ids_clean],
                                                )
                                                for m in list(evals[run_ids_clean[0]][cv]["all"].keys())
                                            ]
                                        ).items()
                                    )
                                )
                            )
                            for cv in cv_ids
                        ]
                    )
                )
                print("%s: loaded %i runs with %i-fold cross validation" % (gs_id, len(run_ids_clean), len(cv_ids)))

                if runparams[run_ids_clean[0]]["model_class"] in ["vae", "cvae", "cvae_ncem"]:
                    evals_posterior_sampling = {}  # Dictionary over runs with dictionary over cross-validations with
                    # results from model neighbourhood transfer evaluation.
                    for x in run_ids_clean:
                        evals_posterior_sampling[x] = {}
                        for cv in cv_ids:
                            if add_posterior_sampling_model:
                                fn_eval = indir + x + "_" + cv + "_evaluation_posterior_sampling.pickle"
                                with open(fn_eval, "rb") as f:
                                    evals_posterior_sampling[x][cv] = pickle.load(f)
                    self.evals_posterior_sampling[gs_id + "_POSTERIOR_SAMPLING"] = evals_posterior_sampling

                    if add_posterior_sampling_model:
                        self.summary_table.append(
                            pd.concat(
                                [
                                    pd.DataFrame(
                                        dict(
                                            list(params_list)
                                            + list(
                                                {
                                                    "cv": cv,
                                                    "run_id": [x + "_posterior_sampling" for x in run_ids_clean],
                                                    "model": "_".join(gs_id.split("/")[-1].split("_")[1:-1])
                                                    + "_POSTERIOR_SAMPLING",
                                                    "posterior_sampling_eval": True,
                                                }.items()
                                            )
                                            + list(
                                                dict(
                                                    [
                                                        (
                                                            "train_"
                                                            + m.replace("reconstruction_", "").replace("logp1_", ""),
                                                            [
                                                                evals_posterior_sampling[x][cv]["train"][m]
                                                                for x in run_ids_clean
                                                            ],
                                                        )
                                                        for m in list(
                                                            evals_posterior_sampling[run_ids_clean[0]][cv]["train"].keys()
                                                        )
                                                    ]
                                                ).items()
                                            )
                                            + list(
                                                dict(
                                                    [
                                                        (
                                                            "val_" + m.replace("reconstruction_", "").replace("logp1_", ""),
                                                            [
                                                                evals_posterior_sampling[x][cv]["val"][m]
                                                                for x in run_ids_clean
                                                            ],
                                                        )
                                                        for m in list(
                                                            evals_posterior_sampling[run_ids_clean[0]][cv]["val"].keys()
                                                        )
                                                    ]
                                                ).items()
                                            )
                                            + list(
                                                dict(
                                                    [
                                                        (
                                                            "test_"
                                                            + m.replace("reconstruction_", "").replace("logp1_", ""),
                                                            [
                                                                evals_posterior_sampling[x][cv]["test"][m]
                                                                for x in run_ids_clean
                                                            ],
                                                        )
                                                        for m in list(
                                                            evals_posterior_sampling[run_ids_clean[0]][cv]["test"].keys()
                                                        )
                                                    ]
                                                ).items()
                                            )
                                            + list(
                                                dict(
                                                    [
                                                        (
                                                            "all_" + m.replace("reconstruction_", "").replace("logp1_", ""),
                                                            [
                                                                evals_posterior_sampling[x][cv]["all"][m]
                                                                for x in run_ids_clean
                                                            ],
                                                        )
                                                        for m in list(
                                                            evals_posterior_sampling[run_ids_clean[0]][cv]["all"].keys()
                                                        )
                                                    ]
                                                ).items()
                                            )
                                        )
                                    )
                                    for cv in cv_ids
                                ]
                            )
                        )
                        print(
                            "%s: loaded posterior sampling as seperate model (%i runs with %i-fold cross validation)"
                            % (gs_id, len(run_ids_clean), len(cv_ids))
                        )
            pbar.update(1)

        self.summary_table = pd.concat(self.summary_table)
        self.runparams_table = pd.concat(self.runparams_table)
        self.summary_table["um_radius"] = (self.summary_table["radius"] * self.lateral_resolution).astype(int)

    def load_target_cell_evaluation(self, report_unsuccessful_runs: bool = False):
        """Load all metrics from grid search output files of target cell evaluation.

        Core results are save in self.target_cell_table.

        Parameters
        ----------
        report_unsuccessful_runs : bool
            Whether to print reporting statements in out stream.
        """
        self.target_cell_runparams = {}
        self.target_cell_evals = {}
        self.target_cell_indices = {}
        target_cell_table = []
        for gs_id in self.gs_ids:
            # Collect runs that belong to grid search by looping over file names in directory.
            indir = self.source_path + gs_id + "/results/"
            # runs_ids are the unique hyper-parameter settings, which are again subsetted by cross-validation.
            # These ids are present in all files names but are only collected from the *model.tf file names here.

            run_ids = np.sort(
                np.unique(
                    [
                        "_".join(".".join(x.split(".")[:-1]).split("_")[:-1])
                        for x in os.listdir(indir)
                        if x.split("_")[-1].split(".")[0] == "time"
                    ]
                )
            )
            cv_ids = np.sort(np.unique([x.split("_")[-1] for x in run_ids]))  # identifiers of cross-validation splits
            run_ids = np.sort(
                np.unique(["_".join(x.split("_")[:-1]) for x in run_ids])  # identifiers of hyper-parameters settings
            )
            run_ids_clean = []  # only IDs of completed runs (all files present)
            for r in run_ids:
                complete_run = True
                for cv in cv_ids:
                    fn = r + "_" + cv + "_ntevaluation.pickle"
                    if not os.path.isfile(indir + fn):
                        if report_unsuccessful_runs:
                            print("File %r missing" % fn)
                        complete_run = False
                # Check run parameter files (one per cross-validation set):
                fn = r + "_runparams.pickle"
                if not os.path.isfile(indir + fn):
                    if report_unsuccessful_runs:
                        print("File %r missing" % fn)
                    complete_run = False
                if not complete_run:
                    if report_unsuccessful_runs:
                        print("Run %r not successful" % r + "_" + cv)
                else:
                    run_ids_clean.append(r)
            # Load results and settings from completed runs:
            evals = (
                {}
            )  # Dictionary over runs with dictionary over cross-validations with results from model evaluation.
            indices = {}
            runparams = {}  # Dictionary over runs with model settings.
            for x in run_ids_clean:
                # Load model settings (these are shared across all partitions).
                fn_runparams = indir + x + "_runparams.pickle"
                with open(fn_runparams, "rb") as f:
                    runparams[x] = pickle.load(f)
                evals[x] = {}
                for cv in cv_ids:
                    fn_eval = indir + x + "_" + cv + "_ntevaluation.pickle"
                    with open(fn_eval, "rb") as f:
                        evals[x][cv] = pickle.load(f)
                indices[x] = {}
                for cv in cv_ids:
                    fn_eval = indir + x + "_" + cv + "_ntindices.pickle"
                    with open(fn_eval, "rb") as f:
                        indices[x][cv] = pickle.load(f)
            self.target_cell_runparams[gs_id] = runparams
            self.target_cell_evals[gs_id] = evals
            self.target_cell_indices[gs_id] = indices

            for x in run_ids_clean:
                for cv in cv_ids:
                    for target_cell in evals[x][cv].keys():
                        if evals[x][cv][target_cell]:
                            tc_frequencies = {
                                k: len(indices[x][cv][target_cell]["nodes_idx"][k])
                                for k in indices[x][cv][target_cell]["nodes_idx"].keys()
                            }

                            target_cell_table.append(
                                pd.concat(
                                    [
                                        pd.DataFrame(
                                            dict(
                                                list(
                                                    {
                                                        "model_class": [runparams[x]["model_class"]],
                                                        "cond_type": [runparams[x]["cond_type"]]
                                                        if "cond_type" in list(runparams[x].keys())
                                                        else "none",
                                                        "gs_id": [runparams[x]["gs_id"]],
                                                        "model_id": [runparams[x]["model_id"]],
                                                        #"split_mode": [runparams[x]["split_mode"]],
                                                        "radius": [runparams[x]["radius"]] if "radius" in list(runparams[x].keys()) else [runparams[x]["max_dist"]],
                                                        "n_rings": [runparams[x]["n_rings"]] if "n_rings" in list(runparams[x].keys()) else "none",
                                                        "graph_covar_selection": [
                                                            runparams[x]["graph_covar_selection"]
                                                        ],
                                                        #"node_label_space_id": [
                                                        #    runparams[x]["node_label_space_id"]
                                                        #],
                                                        #"node_feature_space_id": [
                                                        #    runparams[x]["node_feature_space_id"]
                                                        #],
                                                        #"feature_transformation": [
                                                        #    runparams[x]["feature_transformation"]
                                                        #],
                                                        "use_covar_node_position": [
                                                            runparams[x]["use_covar_node_position"]
                                                        ],
                                                        "use_covar_node_label": [runparams[x]["use_covar_node_label"]],
                                                        "use_covar_graph_covar": [
                                                            runparams[x]["use_covar_graph_covar"]
                                                        ],
                                                        # "hold_out_covariate": [runparams[x]['hold_out_covariate']],
                                                        "optimizer": [runparams[x]["optimizer"]],
                                                        "learning_rate": [runparams[x]["learning_rate"]],
                                                        #"intermediate_dim_enc": [runparams[x]["intermediate_dim_enc"]],
                                                        #"intermediate_dim_dec": [runparams[x]["intermediate_dim_dec"]],
                                                        #"latent_dim": [runparams[x]["latent_dim"]],
                                                        # "depth_enc": [runparams[x]['depth_enc']],
                                                        # "depth_dec": [runparams[x]['depth_dec']],
                                                        #"dropout_rate": [runparams[x]["dropout_rate"]],
                                                        "l2_coef": [runparams[x]["l2_coef"]],
                                                        "l1_coef": [runparams[x]["l1_coef"]],
                                                        "use_domain": [runparams[x]["use_domain"]],
                                                        "domain_type": [runparams[x]["domain_type"]],
                                                        #"use_batch_norm": [runparams[x]["use_batch_norm"]],
                                                        "scale_node_size": [runparams[x]["scale_node_size"]],
                                                        #"transform_input": [runparams[x]["transform_input"]],
                                                        "output_layer": [runparams[x]["output_layer"]],
                                                        "log_transform": [runparams[x]["log_transform"]]
                                                        if "log_transform" in list(runparams[x].keys())
                                                        else False,
                                                        "epochs": [runparams[x]["epochs"]],
                                                        "batch_size": [runparams[x]["batch_size"]],
                                                        "run_id": x,
                                                        "cv": cv,
                                                        "target_cell_frequencies": sum(tc_frequencies.values()),
                                                        "model": "_".join(gs_id.split("/")[-1].split("_")[1:-1]),
                                                        "target_cell": target_cell,
                                                        "loss": [evals[x][cv][target_cell]["loss"]]
                                                        if "loss" in list(evals[x][cv][target_cell].keys())
                                                        else np.nan,
                                                        "custom_mean_sd": [evals[x][cv][target_cell]["custom_mean_sd"]]
                                                        if "custom_mean_sd" in list(evals[x][cv][target_cell].keys())
                                                        else np.nan,
                                                        "custom_mse": [evals[x][cv][target_cell]["custom_mse"]]
                                                        if "custom_mse" in list(evals[x][cv][target_cell].keys())
                                                        else np.nan,
                                                        "custom_mse_scaled": [
                                                            evals[x][cv][target_cell]["custom_mse_scaled"]
                                                        ]
                                                        if "custom_mse_scaled" in list(evals[x][cv][target_cell].keys())
                                                        else np.nan,
                                                        "gaussian_reconstruction_loss": [
                                                            evals[x][cv][target_cell]["gaussian_reconstruction_loss"]
                                                        ]
                                                        if "gaussian_reconstruction_loss"
                                                        in list(evals[x][cv][target_cell].keys())
                                                        else np.nan,
                                                        "r_squared": [evals[x][cv][target_cell]["r_squared"]]
                                                        if "r_squared" in list(evals[x][cv][target_cell].keys())
                                                        else np.nan,
                                                        "r_squared_linreg": [
                                                            evals[x][cv][target_cell]["r_squared_linreg"]
                                                        ]
                                                        if "r_squared_linreg" in list(evals[x][cv][target_cell].keys())
                                                        else np.nan,
                                                    }.items()
                                                )
                                            )
                                        )
                                    ]
                                )
                            )
        self.target_cell_table = pd.concat(target_cell_table)
        self.target_cell_table["um_radius"] = (self.target_cell_table["radius"] * self.lateral_resolution).astype(int)

    def select_cv(self, cv_idx: int) -> str:
        """Return key of of cross-validation selected with numeric index.

        Parameters
        ----------
        cv_idx : int
            Index of cross-validation to plot confusion matrix for.

        Returns
        -------
        cv

        Raises
        ------
        ValueError
            `cv_idx` out of scope of cross-validation set
        """
        if cv_idx >= len(self.cv_keys):
            raise ValueError("cv_idx %i out of scope of cross-validation set: %s" % (cv_idx, self.cv_keys))
        else:
            cv = self.cv_keys[cv_idx]
        print("cross-validation selected: %s" % cv)
        return cv

    def get_best_model_id(
        self,
        subset_hyperparameters: Optional[List[Tuple[str, str]]] = None,
        metric_select: str = "r_squared_linreg",
        partition_select: str = "test",
        cv_mode: str = "mean",
    ):
        """Get best model identifier.

        Parameters
        ----------
        subset_hyperparameters : list, optional
            List of subset hyperparameters.
        metric_select : str
            Selected metric.
        partition_select : str
            Selected parition.
        cv_mode : str
            cross validation mode.

        Returns
        -------
        best_model_id

        Raises
        ------
        ValueError
            If measure, partition or cv_mode not recognized.
        Warning
            If cv_mode max is selected with the following metrics: loss, elbo, mse, mae
        """
        if subset_hyperparameters is None:
            subset_hyperparameters = []
        if metric_select.endswith("loss"):
            ascending = True
            if cv_mode == "max":
                raise Warning("selected cv_mode max with metric_id loss, likely not intended")
        elif metric_select.endswith("elbo"):
            ascending = True
            if cv_mode == "max":
                raise Warning("selected cv_mode max with metric_id elbo, likely not intended")
        elif metric_select.endswith("mse") or metric_select.endswith("mse_scaled"):
            ascending = True
            if cv_mode == "max":
                raise Warning("selected cv_mode max with metric_id mse, likely not intended")
        elif metric_select.endswith("mae"):
            ascending = True
            if cv_mode == "max":
                raise Warning("selected cv_mode max with metric_id mae, likely not intended")
        elif metric_select.endswith("r_squared") or metric_select.endswith("r_squared_linreg"):
            ascending = False
        else:
            raise ValueError("measure %s not recognized" % metric_select)
        if partition_select not in ["test", "val", "train", "all"]:
            raise ValueError("partition %s not recognised" % partition_select)
        metric_select = partition_select + "_" + metric_select
        summary_table = self.summary_table
        for x, y in subset_hyperparameters:
            if not isinstance(y, list):
                y = [y]
            if not np.any([xx in y for xx in summary_table[x].values]):
                print(
                    "subset was empty, available values for %s are %s, given was %s"
                    % (x, str(np.unique(summary_table[x].values).tolist()), str(y))
                )
            summary_table = summary_table.loc[[xx in y for xx in summary_table[x].values], :]
        if cv_mode.lower() == "mean":
            best_model = (
                summary_table.groupby("run_id", as_index=False)[metric_select]
                .mean()
                .sort_values([metric_select], ascending=ascending)
            )
        elif cv_mode.lower() == "median":
            best_model = (
                summary_table.groupby("run_id", as_index=False)[metric_select]
                .median()
                .sort_values([metric_select], ascending=ascending)
            )
        elif cv_mode.lower() == "max":
            best_model = (
                summary_table.groupby("run_id", as_index=False)[metric_select]
                .max()
                .sort_values([metric_select], ascending=ascending)
            )
        elif cv_mode.lower() == "min":
            best_model = (
                summary_table.groupby("run_id", as_index=False)[metric_select]
                .min()
                .sort_values([metric_select], ascending=ascending)
            )
        else:
            raise ValueError("cv_mode %s not recognized" % cv_mode)
        return best_model["run_id"].values[0] if best_model.shape[0] > 0 else None

    def copy_best_model(
        self,
        gs_id: str,
        dst: str = "best",
        metric_select: str = "loss",
        partition_select: str = "val",
        cv_mode: str = "mean",
    ):
        """Copy best model.

        Parameters
        ----------
        gs_id : str
            Grid search identifier.
        dst : str
            dst folder.
        metric_select : str
            Selected metric.
        partition_select : str
            Selected partition.
        cv_mode : str
            Cross validation mode.
        """
        from shutil import copyfile

        if dst[0] != "/":
            dst = self.source_path + gs_id + "/" + dst + "/"
        run_id = self.get_best_model_id(metric_select=metric_select, partition_select=partition_select, cv_mode=cv_mode)
        cvs = self.summary_table.loc[self.summary_table["run_id"].values == run_id, "cv"].values
        src = self.source_path + gs_id + "/results/"
        print("copying model files from %s to %s" % (src, dst))
        for cv in cvs:
            fn_model = run_id + "_" + cv + "_model.h5"
            copyfile(src + fn_model, dst + fn_model)
            fn_idx = run_id + "_" + cv + "_indices.pickle"
            copyfile(src + fn_idx, dst + fn_idx)

    def get_info(self, model_id, expected_pickle: Optional[list] = None):
        """Get information of model.

        Parameters
        ----------
        model_id
            Model identifier.
        expected_pickle : list, optional
            Expected pickle files.

        Raises
        ------
        ValueError
            If file is missing.
        """
        if expected_pickle is None:
            expected_pickle = ["evaluation", "history", "hyperparam"]

        indir = self.source_path + self.source_gs[model_id] + "/results/"
        # Check that all files are present:
        cv_ids = np.sort(
            np.unique([x.split("_")[-2] for x in os.listdir(indir) if x.split("_")[-1].split(".")[0] == "time"])
        )
        for cv in cv_ids:
            # Check pickled files:
            for suffix in expected_pickle:
                fn = model_id + "_" + cv + "_" + suffix + ".pickle"
                if not os.path.isfile(indir + fn):
                    raise ValueError("file %s missing" % suffix)
        info = {}
        for cv in cv_ids:
            info[cv] = {}
            for suffix in expected_pickle:
                fn = model_id + "_" + cv + "_" + suffix + ".pickle"
                with open(indir + fn, "rb") as f:
                    info[cv][suffix] = pickle.load(f)
        fn = model_id + "_runparams.pickle"
        with open(indir + fn, "rb") as f:
            info["runparams"] = pickle.load(f)
        self.info = info

    # Plotting functions:
    def plot_best_model_by_hyperparam(
        self,
        graph_model_class: str,
        baseline_model_class: str,
        partition_show: str = 'test',
        metric_show: str = 'r_squared_linreg',
        partition_select: str = 'val',
        metric_select: str = 'r_squared_linreg',
        param_x: str = 'um_radius',
        param_hue: str = 'model',
        rename_levels: Optional[List[Tuple[str, str]]] = None,
        subset_hyperparameters: Optional[List[Tuple[str, str]]] = None,
        cv_mode: str = "mean",
        yaxis_limit: Optional[Tuple[float, float]] = None,
        xticks: Optional[List[int]] = None,
        rotate_xticks: bool = True,
        figsize: Tuple[float, float] = (3.5, 4.0),
        fontsize: Optional[int] = None,
        example_cellradius: Optional[int] = 10,
        plot_mode: str = "lineplot",
        palette: Optional[dict] = {"baseline": "C0", "NCEM": "C1"},
        color: Optional[str] = None,
        save: Optional[str] = None,
        suffix: str = "best_by_hyperparam.pdf",
        show: bool = True,
        return_axs: bool = False,
    ):
        """Plot best model by hyperparameter.

        Parameters
        ----------
        graph_model_class : str
            Graph model class.
        baseline_model_class : str
            Baseline model class.
        partition_show : str
            Showing partition.
        metric_show : str
            Showing metric.
        partition_select : str
            Selected partition.
        metric_select : str
            Selected metric.
        param_x : str
            Parameter on x axis.
        param_hue : str
            Parameter for hue.
        rename_levels : list, optional
            Rename levels with stated logic.
        subset_hyperparameters : list, optional
            Subset hyperparameters.
        cv_mode : str
            Cross validation mode.
        yaxis_limit : tuple, optional
            y axis limits.
        xticks : list, optional
            List of x ticks.
        rotate_xticks : bool
            Whether to rotate x ticks.
        figsize : tuple
            Figure size.
        fontsize : int, optional
            Font size.
        plot_mode : str
            Plotting mode, can be `boxplot`, `lineplot` or `mean_lineplot`.
        palette : dict, optional
            Palette.
        color : str, optional
            Color.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        show : bool
            Whether to display plot.
        return_axs : bool
            Whether to return axis objects.

        Returns
        -------
        axis if `return_axs` is True.
        """
        if fontsize:
            sc.set_figure_params(scanpy=True, fontsize=fontsize)
        sns.set_palette("colorblind")
        rename_levels = [] if not rename_levels else rename_levels
        subset_hyperparameters = [] if not subset_hyperparameters else subset_hyperparameters

        plt.ioff()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        if isinstance(param_hue, list):
            param_hue_new = param_hue[0] + "_" + param_hue[1]
            self.summary_table[param_hue_new] = [
                str(self.summary_table[param_hue[0]].values[i]) + "_" + str(self.summary_table[param_hue[1]].values[i])
                for i in range(self.summary_table.shape[0])
            ]
            param_hue = param_hue_new
        params_x_unique = np.sort(np.unique(self.summary_table[param_x].values))
        params_hue_unique = np.sort(np.unique(self.summary_table[param_hue].values))

        run_ids = []
        summary_table = self.summary_table.copy()
        runparams_table = self.runparams_table.copy()
        for x in params_x_unique:
            for hue in params_hue_unique:
                run_id_temp = self.get_best_model_id(
                    subset_hyperparameters=[(param_x, x), (param_hue, hue)] + subset_hyperparameters,
                    partition_select=partition_select,
                    metric_select=metric_select,
                    cv_mode=cv_mode,
                )
                print(run_id_temp)
                if run_id_temp is not None:
                    run_ids.append(run_id_temp)

        summary_table = summary_table.loc[np.array([x in run_ids for x in summary_table["run_id"].values]), :]
        runparams_table = runparams_table.loc[
            np.array([x in run_ids for x in runparams_table["run_id_params"].values]), :
        ]
        for level, d in rename_levels:
            summary_table[level] = [d[x] for x in summary_table[level].values]
        self.best_model_hyperparam_table = runparams_table
        summary_table.sort_values([param_x, param_hue])
        ycol = partition_show + "_" + metric_show

        if plot_mode == "boxplot":
            if color:
                sns.boxplot(x=param_x, y=ycol, data=summary_table, ax=ax, color=color)
                sns.swarmplot(
                    x=param_x,
                    y=ycol,
                    palette=[color],
                    data=summary_table,
                    ax=ax,
                )
            else:
                sns.boxplot(
                    x=param_x,
                    y=ycol,
                    hue=param_hue,
                    data=summary_table,
                    ax=ax,
                )
                sns.swarmplot(x=param_x, y=ycol, hue=param_hue, data=summary_table, ax=ax)
        elif plot_mode == "lineplot":
            temp_baseline = summary_table[summary_table["model_class"] == baseline_model_class].reset_index()
            sns.scatterplot(
                x=param_x,
                y=ycol,
                hue=param_hue,
                style="cv",
                palette=palette,
                data=temp_baseline,
                s=100,
                ax=ax,
            )
            temp_graph = summary_table[summary_table["model_class"] == graph_model_class].reset_index()
            sns.lineplot(
                x=param_x,
                y=ycol,
                style="cv",
                palette=palette,
                data=temp_graph,
                hue=param_hue,
                ax=ax,
                markers=True,
            )
            ax.set_xscale("symlog", linthresh=10)
            if example_cellradius:
                plt.axvline(example_cellradius, color='limegreen', linewidth=3.)
        elif plot_mode == "mean_lineplot":
            temp_summary_table = summary_table[[param_hue, param_x, ycol]].groupby([param_hue, param_x]).mean()
            sns.lineplot(
                x=param_x,
                y=ycol,
                palette=["grey"],
                style=param_hue,
                data=temp_summary_table,
                hue=param_hue,
                ax=ax,
                sort=False,
                markers=True,
            )
            ax.set_xscale("symlog")

        ax.grid(False)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        lgd = ax.legend(loc="center left", bbox_to_anchor=(1.5, 0.5))
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        if yaxis_limit is not None:
            ax.set_ylim(yaxis_limit)
        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%0.3f"))
        if rotate_xticks:
            plt.xticks(rotation=90)
        #ax.yaxis.tick_right()
        #plt.yticks(rotation=180)

        # Save, show and return figure.
        # plt.tight_layout()
        if save is not None:
            plt.savefig(save + "_" + partition_show + suffix, bbox_extra_artists=(lgd,), bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None

    def plot_target_cell_evaluation(
        self,
        metric_show: str,
        metric_select: str,
        param_x: str,
        ncols: int = 8,
        show: bool = True,
        save: Optional[str] = None,
        suffix: str = "target_cell_evaluation.pdf",
        return_axs: bool = False,
        yaxis_limit: Optional[Tuple[float, float]] = None,
        panelsize: Tuple[float, float] = (3.0, 3.0),
        sharey: bool = False,
    ):
        """Plot target cell evaluation.

        Parameters
        ----------
        metric_show : str
            Showing metric.
        metric_select : str
            Selected metric.
        param_x : str
            Parameter on x axis.
        yaxis_limit : tuple, optional
            y axis limits.
        panelsize : tuple
            Panel size.
        ncols : int
            Number of columns.
        save : str, optional
            Whether (if not None) and where (path as string given as save) to save plot.
        suffix : str
            Suffix of file name to save to.
        show : bool
            Whether to display plot.
        return_axs : bool
            Whether to return axis objects.

        Returns
        -------
        axis if `return_axs` is True.
        """
        params_x_unique = np.sort(np.unique(self.target_cell_table[param_x].values))
        params_tc_unique = np.sort(np.unique(self.target_cell_table["target_cell"].values))

        ct = len(params_tc_unique)
        nrows = len(params_tc_unique) // ncols + int(len(params_tc_unique) % ncols > 0)

        fig, ax = plt.subplots(
            ncols=ncols, nrows=nrows, figsize=(ncols * panelsize[0], nrows * panelsize[1]), sharex=True, sharey=sharey
        )
        ax = ax.flat
        for a in ax[ct:]:
            a.remove()
        ax = ax[:ct]
        ax = ax.ravel()
        target_cell_frequencies = {}
        for i, tc in enumerate(params_tc_unique):
            run_ids = []
            for x in params_x_unique:

                target_cell_table = self.target_cell_table[self.target_cell_table["target_cell"] == tc]
                subset_hyperparameters = [(param_x, x)]
                for a, b in subset_hyperparameters:
                    if not isinstance(b, list):
                        b = [b]
                    if not np.any([xx in b for xx in target_cell_table[a].values]):
                        print(
                            "subset was empty, available values for %s are %s, given was %s"
                            % (a, str(np.unique(target_cell_table[a].values).tolist()), str(b))
                        )
                    summary_table = target_cell_table.loc[[xx in b for xx in target_cell_table[a].values], :]

                best_model = (
                    summary_table.groupby("run_id", as_index=False)[metric_select]
                    .mean()
                    .sort_values([metric_select], ascending=False, na_position="last")
                )
                run_id_temp = best_model["run_id"].values[0]

                if run_id_temp is not None:
                    run_ids.append(run_id_temp)

            table = target_cell_table.loc[np.array([x in run_ids for x in target_cell_table["run_id"].values]), :]
            tc_frequencies = int(np.unique(table["target_cell_frequencies"])[0])
            target_cell_frequencies.update({tc: tc_frequencies})
            sns.boxplot(x=param_x, y=metric_show, data=table, ax=ax[i], color="steelblue")
            sns.swarmplot(x=param_x, y=metric_show, data=table, ax=ax[i], color="steelblue")
            ax[i].set_title(f"{tc.replace('_', ' ')} \n({str(tc_frequencies)} cells)")
            ax[i].set_xlabel("")
            ax[i].set_ylabel("")
            if yaxis_limit is not None:
                ax[i].set_ylim(yaxis_limit)
            ax[i].yaxis.set_major_formatter(FormatStrFormatter("%0.3f"))
            for tick in ax[i].get_xticklabels():
                tick.set_rotation(90)

        self.target_cell_frequencies = target_cell_frequencies
        plt.tight_layout()
        # Save, show and return figure.
        if save is not None:
            plt.savefig(save + "_" + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None
