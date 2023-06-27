..
    This code was adapted from https://github.com/theislab/cellrank/tree/master/docs/_templates/autosummary/class.rst
    This file is therefore licensed under the license of the cellrank project,
    available from https://github.com/theislab/cellrank and copied here at the time of accession.

    BSD 3-Clause License

    Copyright (c) 2019, Theis Lab
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

:github_url: ncem.estimators.Estimator

ncem.estimators.Estimator
=========================

.. currentmodule:: ncem.estimators

.. autoclass:: Estimator



   .. rubric:: Attributes

   .. autosummary::
      :toctree: .

      ~ncem.estimators.Estimator.img_keys_all
      ~ncem.estimators.Estimator.nodes_idx_all
      ~ncem.estimators.Estimator.nodes_idx_eval
      ~ncem.estimators.Estimator.nodes_idx_test
      ~ncem.estimators.Estimator.nodes_idx_train
      ~ncem.estimators.Estimator.patient_ids_bytarget
      ~ncem.estimators.Estimator.patient_ids_unique
      ~ncem.estimators.Estimator.img_to_patient_dict
      ~ncem.estimators.Estimator.complete_img_keys
      ~ncem.estimators.Estimator.a
      ~ncem.estimators.Estimator.h_0
      ~ncem.estimators.Estimator.h_1
      ~ncem.estimators.Estimator.size_factors
      ~ncem.estimators.Estimator.graph_covar
      ~ncem.estimators.Estimator.node_covar
      ~ncem.estimators.Estimator.domains
      ~ncem.estimators.Estimator.covar_selection
      ~ncem.estimators.Estimator.node_types
      ~ncem.estimators.Estimator.node_type_names
      ~ncem.estimators.Estimator.graph_covar_names
      ~ncem.estimators.Estimator.node_feature_names
      ~ncem.estimators.Estimator.n_features_type
      ~ncem.estimators.Estimator.n_features_standard
      ~ncem.estimators.Estimator.n_features_0
      ~ncem.estimators.Estimator.n_features_1
      ~ncem.estimators.Estimator.n_graph_covariates
      ~ncem.estimators.Estimator.n_node_covariates
      ~ncem.estimators.Estimator.n_domains
      ~ncem.estimators.Estimator.max_nodes
      ~ncem.estimators.Estimator.n_eval_nodes_per_graph
      ~ncem.estimators.Estimator.vi_model
      ~ncem.estimators.Estimator.log_transform
      ~ncem.estimators.Estimator.model_type
      ~ncem.estimators.Estimator.adj_type
      ~ncem.estimators.Estimator.cond_type
      ~ncem.estimators.Estimator.cond_depth
      ~ncem.estimators.Estimator.output_layer
      ~ncem.estimators.Estimator.steps_per_epoch
      ~ncem.estimators.Estimator.validation_steps





   .. rubric:: Methods

   .. autosummary::
      :toctree: .

      ~ncem.estimators.Estimator.evaluate_any
      ~ncem.estimators.Estimator.evaluate_per_node_type
      ~ncem.estimators.Estimator.get_data
      ~ncem.estimators.Estimator.init_model
      ~ncem.estimators.Estimator.predict
      ~ncem.estimators.Estimator.pretrain_decoder
      ~ncem.estimators.Estimator.split_data_given
      ~ncem.estimators.Estimator.split_data_node
      ~ncem.estimators.Estimator.split_data_target_cell
      ~ncem.estimators.Estimator.train
      ~ncem.estimators.Estimator.train_aggressive
      ~ncem.estimators.Estimator.train_normal



   .. _sphx_glr_backref_ncem.estimators.Estimator:

   .. minigallery:: ncem.estimators.Estimator
       :add-heading: Examples
       :heading-level: -
