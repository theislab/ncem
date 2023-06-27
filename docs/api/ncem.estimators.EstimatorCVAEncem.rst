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

:github_url: ncem.estimators.EstimatorCVAEncem

ncem.estimators.EstimatorCVAEncem
=================================

.. currentmodule:: ncem.estimators

.. autoclass:: EstimatorCVAEncem



   .. rubric:: Attributes

   .. autosummary::
      :toctree: .

      ~ncem.estimators.EstimatorCVAEncem.img_keys_all
      ~ncem.estimators.EstimatorCVAEncem.nodes_idx_all
      ~ncem.estimators.EstimatorCVAEncem.nodes_idx_eval
      ~ncem.estimators.EstimatorCVAEncem.nodes_idx_test
      ~ncem.estimators.EstimatorCVAEncem.nodes_idx_train
      ~ncem.estimators.EstimatorCVAEncem.patient_ids_bytarget
      ~ncem.estimators.EstimatorCVAEncem.patient_ids_unique
      ~ncem.estimators.EstimatorCVAEncem.img_to_patient_dict
      ~ncem.estimators.EstimatorCVAEncem.complete_img_keys
      ~ncem.estimators.EstimatorCVAEncem.a
      ~ncem.estimators.EstimatorCVAEncem.h_0
      ~ncem.estimators.EstimatorCVAEncem.h_1
      ~ncem.estimators.EstimatorCVAEncem.size_factors
      ~ncem.estimators.EstimatorCVAEncem.graph_covar
      ~ncem.estimators.EstimatorCVAEncem.node_covar
      ~ncem.estimators.EstimatorCVAEncem.domains
      ~ncem.estimators.EstimatorCVAEncem.covar_selection
      ~ncem.estimators.EstimatorCVAEncem.node_types
      ~ncem.estimators.EstimatorCVAEncem.node_type_names
      ~ncem.estimators.EstimatorCVAEncem.graph_covar_names
      ~ncem.estimators.EstimatorCVAEncem.node_feature_names
      ~ncem.estimators.EstimatorCVAEncem.n_features_type
      ~ncem.estimators.EstimatorCVAEncem.n_features_standard
      ~ncem.estimators.EstimatorCVAEncem.n_features_0
      ~ncem.estimators.EstimatorCVAEncem.n_features_1
      ~ncem.estimators.EstimatorCVAEncem.n_graph_covariates
      ~ncem.estimators.EstimatorCVAEncem.n_node_covariates
      ~ncem.estimators.EstimatorCVAEncem.n_domains
      ~ncem.estimators.EstimatorCVAEncem.max_nodes
      ~ncem.estimators.EstimatorCVAEncem.n_eval_nodes_per_graph
      ~ncem.estimators.EstimatorCVAEncem.vi_model
      ~ncem.estimators.EstimatorCVAEncem.log_transform
      ~ncem.estimators.EstimatorCVAEncem.model_type
      ~ncem.estimators.EstimatorCVAEncem.adj_type
      ~ncem.estimators.EstimatorCVAEncem.cond_type
      ~ncem.estimators.EstimatorCVAEncem.cond_depth
      ~ncem.estimators.EstimatorCVAEncem.output_layer
      ~ncem.estimators.EstimatorCVAEncem.steps_per_epoch
      ~ncem.estimators.EstimatorCVAEncem.validation_steps





   .. rubric:: Methods

   .. autosummary::
      :toctree: .

      ~ncem.estimators.EstimatorCVAEncem.evaluate_any
      ~ncem.estimators.EstimatorCVAEncem.evaluate_any_posterior_sampling
      ~ncem.estimators.EstimatorCVAEncem.evaluate_per_node_type
      ~ncem.estimators.EstimatorCVAEncem.get_data
      ~ncem.estimators.EstimatorCVAEncem.init_model
      ~ncem.estimators.EstimatorCVAEncem.predict
      ~ncem.estimators.EstimatorCVAEncem.pretrain_decoder
      ~ncem.estimators.EstimatorCVAEncem.split_data_given
      ~ncem.estimators.EstimatorCVAEncem.split_data_node
      ~ncem.estimators.EstimatorCVAEncem.split_data_target_cell
      ~ncem.estimators.EstimatorCVAEncem.train
      ~ncem.estimators.EstimatorCVAEncem.train_aggressive
      ~ncem.estimators.EstimatorCVAEncem.train_normal



   .. _sphx_glr_backref_ncem.estimators.EstimatorCVAEncem:

   .. minigallery:: ncem.estimators.EstimatorCVAEncem
       :add-heading: Examples
       :heading-level: -
