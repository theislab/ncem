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

:github_url: ncem.estimators.EstimatorInteractions

ncem.estimators.EstimatorInteractions
=====================================

.. currentmodule:: ncem.estimators

.. autoclass:: EstimatorInteractions

   
   
   .. rubric:: Attributes

   .. autosummary::
      :toctree: .
   
      ~ncem.estimators.EstimatorInteractions.img_keys_all
      ~ncem.estimators.EstimatorInteractions.nodes_idx_all
      ~ncem.estimators.EstimatorInteractions.nodes_idx_eval
      ~ncem.estimators.EstimatorInteractions.nodes_idx_test
      ~ncem.estimators.EstimatorInteractions.nodes_idx_train
      ~ncem.estimators.EstimatorInteractions.patient_ids_bytarget
      ~ncem.estimators.EstimatorInteractions.patient_ids_unique
      ~ncem.estimators.EstimatorInteractions.img_to_patient_dict
      ~ncem.estimators.EstimatorInteractions.complete_img_keys
      ~ncem.estimators.EstimatorInteractions.a
      ~ncem.estimators.EstimatorInteractions.h_0
      ~ncem.estimators.EstimatorInteractions.h_1
      ~ncem.estimators.EstimatorInteractions.size_factors
      ~ncem.estimators.EstimatorInteractions.graph_covar
      ~ncem.estimators.EstimatorInteractions.node_covar
      ~ncem.estimators.EstimatorInteractions.domains
      ~ncem.estimators.EstimatorInteractions.covar_selection
      ~ncem.estimators.EstimatorInteractions.node_types
      ~ncem.estimators.EstimatorInteractions.node_type_names
      ~ncem.estimators.EstimatorInteractions.graph_covar_names
      ~ncem.estimators.EstimatorInteractions.node_feature_names
      ~ncem.estimators.EstimatorInteractions.n_features_type
      ~ncem.estimators.EstimatorInteractions.n_features_standard
      ~ncem.estimators.EstimatorInteractions.n_features_0
      ~ncem.estimators.EstimatorInteractions.n_features_1
      ~ncem.estimators.EstimatorInteractions.n_graph_covariates
      ~ncem.estimators.EstimatorInteractions.n_node_covariates
      ~ncem.estimators.EstimatorInteractions.n_domains
      ~ncem.estimators.EstimatorInteractions.max_nodes
      ~ncem.estimators.EstimatorInteractions.n_eval_nodes_per_graph
      ~ncem.estimators.EstimatorInteractions.vi_model
      ~ncem.estimators.EstimatorInteractions.log_transform
      ~ncem.estimators.EstimatorInteractions.model_type
      ~ncem.estimators.EstimatorInteractions.adj_type
      ~ncem.estimators.EstimatorInteractions.cond_type
      ~ncem.estimators.EstimatorInteractions.cond_depth
      ~ncem.estimators.EstimatorInteractions.output_layer
      ~ncem.estimators.EstimatorInteractions.steps_per_epoch
      ~ncem.estimators.EstimatorInteractions.validation_steps
   
   

   
   
   .. rubric:: Methods

   .. autosummary::
      :toctree: .
   
      ~ncem.estimators.EstimatorInteractions.evaluate_any
      ~ncem.estimators.EstimatorInteractions.evaluate_per_node_type
      ~ncem.estimators.EstimatorInteractions.get_data
      ~ncem.estimators.EstimatorInteractions.init_model
      ~ncem.estimators.EstimatorInteractions.predict
      ~ncem.estimators.EstimatorInteractions.pretrain_decoder
      ~ncem.estimators.EstimatorInteractions.split_data_given
      ~ncem.estimators.EstimatorInteractions.split_data_node
      ~ncem.estimators.EstimatorInteractions.split_data_target_cell
      ~ncem.estimators.EstimatorInteractions.train
      ~ncem.estimators.EstimatorInteractions.train_aggressive
      ~ncem.estimators.EstimatorInteractions.train_normal
   
   

   .. _sphx_glr_backref_ncem.estimators.EstimatorInteractions:

   .. minigallery:: ncem.estimators.EstimatorInteractions
       :add-heading: Examples
       :heading-level: -