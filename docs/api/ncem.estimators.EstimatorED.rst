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

:github_url: ncem.estimators.EstimatorED

ncem.estimators.EstimatorED
===========================

.. currentmodule:: ncem.estimators

.. autoclass:: EstimatorED

   
   
   .. rubric:: Attributes

   .. autosummary::
      :toctree: .
   
      ~ncem.estimators.EstimatorED.img_keys_all
      ~ncem.estimators.EstimatorED.nodes_idx_all
      ~ncem.estimators.EstimatorED.nodes_idx_eval
      ~ncem.estimators.EstimatorED.nodes_idx_test
      ~ncem.estimators.EstimatorED.nodes_idx_train
      ~ncem.estimators.EstimatorED.patient_ids_bytarget
      ~ncem.estimators.EstimatorED.patient_ids_unique
      ~ncem.estimators.EstimatorED.img_to_patient_dict
      ~ncem.estimators.EstimatorED.complete_img_keys
      ~ncem.estimators.EstimatorED.a
      ~ncem.estimators.EstimatorED.h_0
      ~ncem.estimators.EstimatorED.h_1
      ~ncem.estimators.EstimatorED.size_factors
      ~ncem.estimators.EstimatorED.graph_covar
      ~ncem.estimators.EstimatorED.node_covar
      ~ncem.estimators.EstimatorED.domains
      ~ncem.estimators.EstimatorED.covar_selection
      ~ncem.estimators.EstimatorED.node_types
      ~ncem.estimators.EstimatorED.node_type_names
      ~ncem.estimators.EstimatorED.graph_covar_names
      ~ncem.estimators.EstimatorED.node_feature_names
      ~ncem.estimators.EstimatorED.n_features_type
      ~ncem.estimators.EstimatorED.n_features_standard
      ~ncem.estimators.EstimatorED.n_features_0
      ~ncem.estimators.EstimatorED.n_features_1
      ~ncem.estimators.EstimatorED.n_graph_covariates
      ~ncem.estimators.EstimatorED.n_node_covariates
      ~ncem.estimators.EstimatorED.n_domains
      ~ncem.estimators.EstimatorED.max_nodes
      ~ncem.estimators.EstimatorED.n_eval_nodes_per_graph
      ~ncem.estimators.EstimatorED.vi_model
      ~ncem.estimators.EstimatorED.log_transform
      ~ncem.estimators.EstimatorED.model_type
      ~ncem.estimators.EstimatorED.adj_type
      ~ncem.estimators.EstimatorED.cond_type
      ~ncem.estimators.EstimatorED.cond_depth
      ~ncem.estimators.EstimatorED.output_layer
      ~ncem.estimators.EstimatorED.steps_per_epoch
      ~ncem.estimators.EstimatorED.validation_steps
   
   

   
   
   .. rubric:: Methods

   .. autosummary::
      :toctree: .
   
      ~ncem.estimators.EstimatorED.evaluate_any
      ~ncem.estimators.EstimatorED.evaluate_per_node_type
      ~ncem.estimators.EstimatorED.get_data
      ~ncem.estimators.EstimatorED.init_model
      ~ncem.estimators.EstimatorED.predict
      ~ncem.estimators.EstimatorED.pretrain_decoder
      ~ncem.estimators.EstimatorED.split_data_given
      ~ncem.estimators.EstimatorED.split_data_node
      ~ncem.estimators.EstimatorED.split_data_target_cell
      ~ncem.estimators.EstimatorED.train
      ~ncem.estimators.EstimatorED.train_aggressive
      ~ncem.estimators.EstimatorED.train_normal
   
   

   .. _sphx_glr_backref_ncem.estimators.EstimatorED:

   .. minigallery:: ncem.estimators.EstimatorED
       :add-heading: Examples
       :heading-level: -