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

:github_url: ncem.estimators.EstimatorLinear

ncem.estimators.EstimatorLinear
===============================

.. currentmodule:: ncem.estimators

.. autoclass:: EstimatorLinear

   
   
   .. rubric:: Attributes

   .. autosummary::
      :toctree: .
   
      ~ncem.estimators.EstimatorLinear.img_keys_all
      ~ncem.estimators.EstimatorLinear.nodes_idx_all
      ~ncem.estimators.EstimatorLinear.nodes_idx_eval
      ~ncem.estimators.EstimatorLinear.nodes_idx_test
      ~ncem.estimators.EstimatorLinear.nodes_idx_train
      ~ncem.estimators.EstimatorLinear.patient_ids_bytarget
      ~ncem.estimators.EstimatorLinear.patient_ids_unique
      ~ncem.estimators.EstimatorLinear.img_to_patient_dict
      ~ncem.estimators.EstimatorLinear.complete_img_keys
      ~ncem.estimators.EstimatorLinear.a
      ~ncem.estimators.EstimatorLinear.h_0
      ~ncem.estimators.EstimatorLinear.h_1
      ~ncem.estimators.EstimatorLinear.size_factors
      ~ncem.estimators.EstimatorLinear.graph_covar
      ~ncem.estimators.EstimatorLinear.node_covar
      ~ncem.estimators.EstimatorLinear.domains
      ~ncem.estimators.EstimatorLinear.covar_selection
      ~ncem.estimators.EstimatorLinear.node_types
      ~ncem.estimators.EstimatorLinear.node_type_names
      ~ncem.estimators.EstimatorLinear.graph_covar_names
      ~ncem.estimators.EstimatorLinear.node_feature_names
      ~ncem.estimators.EstimatorLinear.n_features_type
      ~ncem.estimators.EstimatorLinear.n_features_standard
      ~ncem.estimators.EstimatorLinear.n_features_0
      ~ncem.estimators.EstimatorLinear.n_features_1
      ~ncem.estimators.EstimatorLinear.n_graph_covariates
      ~ncem.estimators.EstimatorLinear.n_node_covariates
      ~ncem.estimators.EstimatorLinear.n_domains
      ~ncem.estimators.EstimatorLinear.max_nodes
      ~ncem.estimators.EstimatorLinear.n_eval_nodes_per_graph
      ~ncem.estimators.EstimatorLinear.vi_model
      ~ncem.estimators.EstimatorLinear.log_transform
      ~ncem.estimators.EstimatorLinear.model_type
      ~ncem.estimators.EstimatorLinear.adj_type
      ~ncem.estimators.EstimatorLinear.cond_type
      ~ncem.estimators.EstimatorLinear.cond_depth
      ~ncem.estimators.EstimatorLinear.output_layer
      ~ncem.estimators.EstimatorLinear.steps_per_epoch
      ~ncem.estimators.EstimatorLinear.validation_steps
   
   

   
   
   .. rubric:: Methods

   .. autosummary::
      :toctree: .
   
      ~ncem.estimators.EstimatorLinear.evaluate_any
      ~ncem.estimators.EstimatorLinear.evaluate_per_node_type
      ~ncem.estimators.EstimatorLinear.get_data
      ~ncem.estimators.EstimatorLinear.init_model
      ~ncem.estimators.EstimatorLinear.predict
      ~ncem.estimators.EstimatorLinear.pretrain_decoder
      ~ncem.estimators.EstimatorLinear.split_data_given
      ~ncem.estimators.EstimatorLinear.split_data_node
      ~ncem.estimators.EstimatorLinear.split_data_target_cell
      ~ncem.estimators.EstimatorLinear.train
      ~ncem.estimators.EstimatorLinear.train_aggressive
      ~ncem.estimators.EstimatorLinear.train_normal
   
   

   .. _sphx_glr_backref_ncem.estimators.EstimatorLinear:

   .. minigallery:: ncem.estimators.EstimatorLinear
       :add-heading: Examples
       :heading-level: -