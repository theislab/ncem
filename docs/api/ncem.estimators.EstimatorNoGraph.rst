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

:github_url: ncem.estimators.EstimatorNoGraph

ncem.estimators.EstimatorNoGraph
================================

.. currentmodule:: ncem.estimators

.. autoclass:: EstimatorNoGraph

   
   
   .. rubric:: Attributes

   .. autosummary::
      :toctree: .
   
      ~ncem.estimators.EstimatorNoGraph.img_keys_all
      ~ncem.estimators.EstimatorNoGraph.nodes_idx_all
      ~ncem.estimators.EstimatorNoGraph.nodes_idx_eval
      ~ncem.estimators.EstimatorNoGraph.nodes_idx_test
      ~ncem.estimators.EstimatorNoGraph.nodes_idx_train
      ~ncem.estimators.EstimatorNoGraph.patient_ids_bytarget
      ~ncem.estimators.EstimatorNoGraph.patient_ids_unique
      ~ncem.estimators.EstimatorNoGraph.img_to_patient_dict
      ~ncem.estimators.EstimatorNoGraph.complete_img_keys
      ~ncem.estimators.EstimatorNoGraph.a
      ~ncem.estimators.EstimatorNoGraph.h_0
      ~ncem.estimators.EstimatorNoGraph.h_1
      ~ncem.estimators.EstimatorNoGraph.size_factors
      ~ncem.estimators.EstimatorNoGraph.graph_covar
      ~ncem.estimators.EstimatorNoGraph.node_covar
      ~ncem.estimators.EstimatorNoGraph.domains
      ~ncem.estimators.EstimatorNoGraph.covar_selection
      ~ncem.estimators.EstimatorNoGraph.node_types
      ~ncem.estimators.EstimatorNoGraph.node_type_names
      ~ncem.estimators.EstimatorNoGraph.graph_covar_names
      ~ncem.estimators.EstimatorNoGraph.node_feature_names
      ~ncem.estimators.EstimatorNoGraph.n_features_type
      ~ncem.estimators.EstimatorNoGraph.n_features_standard
      ~ncem.estimators.EstimatorNoGraph.n_features_0
      ~ncem.estimators.EstimatorNoGraph.n_features_1
      ~ncem.estimators.EstimatorNoGraph.n_graph_covariates
      ~ncem.estimators.EstimatorNoGraph.n_node_covariates
      ~ncem.estimators.EstimatorNoGraph.n_domains
      ~ncem.estimators.EstimatorNoGraph.max_nodes
      ~ncem.estimators.EstimatorNoGraph.n_eval_nodes_per_graph
      ~ncem.estimators.EstimatorNoGraph.vi_model
      ~ncem.estimators.EstimatorNoGraph.log_transform
      ~ncem.estimators.EstimatorNoGraph.model_type
      ~ncem.estimators.EstimatorNoGraph.adj_type
      ~ncem.estimators.EstimatorNoGraph.cond_type
      ~ncem.estimators.EstimatorNoGraph.cond_depth
      ~ncem.estimators.EstimatorNoGraph.output_layer
      ~ncem.estimators.EstimatorNoGraph.steps_per_epoch
      ~ncem.estimators.EstimatorNoGraph.validation_steps
   
   

   
   
   .. rubric:: Methods

   .. autosummary::
      :toctree: .
   
      ~ncem.estimators.EstimatorNoGraph.evaluate_any
      ~ncem.estimators.EstimatorNoGraph.evaluate_per_node_type
      ~ncem.estimators.EstimatorNoGraph.get_data
      ~ncem.estimators.EstimatorNoGraph.init_model
      ~ncem.estimators.EstimatorNoGraph.predict
      ~ncem.estimators.EstimatorNoGraph.pretrain_decoder
      ~ncem.estimators.EstimatorNoGraph.split_data_given
      ~ncem.estimators.EstimatorNoGraph.split_data_node
      ~ncem.estimators.EstimatorNoGraph.split_data_target_cell
      ~ncem.estimators.EstimatorNoGraph.train
      ~ncem.estimators.EstimatorNoGraph.train_aggressive
      ~ncem.estimators.EstimatorNoGraph.train_normal
   
   

   .. _sphx_glr_backref_ncem.estimators.EstimatorNoGraph:

   .. minigallery:: ncem.estimators.EstimatorNoGraph
       :add-heading: Examples
       :heading-level: -