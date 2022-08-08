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

:github_url: ncem.estimators.EstimatorEDncem

ncem.estimators.EstimatorEDncem
===============================

.. currentmodule:: ncem.estimators

.. autoclass:: EstimatorEDncem

   
   
   .. rubric:: Attributes

   .. autosummary::
      :toctree: .
   
      ~ncem.estimators.EstimatorEDncem.img_keys_all
      ~ncem.estimators.EstimatorEDncem.nodes_idx_all
      ~ncem.estimators.EstimatorEDncem.nodes_idx_eval
      ~ncem.estimators.EstimatorEDncem.nodes_idx_test
      ~ncem.estimators.EstimatorEDncem.nodes_idx_train
      ~ncem.estimators.EstimatorEDncem.patient_ids_bytarget
      ~ncem.estimators.EstimatorEDncem.patient_ids_unique
      ~ncem.estimators.EstimatorEDncem.model
      ~ncem.estimators.EstimatorEDncem.img_to_patient_dict
      ~ncem.estimators.EstimatorEDncem.complete_img_keys
      ~ncem.estimators.EstimatorEDncem.a
      ~ncem.estimators.EstimatorEDncem.h_0
      ~ncem.estimators.EstimatorEDncem.h_1
      ~ncem.estimators.EstimatorEDncem.size_factors
      ~ncem.estimators.EstimatorEDncem.graph_covar
      ~ncem.estimators.EstimatorEDncem.node_covar
      ~ncem.estimators.EstimatorEDncem.domains
      ~ncem.estimators.EstimatorEDncem.covar_selection
      ~ncem.estimators.EstimatorEDncem.node_types
      ~ncem.estimators.EstimatorEDncem.node_type_names
      ~ncem.estimators.EstimatorEDncem.graph_covar_names
      ~ncem.estimators.EstimatorEDncem.node_feature_names
      ~ncem.estimators.EstimatorEDncem.n_features_type
      ~ncem.estimators.EstimatorEDncem.n_features_standard
      ~ncem.estimators.EstimatorEDncem.n_features_0
      ~ncem.estimators.EstimatorEDncem.n_features_1
      ~ncem.estimators.EstimatorEDncem.n_graph_covariates
      ~ncem.estimators.EstimatorEDncem.n_node_covariates
      ~ncem.estimators.EstimatorEDncem.n_domains
      ~ncem.estimators.EstimatorEDncem.max_nodes
      ~ncem.estimators.EstimatorEDncem.n_eval_nodes_per_graph
      ~ncem.estimators.EstimatorEDncem.vi_model
      ~ncem.estimators.EstimatorEDncem.log_transform
      ~ncem.estimators.EstimatorEDncem.model_type
      ~ncem.estimators.EstimatorEDncem.adj_type
      ~ncem.estimators.EstimatorEDncem.cond_type
      ~ncem.estimators.EstimatorEDncem.cond_depth
      ~ncem.estimators.EstimatorEDncem.output_layer
      ~ncem.estimators.EstimatorEDncem.steps_per_epoch
      ~ncem.estimators.EstimatorEDncem.validation_steps
   
   

   
   
   .. rubric:: Methods

   .. autosummary::
      :toctree: .
   
      ~ncem.estimators.EstimatorEDncem.evaluate_any
      ~ncem.estimators.EstimatorEDncem.evaluate_per_node_type
      ~ncem.estimators.EstimatorEDncem.get_data
      ~ncem.estimators.EstimatorEDncem.get_decoding_weights
      ~ncem.estimators.EstimatorEDncem.init_model
      ~ncem.estimators.EstimatorEDncem.predict
      ~ncem.estimators.EstimatorEDncem.predict_embedding_any
      ~ncem.estimators.EstimatorEDncem.pretrain_decoder
      ~ncem.estimators.EstimatorEDncem.split_data_given
      ~ncem.estimators.EstimatorEDncem.split_data_node
      ~ncem.estimators.EstimatorEDncem.split_data_target_cell
      ~ncem.estimators.EstimatorEDncem.train
      ~ncem.estimators.EstimatorEDncem.train_aggressive
      ~ncem.estimators.EstimatorEDncem.train_normal
   
   

   .. _sphx_glr_backref_ncem.estimators.EstimatorEDncem:

   .. minigallery:: ncem.estimators.EstimatorEDncem
       :add-heading: Examples
       :heading-level: -