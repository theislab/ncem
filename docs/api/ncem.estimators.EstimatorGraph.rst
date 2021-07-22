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

:github_url: ncem.estimators.EstimatorGraph

ncem.estimators.EstimatorGraph
==============================

.. currentmodule:: ncem.estimators

.. autoclass:: EstimatorGraph

   
   
   .. rubric:: Attributes

   .. autosummary::
      :toctree: .
   
      ~ncem.estimators.EstimatorGraph.img_keys_all
      ~ncem.estimators.EstimatorGraph.nodes_idx_all
      ~ncem.estimators.EstimatorGraph.nodes_idx_eval
      ~ncem.estimators.EstimatorGraph.nodes_idx_test
      ~ncem.estimators.EstimatorGraph.nodes_idx_train
      ~ncem.estimators.EstimatorGraph.patient_ids_bytarget
      ~ncem.estimators.EstimatorGraph.patient_ids_unique
   
   

   
   
   .. rubric:: Methods

   .. autosummary::
      :toctree: .
   
      ~ncem.estimators.EstimatorGraph.evaluate_any
      ~ncem.estimators.EstimatorGraph.evaluate_per_node_type
      ~ncem.estimators.EstimatorGraph.get_data
      ~ncem.estimators.EstimatorGraph.init_model
      ~ncem.estimators.EstimatorGraph.predict
      ~ncem.estimators.EstimatorGraph.pretrain_decoder
      ~ncem.estimators.EstimatorGraph.split_data_given
      ~ncem.estimators.EstimatorGraph.split_data_node
      ~ncem.estimators.EstimatorGraph.split_data_target_cell
      ~ncem.estimators.EstimatorGraph.train
      ~ncem.estimators.EstimatorGraph.train_aggressive
      ~ncem.estimators.EstimatorGraph.train_normal
   
   

   .. _sphx_glr_backref_ncem.estimators.EstimatorGraph:

   .. minigallery:: ncem.estimators.EstimatorGraph
       :add-heading: Examples
       :heading-level: -