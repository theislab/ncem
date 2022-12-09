============
API
============

Import ncem as::

   import ncem


Tools
--------------------------------
.. module:: ncem.tl
.. currentmodule:: ncem

    tl.glm.linear_ncem
    tl.glm.linear_ncem_deconvoluted
    tl.glm.differential_ncem
    tl.glm.differential_ncem_deconvoluted

    tl.glm.spline_linear_ncem
    tl.glm.spline_linear_ncem_deconvoluted
    tl.glm.spline_differential_ncem
    tl.glm.spline_differential_ncem_deconvoluted

Plotting
--------------------------------
.. module:: ncem.pl
.. currentmodule:: ncem

    pl.cluster_freq
    pl.noise_structure
    pl.circular
    pl.circular_rotated_labels
    pl.ablation

Estimator classes: `estimators`
--------------------------------
.. module:: ncem.estimators
.. currentmodule:: ncem

Estimator classes from ncem for advanced use.

.. autosummary::
   :toctree: api

   estimators.Estimator
   estimators.EstimatorGraph
   estimators.EstimatorNoGraph
   estimators.EstimatorCVAE
   estimators.EstimatorCVAEncem
   estimators.EstimatorED
   estimators.EstimatorEDncem
   estimators.EstimatorInteractions
   estimators.EstimatorLinear

Model classes: `models`
------------------------

.. module:: ncem.models
.. currentmodule:: ncem

Model classes from ncem for advanced use.

Classes that wrap tensorflow models.

.. autosummary::
   :toctree: api

   models.ModelCVAE
   models.ModelCVAEncem
   models.ModelED
   models.ModelEDncem
   models.ModelInteractions
   models.ModelLinear

Train: `train`
---------------

.. module:: ncem.train
.. currentmodule:: ncem

The interface for training ncem compatible models.

Trainer classes
~~~~~~~~~~~~~~~~
Classes that wrap estimator classes to use in grid search training.

.. autosummary::
   :toctree: api

   train.TrainModelCVAE
   train.TrainModelCVAEncem
   train.TrainModelED
   train.TrainModelEDncem
   train.TrainModelInteractions
   train.TrainModelLinear

Grid search summaries
~~~~~~~~~~~~~~~~~~~~~
Classes to pool evaluation metrics across fits in a grid search.

.. autosummary::
   :toctree: api

   train.GridSearchContainer
