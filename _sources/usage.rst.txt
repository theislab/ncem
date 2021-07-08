============
Usage
============


API
--------------

Import ncem as::

   import ncem


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


Tutorials
--------------
We provide tutorials in separate repository_.

* A tutorial for fitting and evaluating a interactions model on the MERFISH - brain dataset (interactions_).

If you would like to add more tutorials, feel free to contibute or open an issue.

.. _repository: https://github.com/theislab/ncem_tutorials/
.. _interactions: https://github.com/theislab/ncem_tutorials/blob/main/tutorials/model_tutorial_interactions.ipynb
