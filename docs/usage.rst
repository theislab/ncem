============
API
============

Import ncem as::

   import ncem


Tools
--------------------------------
.. module:: ncem.tl
.. currentmodule:: ncem

NCEM tools containing linear models, variance decomposition and ablation study.


.. autosummary::
   :toctree: api

    tl.linear_ncem
    tl.linear_ncem_deconvoluted
    tl.differential_ncem
    tl.differential_ncem_deconvoluted

    tl.spline_linear_ncem
    tl.spline_linear_ncem_deconvoluted
    tl.spline_differential_ncem
    tl.spline_differential_ncem_deconvoluted

Plotting
--------------------------------
.. module:: ncem.pl
.. currentmodule:: ncem

NCEM tools containing plotting functions.


.. autosummary::
   :toctree: api

    pl.cluster_freq
    pl.noise_structure
    pl.circular
    pl.circular_rotated_labels
    pl.ablation

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
