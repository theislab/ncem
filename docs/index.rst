|PyPI| |Python Version| |License| |Read the Docs| |Build| |pre-commit| |Black|


NCEM
===========================

**NCEM** is a tool for the inference of cell-cell communication in spatial molecular data. 

.. image:: https://raw.githubusercontent.com/theislab/ncem/main/docs/_static/img/concept.png
   :target: https://raw.githubusercontent.com/theislab/ncem/main/docs/_static/img/concept.png
   :align: center
   :alt: ncem concept
   :width: 1000px


Manuscript
--------
Please see our manuscript :cite:`fischer_modeling_2023` in **Nature Biotechnology** to learn more. 

NCEM's key application
----------------------

* Node-centric expression models (NCEMs) are proposed to improve cell communication inference by using spatial graphs of cells to constrain axes of cellular communication.

* NCEMs can be used for cell communication inference captured with different spatial profiling technologies and are not limited to receptor-ligand signaling.

* NCEMs can be applied to deconvoluted spot transcriptomics. 

* Dependencies inferred by NCEMs are directional. 

Getting started with NCEM
-------------------------

You can install *ncem* via pip_ from PyPI_:

.. code:: console

   $ pip install ncem


Contributing to NCEM
-----------------------
We are happy about any contributions! Before you start, check out our contributor guide.

.. toctree::
    :caption: General
    :maxdepth: 2
    :hidden:

    installation
    contributing
    ecosystem
    authors
    references
    code_of_conduct

.. toctree::
    :caption: Methods
    :maxdepth: 2
    :hidden:

    usage
    tutorials


.. |PyPI| image:: https://img.shields.io/pypi/v/ncem.svg
   :target: https://pypi.org/project/ncem/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/ncem
   :target: https://pypi.org/project/ncem
   :alt: Python Version
.. |License| image:: https://img.shields.io/github/license/theislab/ncem
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/ncem/latest.svg?label=Read%20the%20Docs
   :target: https://ncem.readthedocs.io/
   :alt: Read the documentation at https://ncem.readthedocs.io/
.. |Build| image:: https://github.com/theislab/ncem/workflows/Build%20ncem%20Package/badge.svg
   :target: https://github.com/theislab/ncem/actions?workflow=Package
   :alt: Build Package Status
.. |Tests| image:: https://github.com/theislab/ncem/workflows/Run%20ncem%20Tests/badge.svg
   :target: https://github.com/theislab/ncem/actions?workflow=Tests
   :alt: Run Tests Status
.. |Codecov| image:: https://codecov.io/gh/theislab/ncem/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/theislab/ncem
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black

.. _ncem: https://ncem.readthedocs.io
.. _cookietemple: https://cookietemple.com
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _PyPI: https://pypi.org/
.. _Hypermodern_Python_Cookiecutter: https://github.com/cjolowicz/cookiecutter-hypermodern-python
.. _pip: https://pip.pypa.io/
.. _Usage: https://ncem.readthedocs.io/en/latest/usage.html
.. _preprint: https://www.biorxiv.org/content/10.1101/2021.07.11.451750v1
.. _paper: https://www.nature.com/articles/s41587-022-01467-z

