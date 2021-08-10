Contributor Guide
=================

Thank you for your interest in improving this project.
This project is open-source under the `BSD license`_ and
highly welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- `Source Code`_
- `Documentation`_
- `Issue Tracker`_
- `Code of Conduct`_

.. _BSD license: https://opensource.org/licenses/BSD
.. _Source Code: https://github.com/theislab/ncem
.. _Documentation: https://ncem.readthedocs.io/
.. _Issue Tracker: https://github.com/theislab/ncem/issues

How to add a dataloader
-----------------------
Overview of contributing dataloaders to ncem.

1. Install ncem.
    Clone ncem into a local repository from `development` branch and install via pip.

.. code-block::

    cd target_directory
    git clone https://github.com/theislab/ncem.git
    git checkout development
    # git pull  # use this to update your installation
    cd ncem  # go into ncem directory
    pip install -e .  # install

2. Create a new dataloader in `data.py`
    Your dataloader should be a new class in `data.py` (ideally named by first author, e.g. DataLoaderZhang) and
    should contain the following functions `_register_celldata`, `_register_img_celldata` and `_register_graph_features`.

    `_register_celldata` creates an AnnData object called `celldata` of the complete dataset with feature names
    stored in `celldata.var_names`. Cell type annotations are stored in `celldata.obs`. `celldata.uns['metadata']`
    should contain the naming conventions of files and columns in obs.

    `_register_img_celldata` then automatically splits the `celldata` into a dictionary of AnnData object with one
    AnnData object per image in the dataset.

    `_register_graph_features` can be added in case of additional graph features, e.g. disease status of images.

    Additionally, each dataloader should have a class attribute `cell_type_merge_dict` which provides a logic on how to
    merge cell types in any subsequent analysis. This can be helpful when considering datasets with fine cell type
    annotations and a coarser annotation is wanted.

3. Make loader public (Optional).
        You can contribute the data loader to public ncem as code through a pull request.
        Note that you can also just keep the data loader in your local installation if you do not want to make it public.

.. code-block::

    # make sure you are in the top-level ncem directory from step 1
    git add *
    git commit  # enter your commit description
    # Next make sure you are up to date with dev
    git checkout development
    git pull
    git checkout YOUR_BRANCH_NAME
    git merge development
    git push  # this starts the pull request.
..

In any case, feel free to open an GitHub issue on on the `Issue Tracker`_.

How to report a bug
-------------------

Report bugs on the `Issue Tracker`_.


How to request a feature
------------------------

Request features on the `Issue Tracker`_.


How to set up your development environment
------------------------------------------

You need Python 3.7+ and the following tools:

- Poetry_
- Nox_
- nox-poetry_

You can install them with:

.. code:: console

    $ pip install poetry nox nox-poetry

Install the package with development requirements:

.. code:: console

   $ make install

You can now run an interactive Python session,
or the command-line interface:

.. code:: console

   $ poetry run python
   $ poetry run ncem

.. _Poetry: https://python-poetry.org/
.. _Nox: https://nox.thea.codes/
.. _nox-poetry: https://nox-poetry.readthedocs.io/


How to test the project
-----------------------

Run the full test suite:

.. code:: console

   $ nox

List the available Nox sessions:

.. code:: console

   $ nox --list-sessions

You can also run a specific Nox session.
For example, invoke the unit test suite like this:

.. code:: console

   $ nox --session=tests

Unit tests are located in the ``tests`` directory,
and are written using the pytest_ testing framework.

.. _pytest: https://pytest.readthedocs.io/


How to submit changes
---------------------

Open a `pull request`_ to submit changes to this project against the ``development`` branch.

Your pull request needs to meet the following guidelines for acceptance:

- The Nox test suite must pass without errors and warnings.
- Include unit tests. This project maintains a high code coverage.
- If your changes add functionality, update the documentation accordingly.

To run linting and code formatting checks before committing your change, you can install pre-commit as a Git hook by running the following command:

.. code:: console

   $ nox --session=pre-commit -- install

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

.. _pull request: https://github.com/theislab/ncem/pulls
.. _Code of Conduct: CODE_OF_CONDUCT.rst
