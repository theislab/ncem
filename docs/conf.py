#!/usr/bin/env python
# mypy: ignore-errors
# ncem documentation build configuration file
#
# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import os
import sys
from pathlib import Path

from sphinx.application import Sphinx

HERE = Path(__file__)
sys.path.insert(0, str(HERE.parent.parent))  # this way, we don't have to install ncem

needs_sphinx = "4.0"

# -- Project information ---------------------------------------------

project = "ncem"
author = "David S. Fischer, Anna C. Schaar"
copyright = "2021, theislab"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = "0.1.5"
# The full version, including alpha/beta/rc tags.
release = "0.1.5"


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.

# Add 'sphinx_automodapi.automodapi' if you want to build modules
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_gallery.load_style",
    "nbsphinx",
    "sphinx_rtd_theme",
    "sphinxcontrib.bibtex",
    # "typed_returns",
    # "IPython.sphinxext.ipython_console_highlighting",
]

intersphinx_mapping = dict(  # noqa: C408
    python=("https://docs.python.org/3", None),
    numpy=("https://numpy.org/doc/stable/", None),
    statsmodels=("https://www.statsmodels.org/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    scanpy=("https://scanpy.readthedocs.io/en/stable/", None),
    matplotlib=("https://matplotlib.org/stable/", None),
    seaborn=("https://seaborn.pydata.org/", None),
    joblib=("https://joblib.readthedocs.io/en/latest/", None),
    networkx=("https://networkx.org/documentation/stable/", None),
    dask=("https://docs.dask.org/en/latest/", None),
    skimage=("https://scikit-image.org/docs/stable/", None),
    sklearn=("https://scikit-learn.org/stable/", None),
    numba=("https://numba.readthedocs.io/en/stable/", None),
    xarray=("https://xarray.pydata.org/en/stable/", None),
    omnipath=("https://omnipath.readthedocs.io/en/latest", None),
    napari=("https://napari.org/", None),
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = [".rst", ".ipynb"]
master_doc = "index"
pygments_style = "sphinx"
language = "English"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "estimators/**.py",
    "interpretation/**.py",
    "train/**.py",
    "**.ipynb_checkpoints",
    "_build",
    "Thumbs.db",
    ".DS_Store",
]
suppress_warnings = ["download.not_readable", "git.too_shallow"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
autosummary_generate = True
autodoc_member_order = "groupwise"
autodoc_typehints = "signature"
autodoc_docstring_signature = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
todo_include_todos = False
napoleon_custom_sections = [("Params", "Parameters")]

# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/img/ncem_logo.png"
html_theme_options = {"navigation_depth": 4, "logo_only": True}
html_show_sphinx = False

html_context = dict(
    display_github=True,  # Integrate GitHub
    github_user="theislab",  # Username
    github_repo="ncem",  # Repo name
    github_version="main",  # Version
    conf_py_path="/docs/",  # Path in the checkout to the docs root
)

gh_url = "https://github.com/{github_user}/{github_repo}".format_map(html_context)

# -- Images for plot functions -------------------------------------------------


def insert_function_images(app, what, name, obj, options, lines):
    path = Path(__file__).parent / "api" / f"{name}.png"
    if what != "function" or not path.is_file():
        return
    lines[0:0] = [f".. image:: {path.name}", "   :width: 200", "   :align: right", ""]


# -- GitHub links --------------------------------------------------------------


def autolink(url_template, title_template="{}"):
    from docutils import nodes

    def role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        url = url_template.format(text)
        title = title_template.format(text)
        node = nodes.reference(rawtext, title, refuri=url, **options)
        return [node], []

    return role


def setup(app: Sphinx) -> None:
    # app.connect("autodoc-process-docstring", insert_function_images)
    # app.add_role("pr", autolink(f"{gh_url}/pull/{{}}", "PR {}"))
    app.add_css_file("css/custom.css")
    app.add_css_file("css/sphinx_gallery.css")
    app.add_css_file("css/nbsphinx.css")
    app.add_css_file("css/dataframe.css")  # had to add this manually
