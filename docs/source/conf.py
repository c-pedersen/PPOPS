# Configuration file for the Sphinx documentation builder.
import os
import sys
import re

# Make the project root importable
sys.path.insert(0, os.path.abspath("../.."))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google/NumPy style docstrings
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.autosummary",  # Generate summary tables
    "sphinx_autodoc_typehints",
    "autoapi.extension",
    "sphinx_mdinclude",
    "nbsphinx",
]

autosummary_generate = True
autodoc_typehints = "description"

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PPOPS"
copyright = "2025"
author = "Corey Pedersen, Sophie Abou-Rizk, and Andrew Chu"

with open("../../pyproject.toml") as f:
    text = f.read()
match = re.search(r'version\s*=\s*"([^"]+)"', text)
if match:
    release = match.group(1)
else:
    raise ValueError("Version not found in pyproject.toml")

version = release.rsplit(".", 1)[0]  # Short version (e.g., "1.2")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autosummary_generate = True  # Turn on autosummary
napoleon_google_docstring = True
napoleon_numpy_docstring = True

autoapi_dirs = ["../../src/ppops/"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "alabaster"
html_static_path = ["_static"]