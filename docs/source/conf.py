# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RatesLib'
copyright = '2022, JHM Darbyshire'
author = 'JHM Darbyshire'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # 'sphinx_exec_code',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_automodapi.automodapi'
]
numpydoc_show_class_members = False  # for sphinx automod accoridng to docs
exec_code_working_dir = '../..'

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'bizstyle'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
  "external_links": [
      {"name": "Supplemental", "url": "https://www.amazon.com/Pricing-Trading-Interest-Rate-Derivatives/dp/0995455538"},
  ]
}
html_logo = "_static/rateslib_logo4.gif"
html_css_files = [
    'css/getting_started.css',
    'css/overwrites.css',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

## the following used only to open external links in new tab