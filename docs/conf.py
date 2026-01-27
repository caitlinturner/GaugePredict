# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add src directory to path so GaugePredict can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# -- Project information -----------------------------------------------------
project = 'GaugePredict'
copyright = '2026, Caitlin R. R. Turner, Jo Martin, Matthew Hiatt'
author = 'Caitlin R. R. Turner, Jo Martin, Matthew Hiatt'
release = '1.0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# ReadTheDocs theme options
html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'titles_only': False,
}

html_logo = None
html_favicon = None

# Add custom CSS
def setup(app):
    app.add_css_file('custom.css')

# -- Extension configuration -------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Mock optional heavy deps so autodoc succeeds on RTD build without full stack
autodoc_mock_imports = ['dataretrieval', 'shap', 'geopandas', 'contextily', 'cmocean']
