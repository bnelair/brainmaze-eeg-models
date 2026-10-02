# Sphinx configuration for brainmaze-eeg-models (BrainMaze family style).
# Built by .github/workflows/docs.yml (shared brainmaze-sphinx workflow) and published to
# gh-pages: main at the site root, dev under /dev/.
import os
import sys

curwd = os.path.dirname(os.path.abspath(__file__))
projd = os.path.abspath(os.path.join(curwd, '..', '..'))
sys.path.insert(0, projd)

# Version is single-sourced from the package (pyproject.toml -> importlib.metadata).
from brainmaze_eeg_models import __version__

# -- Project information -----------------------------------------------------
project = 'BrainMaze: A Toolbox to Analyze Brain Electrophysiology, Behavior and Dynamics - EEG Models'
author = 'Filip Mivalt M.Sc., Ph.D.'
copyright = ('2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics '
             'Neurophysiology and Engineering. All rights reserved')
version = __version__
release = __version__
language = 'en'

# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon',
              'sphinx.ext.autosectionlabel']
# Prefix document path to section labels ('path/to/file:heading').
autosectionlabel_prefix_document = True
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['build', '_build', '._*', '._']
pygments_style = 'sphinx'
autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_book_theme'
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 5,
    "repository_url": "https://github.com/bnelair/brainmaze-eeg-models",
    "use_repository_button": True,
    "home_page_in_toc": True,
}
html_title = 'BrainMaze: A Toolbox to Analyze Brain Electrophysiology, Behavior and Dynamics - EEG Models'
html_short_title = 'BrainMaze: EEG Models'
html_logo = "../../img/brainmaze_1757x1762.PNG"
html_favicon = "../../img/brainmaze_173x173.png"
html_show_sphinx = False
html_show_copyright = True
htmlhelp_basename = 'brainmaze-eeg-models'
