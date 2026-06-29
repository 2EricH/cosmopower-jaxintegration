#!/usr/bin/env python

from setuptools import setup, find_packages
import platform
import os
import shutil

def read_file(file):
   with open(file) as f:
        return f.read()

long_description = read_file("README.md")

# Backends are OPTIONAL extras so each can be installed without the other:
#   pip install -e .[jax]   -> JAX/Equinox training backend (cosmopower.jax), no TF
#   pip install -e .[tf]    -> original TensorFlow emulators + likelihoods
# The base install is intentionally minimal (no framework), so a JAX-only user
# pulls neither TensorFlow nor the docs/dev tooling. This also makes the package
# installable on platforms without a TF wheel (e.g. Windows / Python 3.12).

# `platform.machine()` is used instead of `os.uname()` (the latter is missing on
# Windows).
if 'arm' in platform.machine().lower():
    tensorflow = 'tensorflow-metal'
else:
    tensorflow = 'tensorflow<2.14'

# minimal shared runtime dependency
install_requires = ['numpy']

extras_require = {
    # JAX/Equinox training backend (cosmopower.jax)
    'jax': ['jax', 'jaxlib', 'equinox>=0.11', 'optax>=0.2'],
    # original TensorFlow emulators + likelihoods, plus their support libs
    # (training progress bar, PCA, sampling, model download, plotting)
    'tf': [tensorflow, 'tensorflow_probability<0.22',
           'scikit-learn', 'tqdm', 'gdown', 'pyDOE', 'matplotlib'],
    # documentation site
    'docs': ['mkdocs-material', 'mkdocstrings'],
    # test suite
    'dev': ['pytest'],
}

setup(classifiers=['Operating System :: OS Independent',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research'
                  ],
      name='cosmopower',
      version='v0.2.0',
      description='Machine Learning - accelerated Bayesian inference',
      long_description_content_type = "text/markdown",
      long_description = long_description,
      author='Alessio Spurio Mancini',
      author_email='alessio.spuriomancini@rhul.ac.uk',
      license='GNU General Public License v3 (GPLv3)',
      url='https://github.com/alessiospuriomancini/cosmopower',
      packages=find_packages(),
      install_requires=install_requires,
      extras_require=extras_require,
     )

# cd to parent dir of setup.py
os.chdir(os.path.dirname(os.path.abspath(__file__)))
shutil.rmtree("dist", True)

# Clean up
shutil.rmtree("build", True)
shutil.rmtree("cosmopower.egg-info", True)
shutil.rmtree("__pycache__", True)
shutil.rmtree(".pytest_cache", True)
