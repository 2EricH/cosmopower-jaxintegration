#!/usr/bin/env python

from setuptools import setup, find_packages
import platform
import os
import shutil

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
all_requirements = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        all_requirements = [line.strip() for line in f.read().splitlines() if line.strip()]

def read_file(file):
   with open(file) as f:
        return f.read()

long_description = read_file("README.md")

# TensorFlow is now an *optional* dependency: only the original (TF) emulators
# need it. The JAX training backend (`cosmopower.jax`) does not. This also makes
# the package installable on platforms without a TF wheel (e.g. Windows).
# `platform.machine()` is used instead of `os.uname()` (the latter is missing on
# Windows).
if 'arm' in platform.machine().lower():
    tensorflow = 'tensorflow-metal'
else:
    tensorflow = 'tensorflow<2.14'
tf_requirements = [tensorflow, 'tensorflow_probability<0.22']

# base install requirements = everything in requirements.txt that is NOT TensorFlow
install_requires = [r for r in all_requirements if 'tensorflow' not in r.lower()]

extras_require = {
    # original TensorFlow emulators + likelihoods
    'tf': tf_requirements,
    # JAX/Equinox training backend (cosmopower.jax)
    'jax': ['jax', 'jaxlib', 'equinox>=0.11', 'optax>=0.2'],
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
