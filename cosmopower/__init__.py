# The original (TensorFlow) emulators are imported only when TensorFlow is
# available, so the JAX training backend (``cosmopower.jax``) can be used in a
# TF-free environment. With TensorFlow installed, the full original API is
# exposed exactly as before.
import importlib.util as _importlib_util

if _importlib_util.find_spec("tensorflow") is not None:
    from .cosmopower_PCA import cosmopower_PCA
    from .cosmopower_PCAplusNN import cosmopower_PCAplusNN
    from .cosmopower_NN import cosmopower_NN
    from .likelihoods import *

__version__ = "0.1.3"
__author__ = 'Alessio Spurio Mancini'
