# The original (TensorFlow) emulators are imported only when TensorFlow is
# available, so the JAX training backend (``cosmopower.jax``) can be used in a
# TF-free environment. With TensorFlow installed, the full original API is
# exposed exactly as before.
#
# The TF import is best-effort: some environments (e.g. Colab) ship TensorFlow
# but have a numpy/scipy combo the original modules can't import. In that case
# we degrade gracefully — the JAX backend (``cosmopower.jax``) does not need the
# TF emulators, so a failure here must not break ``import cosmopower``.
import importlib.util as _importlib_util

if _importlib_util.find_spec("tensorflow") is not None:
    try:
        from .cosmopower_PCA import cosmopower_PCA
        from .cosmopower_PCAplusNN import cosmopower_PCAplusNN
        from .cosmopower_NN import cosmopower_NN
        from .likelihoods import *
    except Exception as _e:   # TF present but its cosmopower modules failed to import
        import warnings as _warnings
        _warnings.warn(f"cosmopower: TensorFlow emulators unavailable ({_e!r}); "
                       "the JAX backend (cosmopower.jax) is still usable.")

__version__ = "0.1.3"
__author__ = 'Alessio Spurio Mancini'
