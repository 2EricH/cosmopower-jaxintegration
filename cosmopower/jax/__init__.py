"""JAX/Equinox training backend for CosmoPower.

A faithful, autodiff-native port of the original (TensorFlow) ``cosmopower_NN``
training engine. Importable without TensorFlow installed::

    from cosmopower.jax import TrainableCosmoPowerNN, train, save, load

See :mod:`cosmopower.jax.nn` for details.
"""

from .nn import (
    TrainableCosmoPowerNN,
    cosmopower_activation,
    rmse_loss,
    train,
    save,
    load,
    restore,
)

__all__ = [
    "TrainableCosmoPowerNN",
    "cosmopower_activation",
    "rmse_loss",
    "train",
    "save",
    "load",
    "restore",
]
