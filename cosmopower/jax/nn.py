"""A JAX/Equinox training engine for CosmoPower.

This is a faithful, ground-up port of the original (TensorFlow) CosmoPower NN
emulator — ``cosmopower.cosmopower_NN`` (Spurio Mancini et al. 2022) — to
JAX/Equinox/Optax. It provides an autodiff-native *training* backend for
CosmoPower: same network, same trainable activation, same standardisation, same
staged learning-rate recipe, and the same on-disk pickle format — but powered by
JAX instead of TensorFlow, and importable without TensorFlow installed.

Compatibility with ``cosmopower-jax`` (the inference-only loader) is a
*consequence*, not the goal: because :func:`save` writes the original CosmoPower
pickle format, a model trained here also loads through
``CosmoPowerJAX(probe='custom_log', ...)``. But this module is fully
self-sufficient for training, saving, loading, prediction, and derivatives, with
no dependency on that loader.

Faithfulness to the original (TF) ``cosmopower_NN`` — verified, not assumed:
  * activation        act(x) = (beta + sigmoid(alpha*x)*(1 - beta)) * x   (Eq. A1)
  * init              W ~ N(0, 1e-3), b = 0, alpha/beta ~ N(0, 1) (hidden only)
  * forward           standardise -> [linear -> activation] x hidden -> linear
                      -> de-standardise   (NO delog: the loss lives in
                      log-feature space, exactly as in the original)
  * standardisation   parameters/features mean & std are constants, not trained
  * loss              RMSE: sqrt(mean((pred - target)**2))
  * schedule          one persistent Adam, staged learning rates (high -> low)

Two deliberate deviations from the original ``train()`` (documented there):
fixed validation split + best-validation weight rollback.

Weight orientation note: the original stores ``W`` as ``(in, out)`` and computes
``x @ W``. Here we store ``(out, in)`` and compute ``W @ x`` per sample. The two
are exact transposes; :func:`save` transposes back to ``(in, out)`` so the
pickle matches the original CosmoPower format.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import equinox as eqx
import optax


def cosmopower_activation(x: jax.Array, alpha: jax.Array, beta: jax.Array) -> jax.Array:
    """Original CosmoPower trainable activation (Spurio Mancini et al. 2022, Eq. A1).

        act(x) = (beta + sigmoid(alpha * x) * (1 - beta)) * x

    ``alpha`` and ``beta`` are per-neuron trainable vectors (length = layer width).
    """
    return (beta + jax.nn.sigmoid(alpha * x) * (1.0 - beta)) * x


class TrainableCosmoPowerNN(eqx.Module):
    """Equinox port of ``cosmopower.cosmopower_NN``.

    The trainable leaves are ``W``, ``b``, ``alphas``, ``betas``. The four
    standardisation vectors are held as constants (gradients are stopped through
    them in the forward pass, mirroring the original's ``tf.constant``), and are
    set from the training data by :meth:`with_standardisation` before training.
    """

    # trainable parameters (lists, one entry per layer / hidden layer)
    W: list           # each (out, in)
    b: list           # each (out,)
    alphas: list      # each (width,), hidden layers only  -> len = n_layers - 1
    betas: list       # each (width,), hidden layers only

    # standardisation constants (not trained)
    parameters_mean: jax.Array
    parameters_std: jax.Array
    features_mean: jax.Array
    features_std: jax.Array

    # k / ell grid (metadata, not used in the forward pass, not trained)
    modes: jax.Array

    # static metadata
    parameters: tuple = eqx.field(static=True)   # ordered parameter names
    n_parameters: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    n_hidden: tuple = eqx.field(static=True)
    architecture: tuple = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)

    def __init__(
        self,
        parameters: Sequence[str],
        modes,
        *,
        key: jax.Array,
        n_hidden: Sequence[int] = (512, 512, 512),
        parameters_mean=None,
        parameters_std=None,
        features_mean=None,
        features_std=None,
    ):
        parameters = tuple(parameters)
        n_parameters = len(parameters)
        n_hidden = tuple(int(h) for h in n_hidden)

        modes = jnp.asarray(modes, dtype=jnp.float32)
        n_modes = int(modes.shape[0])

        architecture = (n_parameters,) + n_hidden + (n_modes,)
        n_layers = len(architecture) - 1

        # one key per weight matrix + one per alpha + one per beta
        keys = jax.random.split(key, n_layers + 2 * (n_layers - 1))
        ki = 0

        W, b = [], []
        for i in range(n_layers):
            fan_in, fan_out = architecture[i], architecture[i + 1]
            # original: tf.random.normal([fan_in, fan_out], 0., 1e-3); we store (out, in)
            W.append(jax.random.normal(keys[ki], (fan_out, fan_in)) * 1e-3)
            b.append(jnp.zeros((fan_out,)))
            ki += 1

        alphas, betas = [], []
        for i in range(n_layers - 1):
            width = architecture[i + 1]
            alphas.append(jax.random.normal(keys[ki], (width,)))   # N(0, 1)
            ki += 1
            betas.append(jax.random.normal(keys[ki], (width,)))    # N(0, 1)
            ki += 1

        self.W = W
        self.b = b
        self.alphas = alphas
        self.betas = betas

        # standardisation: default to identity (zeros/ones), set from data later
        self.parameters_mean = (jnp.zeros((n_parameters,)) if parameters_mean is None
                                else jnp.asarray(parameters_mean, dtype=jnp.float32))
        self.parameters_std = (jnp.ones((n_parameters,)) if parameters_std is None
                               else jnp.asarray(parameters_std, dtype=jnp.float32))
        self.features_mean = (jnp.zeros((n_modes,)) if features_mean is None
                              else jnp.asarray(features_mean, dtype=jnp.float32))
        self.features_std = (jnp.ones((n_modes,)) if features_std is None
                             else jnp.asarray(features_std, dtype=jnp.float32))

        self.modes = modes
        self.parameters = parameters
        self.n_parameters = n_parameters
        self.n_modes = int(n_modes)
        self.n_hidden = n_hidden
        self.architecture = architecture
        self.n_layers = n_layers

    # ── forward ──────────────────────────────────────────────────────────────
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass for a single sample ``x`` of shape ``(n_parameters,)``.

        Returns the de-standardised (log-feature-space) prediction — NO delog,
        matching the space in which the original computes its training loss.
        Use :meth:`predict` for 10**(...) on log-spectrum probes.
        """
        # standardisation vectors are constants (stop gradients, like tf.constant)
        pm = jax.lax.stop_gradient(self.parameters_mean)
        ps = jax.lax.stop_gradient(self.parameters_std)
        fm = jax.lax.stop_gradient(self.features_mean)
        fs = jax.lax.stop_gradient(self.features_std)

        h = (x - pm) / ps
        for i in range(self.n_layers - 1):
            z = self.W[i] @ h + self.b[i]
            h = cosmopower_activation(z, self.alphas[i], self.betas[i])
        z = self.W[-1] @ h + self.b[-1]
        return z * fs + fm

    # ── inference API (self-contained — no CosmoPowerJAX needed) ─────────────
    def _to_array(self, input_vec) -> jax.Array:
        """Normalise input to a ``(N, n_parameters)`` float32 array.

        Accepts a dict of named parameter arrays (ordered by ``self.parameters``),
        a 1-D ``(n_parameters,)`` vector, or a 2-D ``(N, n_parameters)`` batch —
        the same input flexibility as ``CosmoPowerJAX.predict``.
        """
        if isinstance(input_vec, dict):
            input_vec = jnp.stack([jnp.asarray(input_vec[k]) for k in self.parameters], axis=-1)
        arr = jnp.asarray(input_vec, dtype=jnp.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, self.n_parameters)
        return arr

    def predictions(self, input_vec) -> jax.Array:
        """Network output in (log-)feature space — original ``predictions_np``.

        Batched and dict-aware. Single inputs return a ``(n_modes,)`` vector.
        """
        out = jax.vmap(self.__call__)(self._to_array(input_vec))
        return out.squeeze()

    def ten_to_predictions(self, input_vec) -> jax.Array:
        """``10**predictions`` — the physical spectrum for log-trained models
        (original ``ten_to_predictions_np``)."""
        return 10.0 ** self.predictions(input_vec)

    # cp-jax-style alias: for log spectra this matches CosmoPowerJAX.predict
    def predict(self, input_vec) -> jax.Array:
        return self.ten_to_predictions(input_vec)

    def derivative(self, input_vec, *, mode: str = "forward",
                   quantity: str = "ten_to_predictions") -> jax.Array:
        """d(output)/d(input parameters) by autodiff — native replacement for
        ``CosmoPowerJAX.derivative``.

        quantity : 'ten_to_predictions' (physical spectrum, default) or
                   'predictions' (log-feature space).
        mode     : 'forward' (jacfwd) or 'reverse' (jacrev).
        Returns a ``(n_modes, n_parameters)`` Jacobian per sample
        (leading batch axis kept when N > 1).
        """
        core = (self.__call__ if quantity == "predictions"
                else (lambda x: 10.0 ** self.__call__(x)))
        jac = jax.jacfwd if mode == "forward" else jax.jacrev
        d = jax.vmap(jac(core))(self._to_array(input_vec))
        return d.squeeze()

    # ── standardisation helper ───────────────────────────────────────────────
    def with_standardisation(self, parameters_mean, parameters_std,
                             features_mean, features_std) -> "TrainableCosmoPowerNN":
        """Return a copy with the standardisation constants set from data.

        Mirrors the original ``train()`` step that computes parameter/feature
        mean & std from the training set before optimisation begins.
        """
        return eqx.tree_at(
            lambda m: (m.parameters_mean, m.parameters_std,
                       m.features_mean, m.features_std),
            self,
            (jnp.asarray(parameters_mean, dtype=jnp.float32),
             jnp.asarray(parameters_std, dtype=jnp.float32),
             jnp.asarray(features_mean, dtype=jnp.float32),
             jnp.asarray(features_std, dtype=jnp.float32)),
        )


# ── training ─────────────────────────────────────────────────────────────────

def rmse_loss(model: TrainableCosmoPowerNN, x: jax.Array, y: jax.Array) -> jax.Array:
    """Root-mean-squared error in log-feature space (original CosmoPower loss).

    ``sqrt(mean((prediction - target)^2))`` — the original uses the sqrt, not a
    plain MSE.
    """
    pred = jax.vmap(model)(x)
    return jnp.sqrt(jnp.mean((pred - y) ** 2))


@eqx.filter_jit
def _eval_loss(model, x, y):
    return rmse_loss(model, x, y)


def train(
    model: TrainableCosmoPowerNN,
    training_parameters: jax.Array,
    training_features: jax.Array,
    *,
    key: jax.Array,
    validation_split: float = 0.1,
    learning_rates: Sequence[float] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
    batch_sizes: Sequence[int] = (1024, 1024, 1024, 1024, 1024),
    gradient_accumulation_steps: Sequence[int] | None = None,
    patience_values: Sequence[int] = (100, 100, 100, 100, 100),
    max_epochs: Sequence[int] = (1000, 1000, 1000, 1000, 1000),
    verbose: bool = True,
):
    """Faithful port of ``cosmopower_NN.train`` with two deliberate deviations.

    Faithful to the original:
      * standardisation (params & features mean/std) computed from the training
        data, then held constant;
      * RMSE loss in log-feature space;
      * staged cooling schedule (high -> low LR) with **one persistent Adam**
        whose momentum carries across stages (lr is swapped via
        ``optax.inject_hyperparams``);
      * per-stage early stopping with the given patience.

    Deliberate deviations (agreed):
      * **fixed validation split** drawn once (not re-drawn each stage), so
        validation losses are comparable across all epochs/stages;
      * **best-validation weight rollback** — the returned model is the one with
        the lowest validation loss ever seen, not whatever was current when the
        last stage stopped.

    ``gradient_accumulation_steps`` mirrors the original: each batch is split
    into this many sub-batches whose (size-weighted) gradients are accumulated
    before a single optimiser step — useful only when a batch is too large to
    fit in memory at once. The default of 1 is a plain step. Note that, as in
    the original, accumulation with the RMSE loss is a mild approximation
    (sqrt-of-mean is non-linear, so summed sub-batch gradients ≠ the full-batch
    gradient); keep it at 1 unless memory forces otherwise.

    Parameters
    ----------
    training_parameters : (N, n_parameters) array (columns in ``model.parameters``
        order) or a dict of named parameter arrays (ordered automatically).
    training_features   : (N, n_modes) array of (log)-spectra targets.

    Returns
    -------
    best_model : TrainableCosmoPowerNN with the best-validation weights and the
        standardisation constants set from the data.
    history : dict with 'val_loss' (per-epoch) and 'best_val_loss'.
    """
    # default: no accumulation, length-matched to the schedule
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = (1,) * len(learning_rates)

    assert (len(learning_rates) == len(batch_sizes) == len(gradient_accumulation_steps)
            == len(patience_values) == len(max_epochs)), \
        ("learning_rates, batch_sizes, gradient_accumulation_steps, patience_values, "
         "max_epochs must have equal length")

    X = model._to_array(training_parameters)   # dict -> ordered array, or array passthrough
    Y = jnp.asarray(training_features, dtype=jnp.float32)
    n = X.shape[0]

    # standardisation from the full training set (as in the original), then frozen
    model = model.with_standardisation(
        parameters_mean=jnp.mean(X, axis=0), parameters_std=jnp.std(X, axis=0),
        features_mean=jnp.mean(Y, axis=0),   features_std=jnp.std(Y, axis=0),
    )

    # fixed validation split (drawn once)
    key, skey = jax.random.split(key)
    perm = jax.random.permutation(skey, n)
    n_val = int(n * validation_split)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    x_val, y_val = X[val_idx], Y[val_idx]
    x_tr, y_tr = X[tr_idx], Y[tr_idx]
    n_tr = x_tr.shape[0]

    # one persistent Adam; lr lives in opt_state so we can swap it per stage
    # without resetting the moments (mirrors the original's single optimizer).
    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=float(learning_rates[0]))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def make_step(accum: int):
        @eqx.filter_jit
        def step(model, opt_state, xb, yb):
            if accum == 1:
                loss, grads = eqx.filter_value_and_grad(rmse_loss)(model, xb, yb)
            else:
                # split the batch into `accum` sub-batches; accumulate
                # size-weighted gradients, then take one optimiser step
                total = xb.shape[0]
                xs = jnp.array_split(xb, accum)
                ys = jnp.array_split(yb, accum)
                loss = 0.0
                grads = None
                for cx, cy in zip(xs, ys):
                    w = cx.shape[0] / total
                    li, gi = eqx.filter_value_and_grad(rmse_loss)(model, cx, cy)
                    gi = jax.tree_util.tree_map(lambda a: a * w, gi)
                    loss = loss + w * li
                    grads = gi if grads is None else jax.tree_util.tree_map(jnp.add, grads, gi)
            updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss
        return step

    best_model = model
    best_val = float(_eval_loss(model, x_val, y_val))
    history = {"train_loss": [], "val_loss": [], "best_val_loss": best_val}

    for stage, lr in enumerate(learning_rates):
        # swap the learning rate, keep Adam moments warm
        opt_state.hyperparams["learning_rate"] = jnp.asarray(float(lr), dtype=jnp.float32)
        counter = 0
        bs = int(batch_sizes[stage])
        step = make_step(int(gradient_accumulation_steps[stage]))

        for epoch in range(int(max_epochs[stage])):
            key, skey = jax.random.split(key)
            order = jax.random.permutation(skey, n_tr)
            epoch_losses = []
            for i in range(0, n_tr, bs):
                idx = order[i:i + bs]
                model, opt_state, l = step(model, opt_state, x_tr[idx], y_tr[idx])
                epoch_losses.append(l)


            val = float(_eval_loss(model, x_val, y_val))
            history["val_loss"].append(val)
            history["train_loss"].append(float(jnp.mean(jnp.stack(epoch_losses))))

            if val < best_val:
                best_val = val
                best_model = model          # rollback snapshot (one copy)
                counter = 0
            else:
                counter += 1
                if counter >= int(patience_values[stage]):
                    break

        if verbose:
            print(f"[stage {stage}] lr={lr:.0e}  epochs_run={epoch + 1:4d}  best_val_RMSE={best_val:.6e}")

    history["best_val_loss"] = best_val
    if verbose:
        print(f"\nBest validation RMSE: {best_val:.6e}")
    return best_model, history


# ── save / load (original CosmoPower pickle format) ──────────────────────────

def save(model: TrainableCosmoPowerNN, filepath: str) -> str:
    """Save in the original CosmoPower pickle format (the 15-element attribute
    list written by ``cosmopower_NN.save``).

    A file written here also loads through
    ``CosmoPowerJAX(probe='custom_log', filepath=...)`` for inference/derivatives.
    Weights are transposed back to the original ``(in, out)`` orientation; the
    full final bias is preserved (no ``b[-1]`` scalar truncation).
    """
    import pickle
    import numpy as np

    filepath = str(filepath)
    if not filepath.endswith(".pkl"):
        filepath += ".pkl"

    W_ = [np.asarray(w).T for w in model.W]            # (out, in) -> (in, out)
    b_ = [np.asarray(b) for b in model.b]
    alphas_ = [np.asarray(a) for a in model.alphas]
    betas_ = [np.asarray(b) for b in model.betas]

    attributes = [
        W_, b_, alphas_, betas_,
        np.asarray(model.parameters_mean), np.asarray(model.parameters_std),
        np.asarray(model.features_mean), np.asarray(model.features_std),
        model.n_parameters, list(model.parameters),
        model.n_modes, np.asarray(model.modes),
        list(model.n_hidden), model.n_layers, list(model.architecture),
    ]
    with open(filepath, "wb") as f:
        pickle.dump(attributes, f)
    return filepath


def load(filepath: str) -> TrainableCosmoPowerNN:
    """Load a model saved by :func:`save` (or the original CosmoPower ``save``)
    back into a ``TrainableCosmoPowerNN`` — no ``CosmoPowerJAX`` required.

    Reads the original 15-element attribute list and transposes the stored
    ``(in, out)`` weights back to this module's ``(out, in)`` convention.
    """
    import pickle
    import numpy as np

    filepath = str(filepath)
    if not filepath.endswith(".pkl"):
        filepath += ".pkl"
    with open(filepath, "rb") as f:
        (W_, b_, alphas_, betas_,
         parameters_mean, parameters_std, features_mean, features_std,
         n_parameters, parameters, n_modes, modes,
         n_hidden, n_layers, architecture) = pickle.load(f)

    # build a model with the right shapes (weights overwritten below)
    model = TrainableCosmoPowerNN(
        tuple(parameters), jnp.asarray(modes), key=jax.random.PRNGKey(0),
        n_hidden=tuple(int(h) for h in n_hidden),
        parameters_mean=parameters_mean, parameters_std=parameters_std,
        features_mean=features_mean, features_std=features_std,
    )

    W = [jnp.asarray(np.asarray(w).T) for w in W_]      # (in, out) -> (out, in)
    b = [jnp.asarray(x) for x in b_]
    alphas = [jnp.asarray(a) for a in alphas_]
    betas = [jnp.asarray(x) for x in betas_]
    return eqx.tree_at(lambda m: (m.W, m.b, m.alphas, m.betas), model, (W, b, alphas, betas))


# original CosmoPower vocabulary: ``restore`` is the counterpart to ``save``
restore = load
