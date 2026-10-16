"""seed_bag.py - ver4-102 (catalogue P4, seed bagging of the neural models).

A neural model (MLP, the GATs) is a different function every time it is trained, because its
initial weights, its batch order and its dropout masks come from the process-global random
generators. Averaging the predictions of several fits of the SAME model that differ only in that
random seed ("a seed bag") is the cheapest variance reduction there is, and it is the one the
catalogue (P4) marks as worth a gate before it is made a default. This module is the opt-in
machinery; nothing here runs unless ``SEED_BAG`` is set, and ``SEED_BAG=None`` (the default)
changes no byte of a run (I11).

Config (``SEED_BAG``)
---------------------
* ``None`` (default): off.
* an integer ``n`` (2 to ``MAX_BAG``): ``n`` fits per fit call, base seed ``0``.
* a dict ``{'n': int, 'seed': int}``: the same with a chosen base seed (default 0).

Anything else is ignored with a printed WARNING (the run goes on unbagged).

What a bag does
---------------
For each of the models in ``BAG_LAYOUT`` (MLP and the four non-biological-prior GATs) the engine's
single fit call is made ``n`` times. Member ``k`` runs inside :func:`isolated_rng`, which seeds
Python's ``random``, NumPy and PyTorch (CPU and CUDA) with ``member_seed(base, task_seed, k)`` and
restores the previous generator states afterwards, so the bag neither depends on nor disturbs the
random stream any other model sees. The returned prediction vectors (test, validation, train) are
the member means; Pearson r and MSE are recomputed from the averaged test prediction against the
test actuals (the same two formulas the models use), and every other return value (marker effects,
interaction pairs, attention) is member 0's, because averaging an explanation across members is a
different question this change does not answer.

The seed of a member is a pure function of the base seed, the TASK's sample number and the member
index, so it is identical whatever the sharding or the number of workers (I9), and a bagged run is
reproducible, which an unbagged neural run is not.

Where it does not apply (stated in the run log, never silent): the hyperparameter-tuning
confirmatory fit and every tuning trial (the tuned path is a different call chain), the joint
search, the Super-learner refits, and ``GAT_biological_prior_knowledge`` (its fit function has a
different, cached call signature). The bag applies to the untuned main-dispatch fit only.
"""
import contextlib
import random
from typing import Callable, Optional, Sequence

import numpy as np

MAX_BAG = 20
DEFAULT_SEED = 0
_SEED_MOD = 2 ** 31 - 1

# model name -> (index of predicted_test, predicted_valid, predicted_train) in the model function's
# return tuple. Index 0 is always pearson r and index 1 the mean squared error.
BAG_LAYOUT = {
    'MLP': (4, 5, 6),
    'GAT_infinitesimal': (3, 4, 5),
    'GAT_infinitesimal_node_level': (3, 4, 5),
    'GAT_fully_connected': (3, 4, 5),
    'GAT_prior_knowledge': (3, 4, 5),
}


def bag_applies(model_name: str) -> bool:
    """True when `model_name` (a main-dispatch ``base_model_name``) is bagged by SEED_BAG."""
    return model_name in BAG_LAYOUT


def resolve_seed_bag(raw) -> Optional[dict]:
    """Validate the ``SEED_BAG`` config value. Returns ``None`` (off) or ``{'n': int, 'seed':
    int}``. A bad value is reported once and treated as off. ``n`` below 2 is off too (a bag of one
    fit is the unbagged fit with a fixed seed, which is a different, unrequested, change)."""
    if raw is None or raw is False:
        return None
    if isinstance(raw, bool):                       # True is not a size
        print(f"[seed_bag] WARNING: SEED_BAG={raw!r} is not an integer or a dict - ignored "
              f"(no seed bagging).")
        return None
    seed = DEFAULT_SEED
    if isinstance(raw, dict):
        unknown = sorted(set(raw) - {'n', 'seed'})
        if unknown:
            print(f"[seed_bag] WARNING: unknown SEED_BAG key(s) {unknown} ignored (known: n, seed).")
        n_raw = raw.get('n')
        seed = raw.get('seed', DEFAULT_SEED)
    else:
        n_raw = raw
    try:
        if isinstance(n_raw, bool) or isinstance(seed, bool):
            raise TypeError
        if isinstance(n_raw, float) and n_raw != int(n_raw):
            raise TypeError
        n = int(n_raw)
        seed = int(seed)
    except (TypeError, ValueError):
        print(f"[seed_bag] WARNING: SEED_BAG={raw!r} is not an integer or {{'n': int, 'seed': int}} "
              f"- ignored (no seed bagging).")
        return None
    if n < 2:
        print(f"[seed_bag] WARNING: SEED_BAG n={n} is below 2 - ignored (no seed bagging).")
        return None
    if n > MAX_BAG:
        print(f"[seed_bag] WARNING: SEED_BAG n={n} is above {MAX_BAG}; using {MAX_BAG}.")
        n = MAX_BAG
    return {'n': n, 'seed': seed}


def fingerprint_view(cfg: Optional[dict]) -> Optional[dict]:
    """What the checkpoint fingerprint hashes: the resolved dict, or None when off (the engine adds
    the key only when this is not None, so older checkpoints still resume)."""
    return None if cfg is None else {'n': int(cfg['n']), 'seed': int(cfg['seed'])}


def member_seed(base_seed: int, task_seed: int, member: int) -> int:
    """The seed of bag member `member` in the task numbered `task_seed` (a pure function, so it is
    the same whatever the sharding or worker count)."""
    return (int(base_seed) * 1000003 + int(task_seed) * 7919 + int(member) * 104729 + 12345) % _SEED_MOD


@contextlib.contextmanager
def isolated_rng(seed: int):
    """Seed Python ``random``, NumPy and (when installed) PyTorch CPU and CUDA with `seed`, and put
    every generator back to its prior state on exit (also when the body raises)."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch = None
    torch_state = cuda_state = None
    try:
        import torch as _torch                      # optional: a CPU-only tree has no torch
        torch = _torch
        torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()
    except ImportError:
        torch = None
    try:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        if torch is not None:
            torch.manual_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch is not None:
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)


def _mean_vectors(vectors: Sequence, what: str):
    """Element-wise mean of equal-length prediction vectors, returned as the first member's type
    (a list stays a list, an array an array). An empty first vector (no validation set) stays as
    it is."""
    first = vectors[0]
    if len(first) == 0:
        if any(len(v) != 0 for v in vectors[1:]):
            raise ValueError(f"seed bag: members disagree on the length of the {what} predictions")
        return first
    arrays = [np.asarray(v, dtype=float).ravel() for v in vectors]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError(f"seed bag: members disagree on the length of the {what} predictions "
                         f"({[a.shape[0] for a in arrays]})")
    mean = np.mean(np.vstack(arrays), axis=0)
    return mean.tolist() if isinstance(first, list) else mean


def bag_call(fit: Callable, model_name: str, n: int, base_seed: int, task_seed: int,
             actual_test) -> tuple:
    """Call ``fit()`` (a zero-argument closure over the model function and its arguments) `n`
    times, each member inside :func:`isolated_rng`, and return one tuple of the model function's
    own layout: member means for the three prediction vectors, r and MSE recomputed from the mean
    test prediction against `actual_test`, every other element from member 0.

    Raises KeyError for a model not in ``BAG_LAYOUT`` and ValueError when members return vectors
    of different lengths."""
    from scipy import stats
    from sklearn.metrics import mean_squared_error

    from models.gat_stability import external_seed_scope   # the GAT decorator must not re-seed a member

    it, iv, itr = BAG_LAYOUT[model_name]
    members = []
    for k in range(int(n)):
        with isolated_rng(member_seed(base_seed, task_seed, k)), external_seed_scope():
            members.append(list(fit()))
    out = list(members[0])
    out[it] = _mean_vectors([m[it] for m in members], 'test')
    out[iv] = _mean_vectors([m[iv] for m in members], 'validation')
    out[itr] = _mean_vectors([m[itr] for m in members], 'training')
    actual = np.asarray(actual_test, dtype=float).ravel()
    pred = np.asarray(out[it], dtype=float).ravel()
    if actual.shape != pred.shape:
        raise ValueError(f"seed bag: test actuals ({actual.shape[0]}) and averaged predictions "
                         f"({pred.shape[0]}) differ in length")
    out[1] = float(mean_squared_error(actual, pred))
    out[0] = float(stats.pearsonr(actual, pred)[0]) if np.std(pred) > 0 and np.std(actual) > 0 \
        else float('nan')
    return tuple(out)
