"""Run-to-run stability of the GAT family (GAT_infinitesimal, GAT_infinitesimal_node_level,
GAT_fully_connected, GAT_prior_knowledge, GAT_biological_prior_knowledge).

WHY THE GAT RESULTS WERE UNSTABLE
---------------------------------
No GAT fit was ever seeded: weight initialisation, dropout masks and the DataLoader's
shuffling all came from torch's global generator, which is seeded from the OS clock at start-up.
Hyperparameter tuning therefore had three compounding problems, none of which more search
iterations can fix:

  1. every candidate was scored with a DIFFERENT random draw, so the Bayesian optimiser's
     surrogate modelled initialisation noise as if it were hyperparameter signal;
  2. the best of N noisy scores is biased upwards (winner's curse), so the "winner" is usually
     the luckiest draw, not the best hyperparameters;
  3. the final confirmatory fit is one more random draw, so the reported accuracy of the
     "tuned" model differs from the score that selected it and from run to run.

WHAT THIS MODULE DOES (three independent layers, all switchable)
----------------------------------------------------------------
F1  Deterministic fits (``GAT_SEED``).  ``@gat_stable`` seeds python/numpy/torch from a digest
    of the TRAINING data (+ model family + base seed), runs the fit, and restores the global RNG
    states afterwards, so nothing else in the pipeline sees a changed random stream.  The seed
    does not depend on the hyperparameters, so every candidate of a search sees the SAME
    initialisation/dropout/shuffle stream (common random numbers): score differences between
    candidates are then caused by the hyperparameters, and the confirmatory fit of the winner is
    the same fit that was scored.  Re-running on the same data gives the same result (on CPU; see
    LIMITS).

F2  Seed ensemble for FINAL fits (``GAT_SEED_ENSEMBLE`` = K).  A single network is one random
    draw from a wide distribution; the mean of K networks has about 1/K of the initialisation
    variance and is usually more accurate.  Only fits made inside ``final_fit_role()`` (the
    untuned main-loop fit, the confirmatory fit after tuning, the async-lane job) are repeated;
    search, joint, out-of-fold and iterative-ensemble fits stay single-seed.  Member 0 is the
    F1 fit, so K>1 only ever adds members.  Predictions of valid/test/train are averaged and
    r/MSE recomputed from the averaged test predictions.  Members 1..K-1 run with
    explainability off (the marker-effect / attention / interaction outputs come from member 0).

F3  Seed-robust winner selection (``GAT_RESCORE_TOP`` = m, ``GAT_RESCORE_SEEDS`` = S).  After the
    search, the m best candidates and the default are re-scored under S additional seed salts
    (``seed_salt``); the winner is the best MEAN score, and the paired standard error of
    (winner - default) over the draws raises the margin the winner must beat the default by.

Interaction with ``SEED_BAG`` (models/seed_bag.py): a seed bag seeds each member itself, so inside a
bag member (``external_seed_scope()``) ``@gat_stable`` does nothing - the bag's per-member seeds govern
and the seed ensemble is not applied on top of a bag.

LIMITS
------
* GPU scatter/atomic kernels are not bit-deterministic, so on CUDA the fit-to-fit differences
  shrink to float round-off-driven drift but are not exactly zero.  Seed averaging (F2) and
  re-scoring (F3) work regardless of that.
* ``GAT_SEED='off'`` restores the legacy (unseeded) behaviour exactly.
"""

import contextlib
import functools
import hashlib
import random
import threading

import numpy as np
import pandas as pd

DEFAULT_GAT_SEED = 20240615

_STATE = threading.local()


# --------------------------------------------------------------------------- #
# Settings (resolved from the run's ACCELERATION block via the compute resources)
# --------------------------------------------------------------------------- #

def gat_stability_settings():
    """{'seed': int|None, 'ensemble': int>=1, 'rescore_top': int>=0, 'rescore_seeds': int>=0}.

    Absent keys (engine default): seeding ON, no ensemble, no re-scoring."""
    try:
        from pipeline_utils import get_active_compute_resources
        res = get_active_compute_resources() or {}
    except Exception:
        res = {}
    seed = res.get('accel_gat_seed', DEFAULT_GAT_SEED)
    return {
        'seed': None if seed is None else int(seed),
        'ensemble': max(1, int(res.get('accel_gat_seed_ensemble', 1) or 1)),
        'rescore_top': max(0, int(res.get('accel_gat_rescore_top', 0) or 0)),
        'rescore_seeds': max(0, int(res.get('accel_gat_rescore_seeds', 0) or 0)),
    }


def is_gat_model(model_name):
    """True for any GAT family model (including numbered instances, e.g. ..._2)."""
    return str(model_name).startswith('GAT_')


# --------------------------------------------------------------------------- #
# Thread-local roles
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def final_fit_role():
    """Mark the enclosed GAT call(s) as a FINAL fit (eligible for the seed ensemble)."""
    previous = getattr(_STATE, 'final', False)
    _STATE.final = True
    try:
        yield
    finally:
        _STATE.final = previous


def in_final_fit_role():
    return bool(getattr(_STATE, 'final', False))


@contextlib.contextmanager
def seed_salt(salt):
    """Offset every GAT seed drawn inside the block (used for re-scoring under new draws)."""
    previous = getattr(_STATE, 'salt', 0)
    _STATE.salt = int(salt)
    try:
        yield
    finally:
        _STATE.salt = previous


@contextlib.contextmanager
def external_seed_scope():
    """Declare that an OUTER caller (models/seed_bag.py, ``SEED_BAG``) has already seeded the global
    generators for the enclosed call: ``@gat_stable`` then runs the model exactly as given - no
    re-seeding (which would give every bag member the same draw) and no seed ensemble."""
    previous = getattr(_STATE, 'external', False)
    _STATE.external = True
    try:
        yield
    finally:
        _STATE.external = previous


def externally_seeded():
    return bool(getattr(_STATE, 'external', False))


def current_salt():
    return int(getattr(_STATE, 'salt', 0))


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #

def _data_digest(frame):
    """8-byte digest of a training frame (values + index + shape). Candidate-independent."""
    h = hashlib.blake2b(digest_size=8)
    try:
        h.update(repr(tuple(frame.shape)).encode())
        h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    except Exception:
        try:
            h.update(repr(tuple(frame.shape)).encode())
            h.update(np.ascontiguousarray(np.asarray(frame.iloc[:, -1], dtype=float)).tobytes())
        except Exception:
            h.update(b'unhashable')
    return h.digest()


def derive_seed(name, digest, base, salt, member):
    h = hashlib.blake2b(digest_size=8)
    h.update(str(name).encode())
    h.update(digest)
    h.update(repr((int(base), int(salt), int(member))).encode())
    return int.from_bytes(h.digest(), 'little') % (2 ** 31 - 1)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    import torch
    torch.manual_seed(seed)


@contextlib.contextmanager
def rng_isolated(seed):
    """Seed python/numpy/torch with `seed`; restore the previous global states on exit."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = cuda_states = None
    try:
        import torch
        torch_state = torch.get_rng_state()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            cuda_states = torch.cuda.get_rng_state_all()
    except Exception:
        torch = None
    _seed_everything(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch is not None and torch_state is not None:
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                try:
                    torch.cuda.set_rng_state_all(cuda_states)
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
# Decorator
# --------------------------------------------------------------------------- #

def _finite_vec(x, n):
    v = np.asarray(x, dtype=float).ravel()
    return v if (v.shape[0] == n and np.all(np.isfinite(v))) else None


def _merge_members(results, data_valid, data_test):
    """Average members' valid/test/train predictions (indices 4/3/5) and recompute r, mse."""
    base = list(results[0])
    merged = {}
    for idx in (3, 4, 5):
        first = base[idx]
        n = len(first) if first is not None else 0
        if n == 0:
            continue
        vecs = [v for v in (_finite_vec(r[idx], n) for r in results) if v is not None]
        if len(vecs) > 1:
            merged[idx] = np.mean(np.vstack(vecs), axis=0).tolist()
    for idx, val in merged.items():
        base[idx] = val
    if 3 in merged:
        try:
            from pipeline_utils import safe_regression_metrics
            actual = np.asarray(data_test.iloc[:, -1], dtype=float)
            r, mse = safe_regression_metrics(actual, np.asarray(merged[3], dtype=float))
            if np.isfinite(r) or np.isfinite(mse):
                base[0], base[1] = r, mse
        except Exception:
            pass
    return tuple(base)


def _valid_r(result, data_valid):
    try:
        if data_valid is None or data_valid.shape[0] == 0 or len(result[4]) == 0:
            return None
        from pipeline_utils import safe_regression_metrics
        r, _ = safe_regression_metrics(np.asarray(data_valid.iloc[:, -1], dtype=float),
                                       np.asarray(result[4], dtype=float))
        return r if np.isfinite(r) else None
    except Exception:
        return None


def gat_stable(model_name):
    """Decorator for the GAT model functions (see the module docstring)."""

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(data_train, data_valid, data_test, params, *args, **kwargs):
            cfg = gat_stability_settings()
            if cfg['seed'] is None or externally_seeded():
                return fn(data_train, data_valid, data_test, params, *args, **kwargs)
            digest = _data_digest(data_train)
            salt = current_salt()

            def run(member, member_params):
                seed = derive_seed(model_name, digest, cfg['seed'], salt, member)
                with rng_isolated(seed):
                    return fn(data_train, data_valid, data_test, member_params, *args, **kwargs)

            first = run(0, params)
            k = cfg['ensemble']
            if k <= 1 or not in_final_fit_role() or _in_search_fit():
                return first

            from models.hyperparameter_tuning import _disable_explainability
            extra_params = _disable_explainability(model_name, params)
            results = [first]
            for member in range(1, k):
                try:
                    out = run(member, extra_params)
                except Exception as exc:
                    print(f"[gat_stability] {model_name}: seed-ensemble member {member} failed "
                          f"({type(exc).__name__}: {exc}); continuing without it.")
                    continue
                if isinstance(out, tuple) and len(out) == len(first):
                    results.append(out)
            if len(results) == 1:
                return first
            rs = [x for x in (_valid_r(r, data_valid) for r in results) if x is not None]
            spread = (f", validation r per seed {min(rs):.3f}..{max(rs):.3f}" if len(rs) > 1 else "")
            print(f"[gat_stability] {model_name}: averaged {len(results)} seeded networks{spread}.")
            return _merge_members(results, data_valid, data_test)

        wrapper.__gat_stable__ = True
        return wrapper

    return decorate


def _in_search_fit():
    try:
        from pipeline_utils import search_fit_cfg
        return search_fit_cfg() is not None
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Seed-robust winner selection (used by hyperparameter_tuning)
# --------------------------------------------------------------------------- #

def trial_to_xs(trial, n_dims):
    """(x, score) from either a (x, score) tuple or a bayes_opt result dict; None if unusable."""
    try:
        if isinstance(trial, dict):
            x = np.array([trial['params'][f'x{i}'] for i in range(n_dims)], dtype=float)
            return x, float(trial['target'])
        x, score = trial[0], trial[1]
        return np.asarray(x, dtype=float), float(score)
    except Exception:
        return None


def top_distinct(trials, n_dims, m, floor=None):
    """The m best distinct candidates [(x, score), ...] from a search's trial list."""
    seen, out = set(), []
    parsed = [p for p in (trial_to_xs(t, n_dims) for t in trials) if p is not None]
    for x, s in sorted(parsed, key=lambda p: -p[1]):
        if not np.isfinite(s) or (floor is not None and s <= floor):
            continue
        key = tuple(np.round(x, 10))
        if key in seen:
            continue
        seen.add(key)
        out.append((x, s))
        if len(out) >= m:
            break
    return out


def select_by_seed_mean(candidates, default, rescore, n_seeds, selection_margin=0.0):
    """Re-score `candidates` ([(x, score0)]) and `default` ((x, score0)) under seed salts 1..S.

    `rescore(x, salts)` -> list of scores (NaN for a failed draw).  Returns a dict with the
    winner's x / mean / draws, the default's mean / draws and the paired SE of
    (winner - default), or None when nothing could be re-scored."""
    salts = list(range(1, int(n_seeds) + 1))

    def draws(x, s0):
        extra = [float(v) for v in rescore(x, salts)]
        return np.asarray([s0] + extra, dtype=float)

    scored = []
    for x, s0 in candidates:
        d = draws(x, s0)
        if np.isfinite(d).sum() >= 2:
            scored.append((x, d))
    d_default = draws(*default)
    if not scored or np.isfinite(d_default).sum() < 2:
        return None
    best_x, best_d = max(scored, key=lambda p: np.nanmean(p[1]))
    diff = best_d - d_default
    diff = diff[np.isfinite(diff)]
    se = float(np.std(diff, ddof=1) / np.sqrt(diff.size)) if diff.size > 1 else 0.0
    return {
        'x': best_x, 'mean': float(np.nanmean(best_d)), 'draws': best_d,
        'default_mean': float(np.nanmean(d_default)), 'default_draws': d_default,
        'paired_se': se,
        'ranked': sorted(((float(np.nanmean(d)), float(np.nanstd(d))) for _, d in scored), reverse=True),
    }
