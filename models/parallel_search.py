"""
EasiGP - models/parallel_search.py
--------------------------------------------------------------------------
ver4-5 blueprint R1.b (Stage 5) - constant-liar BATCH Bayesian optimisation.

WHY THIS MODULE EXISTS
--------------------------------------------------------------------------
Standard Bayesian optimisation is strictly sequential: the surrogate (a
Gaussian Process) must be updated with observation n before candidate
n+1 can be chosen, so a naive implementation can never use more than one
CPU core no matter how many are available. *Batch* (or *asynchronous*)
Bayesian optimisation gets around this by obtaining q candidates per
round instead of 1: for each of the q candidates requested within one
round, a temporary, fabricated ("lied") observation is registered at the
just-suggested point before asking the acquisition function for the
next one - this discourages the acquisition function from proposing the
same promising region q times in a row, purely from having "seen" its
own in-flight suggestions. Constant Liar (Ginsbourger et al., 2010) is
the classical, simplest variant used here; this module implements the
'max' (pessimistic - the standard Constant-Liar-max) and 'mean' liar
strategies, plus an optional 'believer' (Kriging Believer: the lie is
the GP posterior mean at the suggested point, rather than a fixed
aggregate of past observations) strategy behind the same `liar=` key.

Once q candidates are chosen this way, they are handed BACK to the
caller as a single batch (`objective_batch(list_of_x) -> list_of_scores`)
for TRUE evaluation - which the caller is free to run across several
worker PROCESSES (a real wall-clock speed-up) or serially (still
correct, just not faster). This module has no opinion about how
`objective_batch` is implemented: it never imports joblib, never spawns
a process, and never has to reason about picklability (invariant I13) -
that responsibility belongs entirely to the caller
(models/hyperparameter_tuning.py::search_bayesian, via
_evaluate_batch()). This mirrors this module's sibling
models/hyperparameter_tuning.py's own declared "no project imports"
purity, so both can be imported and unit tested completely independently
of genomic_prediction.py, Streamlit, or any EasiGP-specific data shape.

VERIFIED AGAINST THE PINNED LIBRARY
--------------------------------------------------------------------------
`bayesian-optimization==3.0.1` exposes exactly the primitives this module
needs: `BayesianOptimization(f=None, ...)` (no objective function needed
when the caller drives suggest()/register() manually), `.suggest()`,
`.register(params=..., target=...)`, and
`SequentialDomainReductionTransformer.initialize()`/`.transform()`
(called against `optimizer.space`, a `TargetSpace` instance).
`optimizer.save_state()`/`load_state()` is NOT usable here - it raises
`KeyError('kappa')` under the `ExpectedImprovement` acquisition this
codebase configures (that key is specific to the older, UCB-style
`kappa` acquisition) - so instead of persisting optimiser state between
rounds, this module rebuilds a fresh `BayesianOptimization` from the
accumulated TRUE observations at the start of every round and re-derives
sequential domain reduction's own progressively-narrowing bounds by
keeping ONE long-lived `SequentialDomainReductionTransformer` instance
across the whole search (its own internal `previous_optimal`/
`current_optimal` state accumulates naturally across repeated
`.transform()` calls, exactly as it does inside `optimizer.maximize()`'s
own internal loop - `.initialize()` is called exactly ONCE, up front).
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from bayes_opt import BayesianOptimization
except ImportError:  # pragma: no cover - already a project dependency
    BayesianOptimization = None


def _constant_liar_value(x: np.ndarray, optimizer, observed_targets: Sequence[float],
                          liar: str) -> float:
    """The fabricated ('lied') target registered for a just-suggested,
    not-yet-truly-evaluated point within one batch round.

    Parameters
    ----------
    x : the just-suggested point (only used by the 'believer' strategy).
    optimizer : this round's `BayesianOptimization` instance (only used
        by 'believer', to read its underlying GP surrogate).
    observed_targets : every TRUE target observed so far (chronological,
        across every previous round) - the pool 'max'/'mean' draw from.
    liar : 'max' (default - the classical Constant-Liar-max, pessimistic:
        discourages the acquisition function from proposing the same
        promising region again within this same round), 'mean', or
        'believer' (Kriging Believer - the GP posterior mean AT the
        suggested point, reached via a private `_gp` attribute and
        therefore wrapped in try/except exactly like the existing
        `optimizer._gp.set_params(alpha=...)` call in
        models/hyperparameter_tuning.py::search_bayesian; falls back to
        'max' if the private attribute/API is unavailable).

    Returns
    -------
    A single float. 0.0 when there are no true observations yet at all
    (round 0, before anything has ever been truly evaluated, and
    'believer' isn't usable yet either since the GP hasn't been fit).
    """
    if liar == 'believer' and optimizer is not None:
        try:
            x_row = np.asarray(x, dtype=float).reshape(1, -1)
            mean_pred = optimizer._gp.predict(x_row)
            return float(np.asarray(mean_pred).reshape(-1)[0])
        except Exception:
            pass  # falls through to 'max' below - guarded private-attribute access
    if not observed_targets:
        return 0.0
    if liar == 'mean':
        return float(np.mean(observed_targets))
    return float(np.max(observed_targets))  # 'max' (default) and the 'believer' fallback


def batch_bayesian_maximise(objective_batch: Callable[[List[np.ndarray]], Sequence[float]],
                             pbounds: Sequence[Tuple[float, float]], *,
                             n_iter: int, init_points: int, seed: int, q: int,
                             probe_points: Optional[List[np.ndarray]] = None,
                             acquisition_factory: Optional[Callable[[], object]] = None,
                             bounds_transformer_factory: Optional[Callable[[], object]] = None,
                             liar: str = 'max') -> Tuple[np.ndarray, float, List[Tuple[np.ndarray, float]]]:
    """Constant-liar batch Bayesian optimisation over `pbounds`.

    Update ID ver4-6, R1/R2 note (no signature/logic change in this
    module - `pbounds` is simply whatever encoding the caller supplies):
    as of ver4-6, `models/hyperparameter_tuning.py::search_bayesian`
    calls this with `pbounds` already unit-cube-encoded (every dimension
    `(0.0, 1.0)` - see that function's own R1.2(a) docstring), so this
    module's own candidate suggestion/constant-liar machinery
    automatically benefits from the same isotropic scaling without any
    change here. Similarly, R2's relative failure scoring
    (`FAILURE_FLOOR`/`_failure_score()` in the caller) is computed
    PER-WORKER inside `objective_batch` (via `_evaluate_batch()`), not in
    this module - each worker's own `worst_finite` closure state does
    NOT propagate back here or across workers (this module never touches
    it), which is expected and disclosed (ver4-6 blueprint RK-10): a
    worker's own first failure this round falls back to `FAILURE_FLOOR`,
    already well-scaled on its own, so the E2 GP-flattening failure mode
    this update fixes does not return - only the exact SEQUENCE of
    "worst score seen so far" differs from a serial run, never the
    correctness of the scoring itself.

    `objective_batch(list_of_x) -> list_of_floats` truly evaluates a
    whole round of candidates at once; the caller decides whether that
    is a process fan-out or a serial loop (see the module docstring).

    Evaluation-budget accounting: exactly `1 + init_points + n_iter`
    candidates are evaluated in total (matching the SERIAL
    `search_bayesian()` code path's own budget exactly - one explicit
    probe + init_points random draws + n_iter guided draws), spread over
    `ceil((1 + init_points + n_iter) / q)` rounds - never more, so a
    user's configured budget continues to mean what it says and cost
    does not silently multiply by q.

    Round 0 evaluates `probe_points` (typically the user's own current/
    default hyperparameters, passed in by the caller so it is scored on
    the same footing as every other candidate - see search_bayesian()'s
    own use of this) together with `init_points` random draws, as a
    single batch. Every later round rebuilds a fresh
    `BayesianOptimization` from the accumulated TRUE observations only
    (see the module docstring for why - `save_state`/`load_state` isn't
    usable here), then calls `.suggest()`/`.register(lie)` `q` times to
    obtain `q` mutually-diverse candidates before truly evaluating any
    of them.

    Sequential domain reduction, when `bounds_transformer_factory` is
    supplied, is re-applied by keeping ONE long-lived transformer
    instance across the whole search and calling `.transform()` against
    an updated (true-observations-only) `TargetSpace` once per round -
    wrapped in try/except (reaches library internals, exactly as the
    existing `optimizer._gp.set_params(alpha=...)` call elsewhere in
    this codebase already does): a failure degrades to "no further
    domain reduction this search", one NOTE, never a crash.

    `q <= 1` must never reach this function - callers keep the existing
    `optimizer.maximize()` path for that case, byte-identically (see
    search_bayesian()'s own branch).

    Returns
    -------
    (best_x, best_score, trials) - `trials` is a list of
    `(x_array, score)` tuples in EVALUATION order, mirroring
    search_grid()/search_random()'s own trials shape (NOT bayes_opt's
    own `optimizer.res` dict-list shape, which the SERIAL code path in
    search_bayesian() still returns) - the sole caller,
    tune_model_hyperparameters(), discards this value (`_trials`), so
    the two shapes never need to agree with each other.
    """
    if BayesianOptimization is None:
        raise ImportError(
            "bayes_opt is required for batch Bayesian hyperparameter tuning "
            "(already a project dependency via models/Bayesian_optimisation.py)."
        )
    if q <= 1:
        raise ValueError("batch_bayesian_maximise: q must be > 1 - callers keep the "
                          "existing serial optimizer.maximize() path for q<=1.")

    names = [f'x{i}' for i in range(len(pbounds))]
    pbounds_dict = {n: (float(lo), float(hi)) for n, (lo, hi) in zip(names, pbounds)}

    xs_chrono: List[np.ndarray] = []
    ys_chrono: List[float] = []
    trials: List[Tuple[np.ndarray, float]] = []
    best_x, best_score = None, -math.inf

    total_budget = 1 + max(0, init_points) + max(0, n_iter)
    remaining = total_budget

    rng = np.random.default_rng(seed)
    round0 = list(probe_points or [])
    n_random0 = max(0, min(init_points, remaining - len(round0)))
    for _ in range(n_random0):
        round0.append(np.array([rng.uniform(lo, hi) for lo, hi in pbounds], dtype=float))
    round0 = round0[:remaining]

    def _space_with_history(pbounds_now):
        """Fresh BayesianOptimization instance, TRUE observations only,
        registered under the CURRENT (possibly already-narrowed) bounds -
        used both to drive .suggest() each round and to feed sequential
        domain reduction's own .transform() call afterwards."""
        opt = BayesianOptimization(f=None, pbounds=pbounds_now, random_state=seed, verbose=0,
                                    allow_duplicate_points=True)
        for x, y in zip(xs_chrono, ys_chrono):
            try:
                opt.register(params=dict(zip(names, x)), target=y)
            except Exception:
                pass  # a point that now falls outside narrowed bounds is skipped, not fatal
        return opt

    bounds_transformer = bounds_transformer_factory() if bounds_transformer_factory else None
    current_pbounds = pbounds_dict
    _sdr_initialised = False

    # bayes_opt warns (UserWarning, not an exception) whenever a
    # previously-registered TRUE point falls outside newly-narrowed
    # bounds - an expected, harmless consequence of sequential domain
    # reduction replaying history against shrinking bounds (verified in
    # the blueprint's own pre-check), not a sign anything is wrong.
    # Silenced here, scoped to this function only, so a batch search
    # with SDR enabled does not flood the run log with dozens of these
    # per round.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)

        first_round = True
        while remaining > 0:
            if first_round and round0:
                batch_xs = list(round0)
            else:
                acquisition_function = acquisition_factory() if acquisition_factory else None
                optimizer = BayesianOptimization(
                    f=None, pbounds=current_pbounds, random_state=seed, verbose=0,
                    acquisition_function=acquisition_function, allow_duplicate_points=True,
                )
                for x, y in zip(xs_chrono, ys_chrono):
                    try:
                        optimizer.register(params=dict(zip(names, x)), target=y)
                    except Exception:
                        pass
                try:
                    optimizer._gp.set_params(alpha=1e-3)
                except Exception:
                    pass

                this_q = max(1, min(q, remaining))
                batch_xs = []
                liar_targets = list(ys_chrono)
                try:
                    for _ in range(this_q):
                        suggestion = optimizer.suggest()
                        x = np.array([suggestion[n] for n in names], dtype=float)
                        batch_xs.append(x)
                        lie = _constant_liar_value(x, optimizer, liar_targets, liar)
                        try:
                            optimizer.register(params=dict(zip(names, x)), target=lie)
                        except Exception:
                            pass
                        liar_targets.append(lie)
                except Exception as exc:
                    raise RuntimeError(
                        f"batch_bayesian_maximise: suggest()/register() failed at round with "
                        f"{len(xs_chrono)} prior observation(s) ({exc!r})"
                    ) from exc

            if not batch_xs:
                break
            batch_xs = batch_xs[:remaining]
            first_round = False

            scores = objective_batch(batch_xs)
            for x, score in zip(batch_xs, scores):
                score = float(score)
                trials.append((x, score))
                xs_chrono.append(x)
                ys_chrono.append(score)
                if score > best_score:
                    best_score, best_x = score, x
            remaining -= len(batch_xs)

            if bounds_transformer is not None and remaining > 0:
                try:
                    _history_space = _space_with_history(current_pbounds)
                    if not _sdr_initialised:
                        bounds_transformer.initialize(_history_space.space)
                        _sdr_initialised = True
                    new_bounds = bounds_transformer.transform(_history_space.space)
                    current_pbounds = {n: (float(new_bounds[n][0]), float(new_bounds[n][1]))
                                        for n in names}
                except Exception as exc:
                    print(f"[parallel_search] NOTE: sequential domain reduction failed "
                          f"({exc!r}) - continuing this search without further narrowing. "
                          f"This search's own result is unaffected, only how tightly later "
                          f"rounds focus their search.")
                    bounds_transformer = None

    if best_x is None:
        best_x = np.array([])
    return best_x, best_score, trials
