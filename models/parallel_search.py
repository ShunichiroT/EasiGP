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
import time
import warnings
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from bayes_opt import BayesianOptimization
except ImportError:  # pragma: no cover - already a project dependency
    BayesianOptimization = None


# ver4-36, R2.a - noise-aware surrogate and posterior-mean incumbent.
#
# Why: on ~50 scoring rows each candidate score is noisy. The default surrogate
# (alpha=1e-3, near-interpolating) treats every observation as exact, and the
# search then returns the MAXIMUM observed score - the luckiest draw (winner's
# curse), which flows into the components and the ensemble weights. With a white
# noise term the surrogate averages neighbouring observations, and the incumbent
# becomes the evaluated point with the highest posterior MEAN. Nothing is
# re-evaluated, so this costs no fits.
_NOISE_AWARE_MIN_POINTS = 5
_NOTED_FALLBACKS: set = set()


def _noted_once(key: str, message: str) -> None:
    """Print `message` the first time `key` is seen in this process, so a block search
    that calls `posterior_means()` hundreds of times reports a given reason once."""
    if key in _NOTED_FALLBACKS:
        return
    _NOTED_FALLBACKS.add(key)
    print(message)


def noise_aware_kernel():
    """The noise-aware surrogate kernel: ``Matern(nu=2.5) + WhiteKernel(1e-2, (1e-6, 1))``.

    A new instance per call (scikit-learn clones it on fit, but a shared mutable
    object across surrogates would be a trap). Imported lazily: scikit-learn is a
    pinned project dependency (1.6.1) but this module otherwise imports only numpy
    and bayes_opt."""
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    return Matern(nu=2.5) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1.0))


def _install_noise_aware_surrogate(optimizer) -> None:
    """Give `optimizer`'s Gaussian-process surrogate the noise-aware kernel.

    Why not the public `optimizer.set_gp_params(kernel=...)`: in bayes_opt 3.0.1 it
    first runs `wrap_kernel()`, which re-instantiates the kernel from
    `kernel.get_params()` and cannot handle a SUM kernel (`TypeError: KernelOperator.
    __init__() got an unexpected keyword argument 'k1__length_scale'`, found while
    testing ver4-36). The wrap only matters for integer/categorical parameters, and
    every joint-search dimension is a plain unit-cube float, so the kernel is set on
    the surrogate directly - the same private `_gp` attribute the existing
    `alpha=1e-3` call already uses. Raises whatever scikit-learn raises (the callers
    catch it and fall back to the default surrogate with a NOTE)."""
    optimizer._gp.set_params(kernel=noise_aware_kernel())


def posterior_means(xs: Sequence[np.ndarray], ys: Sequence[float], *, seed: int = 0
                    ) -> Optional[np.ndarray]:
    """GP posterior mean at every observed point, from a surrogate that models noise.

    A separate scikit-learn ``GaussianProcessRegressor`` (kernel
    ``noise_aware_kernel()``, ``normalize_y=True``, 2 optimiser restarts,
    ``random_state=seed``) is fitted to the observations ``(xs, ys)`` and evaluated
    at ``xs`` themselves. It is independent of the optimiser's own surrogate so the
    incumbent can be chosen after the search without touching ``bayes_opt``.

    Args: ``xs`` observed points (all the same length, in the unit cube); ``ys`` their
    scores (finite); ``seed`` fixes the optimiser restarts, so the result is
    deterministic. Returns an array of ``len(xs)`` means, or ``None`` (after ONE
    NOTE per reason per process) when there are fewer than 5 points, a non-finite
    score, or the fit fails - the caller then falls back to the observed maximum.
    Never raises."""
    n_points = len(xs)
    if n_points < _NOISE_AWARE_MIN_POINTS:
        _noted_once('few', f"[parallel_search] NOTE: noise-aware incumbent needs at least "
                           f"{_NOISE_AWARE_MIN_POINTS} observed points (a search with fewer is "
                           f"summarised by its maximum observed score instead).")
        return None
    X = np.asarray([np.asarray(x, dtype=float) for x in xs], dtype=float)
    y = np.asarray(ys, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(X)):
        _noted_once('bad', "[parallel_search] NOTE: noise-aware incumbent got a non-finite or "
                           "zero-dimensional observation set - using the maximum observed score.")
        return None
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
    except ImportError:
        _noted_once('import', "[parallel_search] NOTE: noise_aware_surrogate needs scikit-learn "
                              "(pip install scikit-learn==1.6.1) - using the maximum observed score.")
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gp = GaussianProcessRegressor(kernel=noise_aware_kernel(), normalize_y=True,
                                          n_restarts_optimizer=2, random_state=int(seed))
            gp.fit(X, y)
            return np.asarray(gp.predict(X), dtype=float)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        _noted_once('fit', f"[parallel_search] NOTE: noise-aware surrogate fit failed "
                           f"({type(exc).__name__}: {exc}) - using the maximum observed score.")
        return None


def robust_incumbent(xs: Sequence[np.ndarray], ys: Sequence[float], *, seed: int = 0) -> int:
    """Index (into ``xs``) of the observed point with the highest GP posterior mean.

    Falls back to the index of the highest OBSERVED score (first one on a tie, as
    ``batch_bayesian_maximise`` does) when ``posterior_means`` cannot form a fit
    (fewer than 5 points or a fit failure - one NOTE says so). The score to report
    for the winner is always the observed ``ys[index]``.

    Args: as ``posterior_means``. Raises ``ValueError`` for an empty ``xs``."""
    if len(xs) == 0:
        raise ValueError("robust_incumbent: no observations.")
    means = posterior_means(xs, ys, seed=seed)
    if means is None:
        return int(np.argmax(np.asarray(ys, dtype=float)))
    return int(np.argmax(means))


def patience_stop(patience: int, tol: float = 1e-3
                  ) -> Optional[Callable[[List[Tuple[np.ndarray, float]]], bool]]:
    """ver4-40, R1: build the `should_stop` callable for ``batch_bayesian_maximise``
    (and for the serial patience loop of ``hyperparameter_tuning.search_bayesian``).

    Why it lives here (I3): the stopping rule used to be private to
    ``joint_tuning._patience_stop`` (ver4-32, R2). Single-model Bayesian tuning now
    needs the identical rule, so it moved to the module both searches already share;
    ``joint_tuning._patience_stop`` delegates here and its behaviour is unchanged.

    The rule: stop once more than `patience` evaluations exist and the best of the LAST
    `patience` scores does not beat the best of all EARLIER scores by more than `tol`.

    Args: `patience` evaluations without improvement (<= 0 or ``None`` disables the rule
    and returns ``None``, which every caller treats as "never stop"); `tol` minimum
    improvement that counts. The callable receives the ``(x, score)`` history of the
    current search call in evaluation order and returns ``True`` to stop. It is checked
    once per round, so a batch search can overshoot by up to q-1 evaluations. Never raises
    for a well-formed history."""
    patience = int(patience or 0)
    if patience <= 0:
        return None

    def _should_stop(history: List[Tuple[np.ndarray, float]]) -> bool:
        if len(history) <= patience:
            return False
        scores = [float(sc) for _x, sc in history]
        best_before = max(scores[:-patience])
        best_recent = max(scores[-patience:])
        return best_recent <= best_before + tol

    return _should_stop


def time_budget_stop(limit_s: Optional[float], t0: float, stats: dict
                     ) -> Optional[Callable[[List[Tuple[np.ndarray, float]]], bool]]:
    """ver4-61, R3: build a wall-clock `should_stop` callable for a tuning search.

    Why it exists: the only earlier stop rules were the candidate count and
    ``patience_stop``; a bio-prior (GAT) search whose trials take tens of minutes could
    therefore run for days (the attached run never finished task 1). This rule bounds the
    search phase without adding a new search algorithm.

    The rule: stop when ``elapsed + mean trial seconds > limit_s``, so a round that is
    expected to overrun the budget is never started. With q concurrent workers one round
    takes about one mean trial, so the same estimate holds for batch and serial searches.
    Before any trial has finished the mean is 0, so the first round always runs (the
    default probe is therefore always scored).

    Args: `limit_s` budget in seconds (``None`` or <= 0 disables the rule and returns
    ``None``, which every caller treats as "never stop"); `t0` a ``time.monotonic()`` start
    of the search phase; `stats` a mapping with ``'trials'`` (count) and ``'secs'`` (summed
    trial seconds), i.e. ``hyperparameter_tuning._TRIAL_STATS``, read live on every call.
    Returns the ``should_stop(history)`` callable. Never raises for a well-formed `stats`."""
    if not limit_s or float(limit_s) <= 0:
        return None
    limit = float(limit_s)
    announced = []

    def _should_stop(history: List[Tuple[np.ndarray, float]]) -> bool:
        trials = int(stats.get('trials', 0))
        mean = float(stats.get('secs', 0.0)) / trials if trials > 0 else 0.0
        elapsed = time.monotonic() - t0
        hit = elapsed + mean > limit
        if hit and not announced:
            announced.append(True)
            print(f"[hyperparameter_tuning] NOTE: time budget reached after "
                  f"{len(history)} candidates (limit {limit:.0f} s, elapsed {elapsed:.0f} s, "
                  f"mean trial {mean:.0f} s) - no further round is started; the winner is "
                  f"chosen from what has been scored.")
        return hit

    return _should_stop


def any_stop(*fns: Optional[Callable[[List[Tuple[np.ndarray, float]]], bool]]
             ) -> Optional[Callable[[List[Tuple[np.ndarray, float]]], bool]]:
    """ver4-61, R3: OR-compose several ``should_stop`` callables (``None`` entries ignored).

    Why: ``batch_bayesian_maximise`` and the serial stepping loop accept exactly one
    `should_stop`; the patience rule and the wall-clock rule must both be able to end a
    search. Returns ``None`` when no callable remains (callers treat it as "never stop"),
    otherwise a callable that is True as soon as any member is. Every member is evaluated
    (no short-circuit) so a stateful member sees each round."""
    active = [f for f in fns if f is not None]
    if not active:
        return None

    def _should_stop(history: List[Tuple[np.ndarray, float]]) -> bool:
        verdicts = [bool(f(history)) for f in active]
        return any(verdicts)

    return _should_stop


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
                             liar: str = 'max',
                             prior_observations: Optional[List[Tuple[np.ndarray, float]]] = None,
                             should_stop: Optional[Callable[[List[Tuple[np.ndarray, float]]],
                                                            bool]] = None,
                             noise_aware: bool = False, incumbent: str = 'max'
                             ) -> Tuple[np.ndarray, float, List[Tuple[np.ndarray, float]]]:
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

    ver4-28 R3 note: `prior_observations`, appended last so no existing
    caller (`hyperparameter_tuning.search_bayesian`,
    `Bayesian_optimisation.py`) is affected, is an optional list of
    `(x, score)` pairs already TRUELY evaluated elsewhere (e.g. a
    block-coordinate re-score of cached predictions at zero extra fit
    cost - see `models/joint_tuning.py::_block_coordinate_search`).
    When given, these are folded into `xs_chrono`/`ys_chrono` and
    `best_x`/`best_score` BEFORE round 0 is built, exactly as if they had
    been evaluated in a `round -1` - the evaluation `remaining`/`trials`
    budget accounting is untouched (they do not consume budget, since
    they cost no new fit), and they are not appended to the returned
    `trials` list (which stays TRUE-evaluation-order for this call only).
    `None` (default) reproduces the exact prior behaviour.

    ver4-32 R2 note: `should_stop`, appended last so no existing caller is
    affected, is an optional callable `should_stop(trials) -> bool`. It is
    called once after every round has been truly evaluated and registered,
    with `trials` = the (x, score) pairs evaluated by THIS call in
    evaluation order (`prior_observations` are not included). If it returns
    True the search ends early and the best point so far is returned. A
    callable that raises is reported once and then ignored for the rest of
    the search. `None` (default) keeps the exact prior budget behaviour.

    ver4-36 R2.a note: `noise_aware` and `incumbent`, appended last so no existing
    caller is affected (defaults reproduce the previous behaviour exactly).
    `noise_aware=True` gives every round's surrogate the kernel
    `Matern(nu=2.5) + WhiteKernel(1e-2, (1e-6, 1))` (`_install_noise_aware_surrogate`) INSTEAD of
    the near-interpolating `alpha=1e-3`, so noisy scores are averaged rather than
    interpolated. `incumbent='posterior_mean'` returns, as `best_x`, the evaluated point
    (a `prior_observations` entry or a truly evaluated one) with the highest
    posterior mean of a separate noise-aware GP (`robust_incumbent()`), and as
    `best_score` the OBSERVED score at that point; with fewer than 5 observations or a
    failed fit the observed maximum is returned and one NOTE says so. `trials` is
    unaffected. `'max'` (default) is the observed maximum.

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

    for prior_x, prior_y in (prior_observations or []):
        prior_x = np.asarray(prior_x, dtype=float)
        prior_y = float(prior_y)
        xs_chrono.append(prior_x)
        ys_chrono.append(prior_y)
        if prior_y > best_score:
            best_score, best_x = prior_y, prior_x

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
                if noise_aware:
                    try:
                        _install_noise_aware_surrogate(optimizer)
                    except Exception as exc:
                        _noted_once('kernel', f"[parallel_search] NOTE: could not install the "
                                              f"noise-aware surrogate kernel ({exc!r}) - using the "
                                              f"default near-interpolating surrogate for this search.")
                        try:
                            optimizer._gp.set_params(alpha=1e-3)
                        except Exception:
                            pass
                else:
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

            if should_stop is not None and remaining > 0:
                try:
                    stop_now = bool(should_stop(list(trials)))
                except Exception as exc:
                    print(f"[parallel_search] NOTE: should_stop raised {exc!r} - "
                          f"ignoring the stopping rule for the rest of this search; "
                          f"the full evaluation budget will be used.")
                    should_stop = None
                    stop_now = False
                if stop_now:
                    print(f"[parallel_search] early stop after {len(trials)} of "
                          f"{total_budget} budgeted evaluation(s) - stopping rule satisfied.")
                    break

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

    if incumbent == 'posterior_mean' and xs_chrono:
        _i = robust_incumbent(xs_chrono, ys_chrono, seed=seed)
        best_x, best_score = xs_chrono[_i], float(ys_chrono[_i])
    if best_x is None:
        best_x = np.array([])
    return best_x, best_score, trials


def async_bayesian_maximise(submit, result_of, pbounds, *, n_iter, init_points, seed, q,
                            probe_points=None, acquisition_factory=None,
                            bounds_transformer_factory=None, liar='max', should_stop=None):
    """ver4-46, R4 - ASYNCHRONOUS constant-liar Bayesian optimisation.

    `batch_bayesian_maximise()` asks for `q` candidates, evaluates all of them, and only then
    asks again, so every round lasts as long as its slowest candidate. When candidate costs
    differ by two orders of magnitude (a GAT with 128 neurons x 8 heads x 200 epochs next to one
    with 8 x 1 x 10) most workers idle for most of every round. This function keeps up to `q`
    candidates IN FLIGHT instead: the moment one finishes, its score is registered and the next
    candidate is chosen by a fresh surrogate that has been told a constant-liar value for every
    candidate that is still running.

    Like its synchronous sibling it never spawns a process and never reasons about picklability:
    the caller supplies

      `submit(x) -> future`   starts the evaluation of unit-cube point `x` (any object with the
                              `concurrent.futures.Future` interface, e.g. a loky executor's),
      `result_of(future) -> float`  returns that candidate's score once it is done (it may also
                              record timing; an exception it raises aborts the search).

    Budget accounting is identical to the synchronous path: exactly `1 + init_points + n_iter`
    candidates are started. The first `probe_points` (the user's configured point) and
    `init_points` random draws are started first, in that order; later candidates are guided.
    `should_stop(trials)` is checked after every finished candidate; once it fires no new
    candidate is started, but those already running are awaited and recorded (they are paid for
    anyway). Sequential domain reduction is re-applied after every finished candidate exactly as
    in the synchronous path.

    The candidate SEQUENCE depends on completion order (timing), so two runs with the same seed
    can differ - the price of keeping workers busy; the caller says so in the run log.

    Returns `(best_x, best_score, trials)` with `trials` = `[(x, score), ...]` in COMPLETION
    order, the same shape `batch_bayesian_maximise()` returns."""
    import concurrent.futures as _cf

    if BayesianOptimization is None:
        raise ImportError("bayes_opt is required for asynchronous Bayesian hyperparameter tuning.")
    if q <= 1:
        raise ValueError("async_bayesian_maximise: q must be > 1.")

    names = [f'x{i}' for i in range(len(pbounds))]
    current_pbounds = {n: (float(lo), float(hi)) for n, (lo, hi) in zip(names, pbounds)}
    total_budget = 1 + max(0, init_points) + max(0, n_iter)

    rng = np.random.default_rng(seed)
    queue0 = list(probe_points or [])
    for _ in range(max(0, min(init_points, total_budget - len(queue0)))):
        queue0.append(np.array([rng.uniform(lo, hi) for lo, hi in pbounds], dtype=float))
    queue0 = queue0[:total_budget]

    xs_chrono, ys_chrono, trials = [], [], []
    best_x, best_score = None, -math.inf
    in_flight = {}            # future -> (order, x)
    started = 0
    stop_requested = False
    bounds_transformer = bounds_transformer_factory() if bounds_transformer_factory else None
    sdr_initialised = False

    def _fresh_optimizer(pbounds_now):
        acquisition = acquisition_factory() if acquisition_factory else None
        opt = BayesianOptimization(f=None, pbounds=pbounds_now, random_state=seed + started,
                                   verbose=0, acquisition_function=acquisition,
                                   allow_duplicate_points=True)
        try:
            opt._gp.set_params(alpha=1e-3)
        except Exception:
            pass
        return opt

    def _next_guided():
        optimizer = _fresh_optimizer(current_pbounds)
        for x, y in zip(xs_chrono, ys_chrono):
            try:
                optimizer.register(params=dict(zip(names, x)), target=y)
            except Exception:
                pass
        liar_targets = list(ys_chrono)
        for _order, x_run in sorted(in_flight.values(), key=lambda t: t[0]):
            lie = _constant_liar_value(x_run, optimizer, liar_targets, liar)
            try:
                optimizer.register(params=dict(zip(names, x_run)), target=lie)
            except Exception:
                pass
            liar_targets.append(lie)
        suggestion = optimizer.suggest()
        return np.array([suggestion[n] for n in names], dtype=float)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        while (started < total_budget and not stop_requested) or in_flight:
            while started < total_budget and not stop_requested and len(in_flight) < q:
                x = queue0.pop(0) if queue0 else _next_guided()
                in_flight[submit(x)] = (started, x)
                started += 1
            if not in_flight:
                break
            done, _pending = _cf.wait(list(in_flight), return_when=_cf.FIRST_COMPLETED)
            for future in sorted(done, key=lambda f: in_flight[f][0]):
                _order, x = in_flight.pop(future)
                score = float(result_of(future))
                trials.append((x, score))
                xs_chrono.append(x)
                ys_chrono.append(score)
                if score > best_score:
                    best_score, best_x = score, x

            if should_stop is not None and not stop_requested and started < total_budget:
                try:
                    if bool(should_stop(list(trials))):
                        stop_requested = True
                        print(f"[parallel_search] early stop after {len(trials)} finished of "
                              f"{total_budget} budgeted evaluation(s) - stopping rule satisfied "
                              f"({len(in_flight)} still running are awaited).")
                except Exception as exc:
                    print(f"[parallel_search] NOTE: should_stop raised {exc!r} - ignoring the "
                          f"stopping rule for the rest of this search.")
                    should_stop = None

            if bounds_transformer is not None and started < total_budget and not stop_requested:
                try:
                    history = _fresh_optimizer(current_pbounds)
                    for x, y in zip(xs_chrono, ys_chrono):
                        try:
                            history.register(params=dict(zip(names, x)), target=y)
                        except Exception:
                            pass
                    if not sdr_initialised:
                        bounds_transformer.initialize(history.space)
                        sdr_initialised = True
                    new_bounds = bounds_transformer.transform(history.space)
                    current_pbounds = {n: (float(new_bounds[n][0]), float(new_bounds[n][1]))
                                       for n in names}
                except Exception as exc:
                    print(f"[parallel_search] NOTE: sequential domain reduction failed ({exc!r}) "
                          f"- continuing this search without further narrowing.")
                    bounds_transformer = None

    if best_x is None:
        best_x = np.array([])
    return best_x, best_score, trials


def suggest_batch(pbounds: Sequence[Tuple[float, float]],
                  observations: Sequence[Tuple[np.ndarray, float]], *,
                  q: int, seed: int, liar: str = 'max',
                  acquisition_factory: Optional[Callable[[], object]] = None,
                  noise_aware: bool = False
                  ) -> List[np.ndarray]:
    """Propose ``q`` mutually diverse points to evaluate next (ver4-34, R1.c).

    This is the constant-liar SUGGESTION step of ``batch_bayesian_maximise``
    on its own: a fresh ``BayesianOptimization`` is built, every TRUE
    observation is registered, and then ``q`` times a point is suggested and a
    fabricated ("lied") target is registered at it so the next suggestion
    moves elsewhere. Nothing is evaluated here - the caller evaluates the
    returned points however it likes and calls this again with the enlarged
    observation list.

    Why a separate function: ``joint_tuning._parallel_library_search`` asks
    several models for candidates in the SAME round and fits them all in one
    parallel batch, so it needs the suggestion step without
    ``batch_bayesian_maximise``'s own evaluate-and-loop driver.
    ``batch_bayesian_maximise`` is deliberately left unchanged (the same logic
    is mirrored here rather than shared, so no existing search can change
    behaviour).

    Args:
        pbounds: ``[(low, high), ...]`` per dimension (the joint search always
            passes the unit cube, ``(0.0, 1.0)``).
        observations: ``[(x, score), ...]`` already evaluated points. Scores
            must be finite (the caller never registers a failed fit).
        q: number of points to return; ``q <= 0`` returns ``[]``.
        seed: seeds both the random fallback and the optimiser, so a fixed
            ``(observations, seed)`` always gives the same points, in any
            process.
        liar: ``'max'`` (default), ``'mean'`` or ``'believer'`` - see
            ``_constant_liar_value``.
        acquisition_factory: optional zero-argument callable returning a
            ``bayes_opt`` acquisition function (``joint_tuning`` passes
            Expected Improvement, exactly as the monolithic search does).
        noise_aware: ver4-36 R2.a. ``True`` gives the surrogate the kernel
            ``Matern(nu=2.5) + WhiteKernel(1e-2, (1e-6, 1))`` instead of the
            near-interpolating ``alpha=1e-3`` (see ``noise_aware_kernel``);
            ``False`` (default) is the previous behaviour.

    Returns ``q`` arrays of length ``len(pbounds)``. With fewer than 2
    observations a Gaussian-process surrogate cannot be fitted, so ``q``
    uniform-random points from ``np.random.default_rng(seed)`` are returned.

    Raises ``ImportError`` when ``bayes_opt`` is missing and a surrogate is
    needed, and ``RuntimeError`` (wrapping the library error) when
    ``suggest()``/``register()`` fails - the same contract as
    ``batch_bayesian_maximise``, whose caller treats it as "fall back"."""
    q = int(q)
    if q <= 0:
        return []
    bounds = [(float(lo), float(hi)) for lo, hi in pbounds]
    if len(observations) < 2:
        rng = np.random.default_rng(seed)
        return [np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)
                for _ in range(q)]
    if BayesianOptimization is None:
        raise ImportError(
            "bayes_opt is required for batch Bayesian suggestions (already a project "
            "dependency via models/Bayesian_optimisation.py)."
        )
    names = [f'x{i}' for i in range(len(bounds))]
    optimizer = BayesianOptimization(
        f=None, pbounds={n: b for n, b in zip(names, bounds)}, random_state=seed, verbose=0,
        acquisition_function=acquisition_factory() if acquisition_factory else None,
        allow_duplicate_points=True,
    )
    xs_seen = [np.asarray(x, dtype=float) for x, _y in observations]
    ys_seen = [float(y) for _x, y in observations]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        for x, y in zip(xs_seen, ys_seen):
            try:
                optimizer.register(params=dict(zip(names, x)), target=y)
            except (KeyError, ValueError, TypeError, RuntimeError) as exc:
                print(f"[parallel_search] NOTE: suggest_batch skipped one observation that "
                      f"bayes_opt refused to register ({exc!r}) - the other observations are "
                      f"still used")
        if noise_aware:
            try:
                _install_noise_aware_surrogate(optimizer)
            except (AttributeError, ValueError, TypeError, ImportError) as exc:
                print(f"[parallel_search] NOTE: suggest_batch could not install the noise-aware "
                      f"surrogate kernel ({exc!r}) - using the library default noise term")
        else:
            try:
                optimizer._gp.set_params(alpha=1e-3)
            except (AttributeError, ValueError, TypeError) as exc:
                # private attribute; same guard as batch_bayesian_maximise, but reported
                print(f"[parallel_search] NOTE: suggest_batch could not set the surrogate's alpha "
                      f"({exc!r}) - using the library default noise term")
        batch: List[np.ndarray] = []
        liar_targets = list(ys_seen)
        try:
            for _ in range(q):
                suggestion = optimizer.suggest()
                x = np.array([suggestion[n] for n in names], dtype=float)
                batch.append(x)
                lie = _constant_liar_value(x, optimizer, liar_targets, liar)
                try:
                    optimizer.register(params=dict(zip(names, x)), target=lie)
                except (KeyError, ValueError, TypeError, RuntimeError) as exc:
                    print(f"[parallel_search] NOTE: suggest_batch could not register a "
                          f"constant-liar point ({exc!r}) - the next suggestion may repeat it")
                liar_targets.append(lie)
        except Exception as exc:
            raise RuntimeError(
                f"suggest_batch: suggest()/register() failed with {len(xs_seen)} "
                f"observation(s) ({exc!r})"
            ) from exc
    return batch
