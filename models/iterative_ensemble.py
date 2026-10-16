"""
models/iterative_ensemble.py
============================
ver4-57 (design blueprint ``EasiGP_ver4-57_Design_Blueprint.md``, R2 / R5): the opt-in
**Iterative ensemble**.

What it does
------------
After every model has been fitted once (Step 9) the members' marker effects are combined into
one consensus ranking of the markers (computed once, no extra fit). The members that read the
shared marker pool (everything except the bio-prior GAT models, invariant I6) are then refitted
on smaller and smaller marker subsets. A round is accepted only when the equal-weight ensemble
of the refitted members improves on the best round so far by more than the bootstrap noise of
that gain on the validation rows; the loop stops at the first rejected round. Round 0 is the
plain mean of the Step 9 prediction columns, i.e. the naive ensemble, so the result differs from
the naive ensemble only after a clear validation gain.

Why this design (blueprint Appendix A, runs 4 and 5)
----------------------------------------------------
* Sharing marker information across rounds has real headroom (hindsight +0.015 to +0.029 r),
  but ~60 validation rows cannot reliably pick the best round, so the realised change is about
  -0.005 to +0.004 r (within 1 SE): non-inferior to the naive ensemble, not a proven gain.
  Hence opt-in, strict acceptance, per-round diagnostics and a real-data gate.
* One-shot ranking from the Step 9 effects matched re-ranking every round, so every refit runs
  with explainability OFF (the dominant cost) and there are NO out-of-fold fits, NO fitted
  ensemble weights and NO phenotype-distillation channel. Early stopping removes about 60 % of
  the refits at no measurable accuracy cost.
* LD-aware allocation (d'Hondt shares across LD windows) and residual re-entry cost zero fits;
  they raise the hindsight headroom where a QTL's credit is split between tags, and they stop a
  marker that matters only in a smaller set from being lost for good.
* Dropped markers are never shown with a zero effect: the reported effect row keeps the Step 9
  consensus for them.

Contract
--------
* Pure Python / NumPy / pandas; imports only ``models.*`` and ``pipeline_utils`` helpers, never
  ``genomic_prediction`` (I2). Fits go through the caller-supplied ``runner`` (a
  ``ModelTrialRunner``) and ``models.joint_tuning._run_units`` (I3, I10).
* I7: validation rows are used for acceptance only. No decision reads a test phenotype. The test
  frames are handed to the runner unchanged because every model entry point scores its own test
  split (their test scores are discarded here).
* I9: bootstrap seeds are ``seed + 7919 * round``.
* Never raises on a failed member refit (that member keeps its last good column); raises
  ``ValueError`` only on contract violations (fewer than 2 members, mismatched lengths).

ver4-68: ``mode='two_phase'`` (blueprint ``EasiGP_ver4-68_Design_Blueprint.md``, R1-R3)
-----------------------------------------------------------------------------------------
A second, opt-in mode that replaces the marker-subset loop by a *snapshot library*. It is the
consolidation of the ver4-63 -> ver4-67 investigation, and every choice below is one of its
design rules (D1-D8):

* D1 - nothing is selected on the ~60 validation rows between rounds, schedules or settings: the
  output is a POOL of fits. Validation enters only the K member weights ``w0``.
* D2 - no member that has seen ensemble information is ever deployed as such. Phase 1 (target
  distillation towards the other members' out-of-fold predictions) therefore adds its refits as
  EXTRA pooled snapshots next to the untouched Step 9 fit, never in place of it.
* D3 - no learned encoder and no privileged input channel: information reaches a member only
  through its target column (phase 1) or its marker view ``member_frames`` (phase 2).
* D4 - phase 1 defaults to ONE round, built from the clean round-0 out-of-fold predictions;
  later rounds rebuild targets from predictions that were themselves fitted on them (a measured
  target leak, see ``p1_rounds`` in ``pipeline_utils``).
* D5 - no free weights over the library: ``w0`` (K weights, one per member) comes from the clean
  round-0 out-of-fold columns plus the validation rows and every member weighs its own
  snapshots equally (a factorised weighting).
* D6 - every extra fit runs with explainability off; the effect and interaction rings of the
  row are the w0-weighted Step 9 consensus (built by the caller).
* D7 - phase 2 diversifies at member level (CMA: every member keeps a shared core of the
  consensus-relevant markers plus the markers only it favours, chained from its own previous
  snapshot) and contracts at aggregation level (the pooled mean).
* D8 - the mode is opt-in and labelled experimental until it beats its paired controls on real
  tasks (``tools/iterative_gate.py --baseline``).
Wall time (R2): the fits are grouped into dependency WAVES; each wave is ONE parallel
``_run_units`` call (units carry their own test frame, ``_FitUnit.test_frame``), and the
out-of-fold units of rrBLUP / GBLUP / BayesB / RKHS use the parity-guarded 1-row test stub.
``oof_engine='exact'`` (R3) answers the rrBLUP / RKHS out-of-fold units with the closed-form
kernel engine. Unit results do not depend on the grouping, so waves change wall time only.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from models.ensemble_regularization import cv_shrunk_simplex_weights, shrinkage_path_weights
from models.hyperparameter_tuning import _disable_explainability
from models.joint_tuning import _FitUnit, _run_units
from pipeline_utils import (_ITER_ENS_DEFAULTS, iter_ens_tier_of, iter_ens_wave_groups,
                            safe_regression_metrics)

_SPLITS = ('train', 'valid', 'test')
_SEED_STRIDE = 7919   # I9: round r bootstraps with seed + _SEED_STRIDE * r
_OOF_SEED_OFFSET = 104729   # I9 (ver4-68): the positional out-of-fold split uses seed + this


@dataclass(frozen=True, eq=False)
class IterMember:
    """One Step 9 model fit, as the iterative ensemble needs it.

    ``name`` is the output label (may carry a tuning-algorithm suffix), ``base`` the model
    name the runner dispatches on, ``params`` the hyperparameters Step 9 used. ``train``,
    ``valid`` and ``test`` are the Step 9 frames (markers first, phenotype last, I4). ``pred``
    holds the Step 9 predictions per split. ``effect`` / ``interaction`` are the Step 9 marker
    effects (1-D, one value per train marker) and k=2 interaction table, either may be None.
    ``marker_channel`` is False for bio-prior models (I6): they are never subset or refitted."""
    name: str
    base: str
    params: list
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    pred: Dict[str, np.ndarray]
    effect: Optional[np.ndarray]
    interaction: Optional[pd.DataFrame]
    marker_channel: bool


@dataclass
class IterResult:
    """Outcome of :func:`run_iterative_ensemble`.

    ``pred`` maps 'train' / 'valid' / 'test' to the equal-weight ensemble of the selected
    round; it is EMPTY (``{}``) when no row should be written (see ``notes``). ``members`` are
    the labels that were averaged, ``best_round`` the selected round (0 = naive ensemble).
    ``kept`` is a bool mask over the shared pool's markers for the selected round (None when
    round 0 was selected). ``effect`` is a pool-indexed consensus Series (Step 9 consensus for
    every marker; with ``explain_selected`` and ``best_round > 0`` the refit consensus on the
    kept markers, ``effect_refit`` True). ``interactions`` is always None here: the caller
    builds the interaction ring from the Step 9 tables with
    ``weighted_ensemble_interactions`` (single authority, I3). ``rounds`` has one dict per round
    (round 0 first), ``round_test_pred`` the matching ensemble test predictions (diagnostics
    only, never used for a decision), ``n_fits`` the extra model fits made, ``notes`` the
    NOTE messages the caller should print."""
    pred: Dict[str, np.ndarray]
    members: List[str]
    best_round: int
    kept: Optional[np.ndarray]
    effect: Optional[pd.Series]
    interactions: Optional[pd.DataFrame]
    rounds: List[dict]
    round_test_pred: List[np.ndarray]
    n_fits: int
    notes: List[str]
    effect_refit: bool = False
    # ver4-68 (R1), appended with defaults so every pre-ver4-68 constructor call is unchanged:
    # `member_weights` maps each member label to the share its predictions carry in `pred`
    # (sums to 1; None in mode='subset', where every member weighs 1/K). In mode='two_phase'
    # `rounds` has one dict per snapshot round (round 0 = the Step 9 columns, then the phase-1
    # and phase-2 snapshots, each with `phase` in {'r0', 'p1', 'p2'} and `selected` True) and
    # `best_round` is the last round. `diag` carries the per-group facts the caller logs: tier,
    # weight source ('oof_cv' | 'valid_rows' | 'equal'), alpha*, waves planned / run, rounds done,
    # seconds (None in mode='subset').
    member_weights: Optional[Dict[str, float]] = None
    diag: Optional[dict] = None
    # ver4-69 (R1): `attribution` (two_phase only) maps a label to a TEST prediction vector of an
    # intermediate ensemble of the same task, so the caller can report where the final row gains
    # or loses against the naive ensemble: 'step9_equal' (the naive ensemble itself),
    # 'step9_w0' (the Step 9 columns at the member weights w0), 'pool_equal' (all library
    # snapshots, members weighted equally) and 'final'. Reporting only, never a decision.
    attribution: Optional[Dict[str, np.ndarray]] = None


# ----------------------------------------------------------------------------------------- #
# Pure helpers
# ----------------------------------------------------------------------------------------- #

def _usable_effect(effect, n_markers: int) -> Optional[np.ndarray]:
    """`effect` as a float vector when it is finite, non-zero and `n_markers` long, else None."""
    if effect is None:
        return None
    try:
        vec = np.asarray(effect, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if vec.shape[0] != n_markers or not np.all(np.isfinite(vec)):
        return None
    if float(np.abs(vec).sum()) <= 0.0:
        return None
    return vec


def consensus_relevance(effects: Mapping[str, Optional[np.ndarray]], n_markers: int,
                        rank_method: str = 'mean_effect') -> Optional[np.ndarray]:
    """Consensus marker relevance from the members' Step 9 effects (zero fits).

    Keeps every effect vector that is finite, non-zero and `n_markers` long. With
    ``rank_method='mean_effect'`` (default) each is L1-normalised in absolute value and the
    vectors are averaged with equal weight; with ``'rank_average'`` the mean of each marker's
    rank (0..1, ties averaged) across members is returned instead. Returns None when fewer
    than 2 members have a usable effect (the caller then runs no iteration).

    Why: the members' effects live on different scales, so only normalised, equal-weight
    averaging is a fair consensus (the same rule the naive ensemble's effect ring uses)."""
    usable = [v for v in (_usable_effect(e, n_markers) for e in effects.values()) if v is not None]
    if len(usable) < 2:
        return None
    if rank_method == 'rank_average':
        ranks = [pd.Series(np.abs(v)).rank(method='average').to_numpy() / n_markers
                 for v in usable]
        return np.mean(ranks, axis=0)
    normalised = [np.abs(v) / np.abs(v).sum() for v in usable]
    return np.mean(normalised, axis=0)


def protected_markers(tables: Sequence[Optional[pd.DataFrame]], names: Sequence[str],
                      top_frac: float) -> np.ndarray:
    """Bool mask over `names` of the markers in the strongest shared interaction pairs.

    `tables` are the members' Step 9 k=2 interaction tables (columns ``marker1``, ``marker2``,
    ``value``; None or empty tables are skipped). Each table's absolute values are
    L1-normalised, the per-pair values are averaged over the tables that exist (a pair absent
    from a table counts 0 there), and the markers of the top ``ceil(top_frac * n_pairs)`` pairs
    (at least one pair when ``top_frac > 0``) are marked. Pairs are canonical (unordered).
    Returns all-False when there is nothing usable or ``top_frac <= 0``.

    Why: a marker that matters only through an interaction has a small main effect and would
    otherwise be dropped first."""
    mask = np.zeros(len(names), dtype=bool)
    if top_frac is None or top_frac <= 0:
        return mask
    pair_sum: Dict[Tuple[str, str], float] = {}
    n_tables = 0
    for table in tables:
        if table is None or table.shape[0] == 0:
            continue
        if not {'marker1', 'marker2', 'value'}.issubset(table.columns):
            continue
        vals = np.abs(pd.to_numeric(table['value'], errors='coerce').to_numpy(dtype=float))
        vals = np.where(np.isfinite(vals), vals, 0.0)
        total = float(vals.sum())
        if total <= 0.0:
            continue
        n_tables += 1
        for m1, m2, val in zip(table['marker1'].astype(str), table['marker2'].astype(str),
                               vals / total):
            key = (m1, m2) if m1 <= m2 else (m2, m1)
            pair_sum[key] = pair_sum.get(key, 0.0) + float(val)
    if n_tables == 0 or not pair_sum:
        return mask
    pairs = sorted(pair_sum.items(), key=lambda kv: (-kv[1], kv[0]))
    n_top = max(1, int(math.ceil(top_frac * len(pairs))))
    index = {str(n): i for i, n in enumerate(names)}
    for (m1, m2), _ in pairs[:n_top]:
        for marker in (m1, m2):
            pos = index.get(marker)
            if pos is not None:
                mask[pos] = True
    return mask


def window_ids(names: Sequence[str], ld_window: int,
               chrom: Optional[Mapping[str, str]]) -> np.ndarray:
    """Window id per marker column: consecutive blocks of `ld_window` columns.

    The count restarts at every chromosome change when `chrom` (marker name -> chromosome) is
    given and knows both neighbours; markers missing from `chrom` never start a new window by
    themselves. Without `chrom` the column order is assumed genomic (CSV input has no marker
    positions in the engine). Returns an int array of length ``len(names)``."""
    ids = np.zeros(len(names), dtype=int)
    window = 0
    count = 0
    prev_chrom = None
    for pos, name in enumerate(names):
        cur_chrom = chrom.get(name) if chrom is not None else None
        changed = (cur_chrom is not None and prev_chrom is not None
                   and str(cur_chrom) != str(prev_chrom))
        if pos > 0 and (count >= ld_window or changed):
            window += 1
            count = 0
        ids[pos] = window
        count += 1
        if cur_chrom is not None:
            prev_chrom = cur_chrom
    return ids


def marker_order(rel: np.ndarray, window_id: Optional[np.ndarray]) -> np.ndarray:
    """Total order of the markers, most important first (a permutation of 0..M-1).

    Without windows: descending `rel`, ties by column position. With windows (LD-aware, a
    d'Hondt allocation): ``score_j = W(window_j) / k_j`` where ``W`` is the sum of `rel` inside
    the window and ``k_j`` the 1-based rank of marker j inside its window; ordered by descending
    score, ties by column position. Slots therefore go to windows in proportion to their
    strength, and a strong window gets its best tag before a weak window gets its best tag.

    Why: when one QTL's credit is split over several tags, each tag looks weak and plain
    ranking can drop the whole region; the window sum keeps at least its best tag."""
    rel = np.asarray(rel, dtype=float)
    n = rel.shape[0]
    position = np.arange(n)
    if window_id is None:
        return np.lexsort((position, -rel))
    window_id = np.asarray(window_id)
    if window_id.shape[0] != n:
        raise ValueError(f"marker_order: window_id has {window_id.shape[0]} entries, rel has {n}")
    score = np.empty(n, dtype=float)
    for win in np.unique(window_id):
        members = np.flatnonzero(window_id == win)
        total = float(rel[members].sum())
        within = members[np.lexsort((members, -rel[members]))]   # best first, ties by position
        score[within] = total / (np.arange(within.shape[0]) + 1.0)
    return np.lexsort((position, -score))


def round_budget(r: int, keep: float, min_keep: int, n_markers: int) -> int:
    """Base marker budget of round `r`: ``max(min_keep, ceil(keep**r * n))``, at most `n`."""
    return int(min(n_markers, max(int(min_keep), int(math.ceil((keep ** r) * n_markers)))))


def kept_mask(order: np.ndarray, r: int, keep: float, min_keep: int,
              protect: np.ndarray) -> np.ndarray:
    """Bool mask of the base set of round `r`: the first ``round_budget`` markers of `order`,
    OR the `protect` mask. Base sets of successive rounds are nested (fixed order, shrinking
    budget)."""
    n = int(np.asarray(order).shape[0])
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(order)[:round_budget(r, keep, min_keep, n)]] = True
    if protect is not None and np.asarray(protect).shape[0] == n:
        mask |= np.asarray(protect, dtype=bool)
    return mask


def reentry_markers(X_train: np.ndarray, resid: np.ndarray, kept: np.ndarray, share: float,
                    budget: int) -> np.ndarray:
    """Bool mask of dropped markers to put back: the ``ceil(share * budget)`` dropped markers
    most associated with the training residual `resid`.

    Score = |centred x_j . u| / ||centred x_j|| over the TRAINING rows only (I7); one
    matrix-vector product, no model fit. Returns an all-False mask when ``share == 0``, nothing
    is dropped, the residual is non-finite / has zero variance, or the shapes disagree (the
    caller records the reason). Markers in `kept` are never returned.

    Why: a marker that matters only once the set is smaller (LD split, interaction-only,
    dilution in the full set) shows up as unexplained training residual."""
    kept = np.asarray(kept, dtype=bool)
    out = np.zeros(kept.shape[0], dtype=bool)
    if share is None or share <= 0:
        return out
    X = np.asarray(X_train, dtype=float)
    u = np.asarray(resid, dtype=float).ravel()
    if X.ndim != 2 or X.shape[0] != u.shape[0] or X.shape[1] != kept.shape[0]:
        return out
    if not np.all(np.isfinite(u)) or float(np.std(u)) <= 0.0:
        return out
    dropped = np.flatnonzero(~kept)
    n_back = min(int(math.ceil(share * budget)), dropped.shape[0])
    if n_back <= 0:
        return out
    n_rows = X.shape[0]
    u_c = u - u.mean()
    cross = np.abs(X.T @ u_c)                     # centred x . u_c  ==  x . u_c  since sum(u_c)=0
    col_mean = X.mean(axis=0)
    centred_ss = np.einsum('ij,ij->j', X, X) - n_rows * col_mean ** 2
    norm = np.sqrt(np.clip(centred_ss, 0.0, None))
    score = np.where(norm > 0.0, cross / np.where(norm > 0.0, norm, 1.0), 0.0)
    score = np.where(np.isfinite(score), score, 0.0)[dropped]
    best = dropped[np.lexsort((dropped, -score))][:n_back]
    out[best] = True
    return out


def _score(y: np.ndarray, p: np.ndarray, kwargs: Optional[dict]) -> float:
    """Pearson r of `p` against `y` through the single shared scorer (weighted when asked)."""
    return float(safe_regression_metrics(y, p, **(kwargs or {}))[0])


def accept_round(cand, best, y_valid, *, rule: str, margin: float, reps: int, seed: int,
                 score_kwargs: Optional[dict] = None) -> Tuple[bool, float, float]:
    """Decide whether candidate predictions `cand` beat `best` on the validation rows.

    The observed gain is ``r(y, cand) - r(y, best)`` (weighted Pearson when `score_kwargs`
    carries BLUE weights ``w`` / ``W``). ``rule='one_se'``: accept when the gain exceeds the SD
    of ``reps`` paired bootstrap gains (the same rows are resampled for both columns; weights
    are resampled with them). ``rule='margin'``: accept when the gain exceeds `margin`.
    `seed` is used as given (the caller passes ``seed + 7919 * round``, I9).

    Returns ``(accepted, gain, sd)``; ``sd`` is NaN under ``'margin'``. A non-finite gain, or a
    non-finite SD under ``'one_se'``, is never accepted. Never raises on degenerate input."""
    y = np.asarray(y_valid, dtype=float).ravel()
    c = np.asarray(cand, dtype=float).ravel()
    b = np.asarray(best, dtype=float).ravel()
    gain = _score(y, c, score_kwargs) - _score(y, b, score_kwargs)
    if not np.isfinite(gain):
        return False, float('nan'), float('nan')
    if rule == 'margin':
        return bool(gain > margin), float(gain), float('nan')
    n = y.shape[0]
    rng = np.random.default_rng(int(seed))
    gains = []
    for _ in range(int(reps)):
        idx = rng.integers(0, n, n)
        kw = None
        if score_kwargs:
            kw = {}
            for key, val in score_kwargs.items():
                arr = np.asarray(val)
                if key == 'W' and arr.ndim == 2:
                    kw[key] = arr[np.ix_(idx, idx)]
                elif arr.ndim == 1 and arr.shape[0] == n:
                    kw[key] = arr[idx]
                else:
                    kw[key] = val
        g = _score(y[idx], c[idx], kw) - _score(y[idx], b[idx], kw)
        if np.isfinite(g):
            gains.append(g)
    if len(gains) < 2:
        return False, float(gain), float('nan')
    sd = float(np.std(gains, ddof=1))
    return bool(np.isfinite(sd) and gain > sd), float(gain), sd


def member_frames(m: IterMember, kept: Optional[np.ndarray]
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The (train, valid, test) frames `m` is refitted on: for a marker-channel member the kept
    marker columns plus the phenotype column (``iloc`` over positions, so markers-then-phenotype
    order and row alignment hold, I4; an empty validation frame stays empty); every other
    member (bio-prior, I6) and ``kept=None`` return the frames unchanged."""
    if not m.marker_channel or kept is None:
        return m.train, m.valid, m.test
    cols = list(np.flatnonzero(np.asarray(kept, dtype=bool))) + [-1]
    return m.train.iloc[:, cols], m.valid.iloc[:, cols], m.test.iloc[:, cols]


# ----------------------------------------------------------------------------------------- #
# ver4-68: pure helpers of the two-phase mode
# ----------------------------------------------------------------------------------------- #

def prediction_relevance(X: np.ndarray, p: np.ndarray) -> Optional[np.ndarray]:
    """Marker relevance of one prediction column: the absolute Pearson correlation of every
    marker column of `X` with `p`, L1-normalised to sum 1 (a constant marker scores 0).

    Returns None when the relevance is degenerate: `p` is constant or non-finite, `X` is not a
    finite 2-D array with ``len(p)`` rows (and at least 3), or every correlation is 0.

    Why: phase 2 gives each member the markers ITS OWN fitted values lean on; the correlation of a
    marker with the member's training predictions needs no explainability fit (design rule D6).
    Uses the training rows only (I7)."""
    X = np.asarray(X, dtype=float)
    p = np.asarray(p, dtype=float).ravel()
    if X.ndim != 2 or X.shape[0] != p.shape[0] or X.shape[0] < 3:
        return None
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(X)):
        return None
    p_c = p - p.mean()
    p_ss = float(p_c @ p_c)
    if p_ss <= 0.0:
        return None
    X_c = X - X.mean(axis=0)
    x_ss = np.einsum('ij,ij->j', X_c, X_c)
    denom = np.sqrt(x_ss * p_ss)
    cross = np.abs(X_c.T @ p_c)
    rel = np.where(denom > 0.0, cross / np.where(denom > 0.0, denom, 1.0), 0.0)
    total = float(rel.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    return rel / total


def cma_masks(rels: Sequence[np.ndarray], core: float, unique: float) -> List[np.ndarray]:
    """Coordinated marker allocation: one bool mask over the markers per member.

    `rels` are the members' relevance vectors (equal length M, >= 2 members). The CORE is the
    ``ceil(core * M)`` markers with the highest mean relevance over all members - the same set
    for every member. Each member's OWN share is the ``ceil(unique * M)`` non-core markers where
    its relevance exceeds the mean relevance of the OTHER members the most. Mask = core + own
    share. Ordering is a stable argsort (ties by column position), so the result is
    deterministic. The two sizes are clipped so core + own never exceeds M.

    Why: every member sees the markers the group agrees on, and the rest of its view is what it
    alone relies on, so the refitted members differ from each other (diversity) without any
    member seeing the ensemble's predictions (design rules D2, D7).

    Raises ValueError when fewer than 2 vectors are given or their lengths differ."""
    rel = [np.asarray(r, dtype=float).ravel() for r in rels]
    if len(rel) < 2:
        raise ValueError(f"cma_masks needs at least 2 relevance vectors, got {len(rel)}")
    n_markers = rel[0].shape[0]
    if any(r.shape[0] != n_markers for r in rel):
        raise ValueError("cma_masks: the relevance vectors have different lengths "
                         f"({sorted({r.shape[0] for r in rel})})")
    stack = np.vstack(rel)
    n_core = min(n_markers, int(math.ceil(core * n_markers)))
    n_own = min(n_markers - n_core, int(math.ceil(unique * n_markers)))
    consensus = stack.mean(axis=0)
    core_idx = np.argsort(-consensus, kind='stable')[:n_core]
    masks: List[np.ndarray] = []
    for i in range(stack.shape[0]):
        others = np.delete(stack, i, axis=0).mean(axis=0)
        score = stack[i] - others
        score[core_idx] = -np.inf                   # the core is never counted as "own"
        own_idx = np.argsort(-score, kind='stable')[:n_own]
        mask = np.zeros(n_markers, dtype=bool)
        mask[core_idx] = True
        mask[own_idx] = True
        masks.append(mask)
    return masks


def _positional_folds(n: int, k: int, seed: int) -> List[np.ndarray]:
    """`k` disjoint, near-equal arrays of ROW POSITIONS 0..n-1 (a seeded permutation, split with
    ``np.array_split``). `seed` is used as given (the caller passes ``seed + 104729``, I9).

    Why positions, not row labels: the train frame's index may be shuffled or non-unique (a
    ``.loc``-based split would then depend on it, I4); positions make the out-of-fold columns
    independent of the index."""
    return [np.asarray(f, dtype=int)
            for f in np.array_split(np.random.default_rng(int(seed)).permutation(int(n)), int(k))]


def _oof_possible(settings: dict, members: Sequence[IterMember]) -> Tuple[bool, List[str]]:
    """``(possible, reasons)``: whether every member can be fitted out of fold.

    Not possible when a member is a bio-prior model (its runner reads the fold from its own
    context and ignores the frames it is given, so an "out-of-fold" fit would be in-sample - I6),
    or when the training split has fewer than ``10 * oof_folds`` rows. `reasons` says why, for the
    log. Why here: the caller then plans no out-of-fold fits at all instead of paying for fits
    that are certain to be discarded."""
    folds = int(settings.get('oof_folds', _ITER_ENS_DEFAULTS['oof_folds']))
    reasons: List[str] = []
    bio = [m.name for m in members if not m.marker_channel]
    if bio:
        reasons.append(f"{bio!r} is/are bio-prior (never fitted out of fold, I6)")
    n_train = members[0].train.shape[0] if members else 0
    if n_train < 10 * folds:
        reasons.append(f"the training split has {n_train} row(s), fewer than 10 x oof_folds "
                       f"({10 * folds})")
    return (not reasons), reasons


def _pool_refit_members(members: Sequence[IterMember]) -> List[IterMember]:
    """The marker-channel members whose frames are the shared marker pool (the first
    marker-channel member's columns): the ones phase 1 and phase 2 may refit. The same rule
    ``run_iterative_ensemble`` applies before it dispatches."""
    marker_members = [m for m in members if m.marker_channel]
    if not marker_members:
        return []
    pool_cols = list(marker_members[0].train.columns[:-1])
    return [m for m in marker_members
            if list(m.train.columns[:-1]) == pool_cols and list(m.valid.columns[:-1]) == pool_cols
            and list(m.test.columns[:-1]) == pool_cols]


def two_phase_wave_sizes(members: Sequence[IterMember], settings: dict) -> List[int]:
    """Number of fit units in each wave of the two-phase plan for these members (empty list when
    no member can be refitted). ``GP()`` uses ``max(...)`` of it as the parallel width of Step 11c.
    Pure."""
    members = list(members)
    refit_members = _pool_refit_members(members)
    if not members or not refit_members:
        return []
    possible, _reasons = _oof_possible(settings, members)
    groups = iter_ens_wave_groups(settings, len(members), len(refit_members), possible)
    return [int(sum(count for _gid, count in wave)) for wave in groups]


# ----------------------------------------------------------------------------------------- #
# Loop helpers
# ----------------------------------------------------------------------------------------- #

def _equal_mean(columns: Mapping[str, Dict[str, np.ndarray]], split: str) -> np.ndarray:
    """Equal-weight mean of the members' `split` columns, NaN skipped (pandas ``mean`` - the same
    arithmetic as the naive ensemble in ``models/ensemble.py``, so round 0 equals it)."""
    frame = pd.DataFrame({name: cols[split] for name, cols in columns.items()})
    return frame.mean(axis=1).to_numpy(dtype=float)


def _check_member(m: IterMember) -> None:
    """Raise ValueError when a member's prediction lengths disagree with its own frames."""
    for split, frame in (('train', m.train), ('valid', m.valid), ('test', m.test)):
        arr = m.pred.get(split)
        if arr is None or len(arr) != frame.shape[0]:
            raise ValueError(
                f"iterative ensemble: member {m.name!r} has "
                f"{None if arr is None else len(arr)} {split} prediction(s) for a {split} frame "
                f"of {frame.shape[0]} row(s)")


def _refit_columns(units: List[_FitUnit], members: List[IterMember], runner, RESULT_NAME,
                   test: pd.DataFrame, runner_ctx: dict) -> List[Optional[dict]]:
    """Run one round's refit units in a single ``_run_units`` call and return, per member, a
    dict of validated float columns (plus the raw runner dict under ``'raw'``) or None when
    that member's fit failed or returned columns of the wrong length (a failed unit never
    stops the others: ``stop_on_failure=False``)."""
    results = _run_units(units, runner=runner, RESULT_NAME=RESULT_NAME, test=test,
                         stop_on_failure=False, **(runner_ctx or {}))
    out: List[Optional[dict]] = []
    for m, unit, res in zip(members, units, results):
        if not isinstance(res, dict):
            out.append(None)
            continue
        cols = {}
        ok = True
        for split in _SPLITS:
            arr = res.get(f'predicted_{split}')
            if arr is None:
                ok = False
                break
            vec = np.asarray(arr, dtype=float).ravel()
            # ver4-68 (R2): a unit with its own test frame (`_FitUnit.test_frame`) is scored
            # against it; every ver4-57 unit has None here, so `test` is used as before.
            test_ref = unit.test_frame if unit.test_frame is not None else test
            expected = {'train': unit.train, 'valid': unit.valid, 'test': test_ref}[split].shape[0]
            if vec.shape[0] != expected:
                ok = False
                break
            cols[split] = vec
        if ok:
            cols['raw'] = res
            out.append(cols)
        else:
            out.append(None)
    return out


def _refit_effect_consensus(res_by_member: Sequence[Optional[dict]], n_kept: int
                            ) -> Optional[np.ndarray]:
    """Mean of the L1-normalised |effect| of the refitted members over the kept markers, or None
    when fewer than one member returned a usable (finite, non-zero, `n_kept` long) effect."""
    vecs = []
    for cols in res_by_member:
        if cols is None:
            continue
        eff = cols['raw'].get('effect')
        if eff is None:
            continue
        try:
            vec = np.asarray(eff, dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if vec.shape[0] != n_kept or not np.all(np.isfinite(vec)):
            continue
        total = float(np.abs(vec).sum())
        if total <= 0.0:
            continue
        vecs.append(np.abs(vec) / total)
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


# ----------------------------------------------------------------------------------------- #
# The loop
# ----------------------------------------------------------------------------------------- #

def run_iterative_ensemble(members: Sequence[IterMember], y_valid: np.ndarray, *,
                           settings: dict, runner, RESULT_NAME: str, runner_ctx: dict,
                           valid_score_kwargs: Optional[dict] = None,
                           chrom_of_marker: Optional[Mapping[str, str]] = None) -> IterResult:
    """Run the iterative ensemble for ONE task and ONE ensemble group.

    `members` are the group's Step 9 fits (>= 2). `y_valid` is the validation phenotype (used
    for acceptance only; no test target is ever read). `settings` is a resolved
    ``ITERATIVE_ENSEMBLE`` dict (``pipeline_utils.resolve_iterative_ensemble``). `runner` is a
    ``ModelTrialRunner``; `runner_ctx` carries the ``_run_units`` keywords ``n_jobs``,
    ``worker_init`` and ``worker_init_args``. `valid_score_kwargs` is
    ``blue_weights.scoring_for(ctx, 'valid')`` (``{}`` when unweighted). `chrom_of_marker` maps
    marker name -> chromosome (PLINK input) so LD windows never span chromosomes.

    Algorithm (blueprint R2): rank the markers once from the Step 9 effects (LD-aware order,
    optional interaction protection); round 0 = the equal-weight mean of the Step 9 columns;
    for r = 1..max_rounds refit the marker-channel members on the kept set (base budget
    ``max(marker_min, ceil(marker_keep**r * M))``, protected markers, plus the dropped markers
    most associated with the best round's training residual), explainability off; accept the
    round only when its validation gain beats ``accept_round``; stop at the first rejection
    (``patience``), ``max_rounds``, when the budget no longer shrinks, or when the fit budget
    ``max_fits`` (0 = ``max_rounds`` x marker-channel members) would be exceeded.

    ver4-68: with ``settings['mode'] == 'two_phase'`` the same inputs are handed, after the
    shared-pool split, to :func:`_run_two_phase` (snapshot library, no acceptance loop); the
    default ``'subset'`` runs the algorithm above byte for byte as before.

    Returns an :class:`IterResult`; ``pred == {}`` (with a note) means no row for this group.
    Raises ValueError only on contract violations (fewer than 2 members, mismatched lengths)."""
    members = list(members)
    notes: List[str] = []
    if len(members) < 2:
        raise ValueError(f"iterative ensemble needs at least 2 members, got {len(members)}")
    for m in members:
        _check_member(m)
    y_valid = np.asarray(y_valid, dtype=float).ravel()
    n_valid = members[0].valid.shape[0]
    if y_valid.shape[0] != n_valid:
        raise ValueError(f"iterative ensemble: y_valid has {y_valid.shape[0]} value(s) but the "
                         f"validation frames have {n_valid} row(s)")

    def _empty(reason: str) -> IterResult:
        notes.append(reason)
        return IterResult(pred={}, members=[m.name for m in members], best_round=0, kept=None,
                          effect=None, interactions=None, rounds=[], round_test_pred=[],
                          n_fits=0, notes=notes)

    marker_members = [m for m in members if m.marker_channel]
    if len(marker_members) < 1:
        return _empty("no marker-channel member (every member is a bio-prior model, which is "
                      "never refitted) - nothing to iterate")
    pool_cols = list(marker_members[0].train.columns[:-1])
    n_markers = len(pool_cols)

    # A marker-channel member whose frames are not the shared pool cannot be given the same
    # column subset: it keeps its Step 9 column in every round (never refitted).
    refit_members: List[IterMember] = []
    for m in marker_members:
        same_cols = list(m.train.columns[:-1]) == pool_cols \
            and list(m.valid.columns[:-1]) == pool_cols and list(m.test.columns[:-1]) == pool_cols
        if same_cols:
            refit_members.append(m)
        else:
            notes.append(f"member {m.name!r} does not use the shared marker pool "
                         f"({m.train.shape[1] - 1} vs {n_markers} marker columns) - it keeps its "
                         f"Step 9 column in every round")
    if not refit_members:
        return _empty("no member shares the marker pool - nothing to iterate")

    # ver4-68 (R1): the snapshot-library mode. Everything above (contract checks, the shared-pool
    # split) is common; the marker-subset loop below is untouched and stays the default.
    if settings.get('mode', 'subset') == 'two_phase':
        return _run_two_phase(members, refit_members, y_valid, settings=settings, runner=runner,
                              RESULT_NAME=RESULT_NAME, runner_ctx=runner_ctx,
                              valid_score_kwargs=valid_score_kwargs, notes=notes)

    rel = consensus_relevance({m.name: m.effect for m in refit_members}, n_markers,
                              settings['rank_method'])
    if rel is None:
        return _empty("fewer than 2 members have a usable (finite, non-zero, full-length) "
                      "Step 9 marker-effect vector - no consensus ranking, no iteration")

    chrom = chrom_of_marker
    if settings['ld_aware']:
        win = window_ids(pool_cols, settings['ld_window'], chrom)
    else:
        win = None
    order = marker_order(rel, win)
    if settings['protect_interactions']:
        protect = protected_markers([m.interaction for m in refit_members], pool_cols,
                                    settings['interaction_protect_top'])
    else:
        protect = np.zeros(n_markers, dtype=bool)

    keep, min_keep = settings['marker_keep'], settings['marker_min']
    k_marker = len(refit_members)
    fit_cap = int(settings['max_fits']) if settings['max_fits'] > 0 \
        else int(settings['max_rounds']) * k_marker

    train_X = refit_members[0].train.iloc[:, :-1].to_numpy(dtype=float)
    train_y = refit_members[0].train.iloc[:, -1].to_numpy(dtype=float)

    # ---- round 0 = the naive ensemble ------------------------------------------------------
    cur: Dict[str, Dict[str, np.ndarray]] = {
        m.name: {s: np.asarray(m.pred[s], dtype=float).ravel() for s in _SPLITS} for m in members}
    best_cols = {s: _equal_mean(cur, s) for s in _SPLITS}
    best_round, best_kept = 0, None
    best_valid_r = _score(y_valid, best_cols['valid'], valid_score_kwargs)
    rounds: List[dict] = [dict(round=0, n_markers_kept=n_markers, n_reentered=0, accepted=True,
                               valid_pearson=best_valid_r, valid_gain=float('nan'),
                               valid_gain_sd=float('nan'), n_fits=0, seconds=0.0,
                               kept_mask=None)]
    round_test_pred: List[np.ndarray] = [best_cols['test'].copy()]
    n_fits = 0
    rejected = 0
    prev_budget = None

    for r in range(1, int(settings['max_rounds']) + 1):
        budget = round_budget(r, keep, min_keep, n_markers)
        if budget >= n_markers:
            notes.append(f"round {r}: the base budget ({budget}) is not smaller than the marker "
                         f"pool ({n_markers}; marker_min={min_keep}) - nothing to subset, stopped")
            break
        if prev_budget is not None and budget >= prev_budget:
            notes.append(f"round {r}: the marker budget reached marker_min={min_keep} - stopped")
            break
        if n_fits + k_marker > fit_cap:
            notes.append(f"round {r}: {n_fits} fit(s) done and one more round needs {k_marker} - "
                         f"above max_fits={fit_cap} - stopped")
            break
        prev_budget = budget

        t0 = time.time()
        base = kept_mask(order, r, keep, min_keep, protect)
        resid = train_y - best_cols['train']
        back = reentry_markers(train_X, resid, base, settings['reentry_share'], budget)
        if settings['reentry_share'] > 0 and not back.any():
            if not np.all(np.isfinite(resid)) or float(np.std(resid)) <= 0.0:
                notes.append(f"round {r}: the training residual is non-finite or constant - "
                             f"re-entry skipped")
        kept = base | back

        units = []
        for m in refit_members:
            tr, va, _ = member_frames(m, kept)
            units.append(_FitUnit(model=m.base, params=_disable_explainability(m.base, m.params),
                                  split_tag=f'ie{r}', train=tr, valid=va, fidelity='full',
                                  keep_full=True))
        test_red = member_frames(refit_members[0], kept)[2]
        fitted = _refit_columns(units, refit_members, runner, RESULT_NAME, test_red, runner_ctx)
        n_fits += len(units)

        cand = {name: dict(cols) for name, cols in cur.items()}
        n_ok = 0
        for m, cols in zip(refit_members, fitted):
            if cols is None:
                notes.append(f"round {r}: refit of {m.name!r} failed or returned columns of the "
                             f"wrong length - it keeps its last good column")
                continue
            # a refitted member's test column is on the reduced test frame: same rows, same order
            cand[m.name] = {s: cols[s] for s in _SPLITS}
            n_ok += 1
        if n_ok == 0:
            notes.append(f"round {r}: no member could be refitted - stopped")
            break

        cand_cols = {s: _equal_mean(cand, s) for s in _SPLITS}
        accepted, gain, sd = accept_round(
            cand_cols['valid'], best_cols['valid'], y_valid, rule=settings['accept_rule'],
            margin=settings['accept_margin'], reps=settings['bootstrap_reps'],
            seed=int(settings['seed']) + _SEED_STRIDE * r, score_kwargs=valid_score_kwargs)
        cand_valid_r = _score(y_valid, cand_cols['valid'], valid_score_kwargs)
        rounds.append(dict(round=r, n_markers_kept=int(kept.sum()), n_reentered=int(back.sum()),
                           accepted=bool(accepted), valid_pearson=cand_valid_r, valid_gain=gain,
                           valid_gain_sd=sd, n_fits=n_fits, seconds=time.time() - t0,
                           kept_mask=kept.copy()))
        round_test_pred.append(cand_cols['test'].copy())
        if accepted:
            cur, best_cols, best_round, best_kept = cand, cand_cols, r, kept
            best_valid_r = cand_valid_r
        else:
            rejected += 1
            if rejected >= int(settings['patience']):
                break

    # ---- reported effect row: never zero for a dropped marker -----------------------------
    effect = pd.Series(rel, index=pool_cols, dtype=float)
    effect_refit = False
    if settings['explain_selected'] and best_round > 0 and best_kept is not None:
        units = []
        for m in refit_members:
            tr, va, _ = member_frames(m, best_kept)
            units.append(_FitUnit(model=m.base, params=list(m.params),
                                  split_tag='ie_explain', train=tr, valid=va, fidelity='full',
                                  keep_full=True))
        test_red = member_frames(refit_members[0], best_kept)[2]
        explained = _refit_columns(units, refit_members, runner, RESULT_NAME, test_red,
                                   runner_ctx)
        n_fits += len(units)
        consensus = _refit_effect_consensus(explained, int(best_kept.sum()))
        if consensus is None:
            notes.append("explain_selected: no refitted member returned a usable effect vector "
                         "- the effect row keeps the Step 9 consensus for every marker")
        else:
            kept_pos = np.flatnonzero(best_kept)
            step9_mass = float(rel[kept_pos].sum())
            refit_vals = consensus * step9_mass            # same total mass as the Step 9 values
            effect = effect.copy()
            effect.iloc[kept_pos] = refit_vals
            effect_refit = True

    return IterResult(
        pred={s: best_cols[s] for s in _SPLITS}, members=[m.name for m in members],
        best_round=best_round, kept=None if best_kept is None else best_kept.copy(),
        effect=effect, interactions=None, rounds=rounds, round_test_pred=round_test_pred,
        n_fits=n_fits, notes=notes, effect_refit=effect_refit)


# ----------------------------------------------------------------------------------------- #
# ver4-68: the two-phase snapshot library (R1 / R2 / R3)
# ----------------------------------------------------------------------------------------- #

def _run_wave(units: List[_FitUnit], runner, RESULT_NAME, test: pd.DataFrame,
              runner_ctx: Optional[dict]) -> List[Optional[dict]]:
    """Run one dependency wave - every unit whose inputs are ready - as ONE ``_run_units`` call.

    Each unit carries its own ``test_frame`` (R2), so units with different marker views, and the
    out-of-fold units, share a single parallel fan-out; `test` is only the fallback for a unit
    without one. A failed unit never stops the others (``stop_on_failure=False``). Returns one raw
    result (a runner dict for a ``keep_full`` unit, ``{'predicted_valid': ...}`` otherwise) or
    None per unit, in `units` order. Why a wrapper: the same call, spelled once, for the wave
    planner and the tests (a spy on ``_run_units`` counts waves)."""
    return _run_units(units, runner=runner, RESULT_NAME=RESULT_NAME, test=test,
                      stop_on_failure=False, **(runner_ctx or {}))


def _columns_from_result(res, unit: _FitUnit, test: pd.DataFrame
                         ) -> Optional[Dict[str, np.ndarray]]:
    """The validated (train, valid, test) float columns of one ``keep_full`` refit result, or None
    when the fit failed, a column is missing or has the wrong length (against the unit's own
    frames and its own test frame), or a value is not finite. A non-finite snapshot would poison
    the pooled mean, so it is treated as a failed fit."""
    if not isinstance(res, dict):
        return None
    test_ref = unit.test_frame if unit.test_frame is not None else test
    expected = {'train': unit.train.shape[0], 'valid': unit.valid.shape[0],
                'test': test_ref.shape[0]}
    cols: Dict[str, np.ndarray] = {}
    for split in _SPLITS:
        arr = res.get(f'predicted_{split}')
        if arr is None:
            return None
        vec = np.asarray(arr, dtype=float).ravel()
        if vec.shape[0] != expected[split] or not np.all(np.isfinite(vec)):
            return None
        cols[split] = vec
    return cols


def _oof_vector(res, n_expected: int) -> Optional[np.ndarray]:
    """The held-out predictions of one out-of-fold unit as a finite float vector of `n_expected`
    values, or None (failed fit, missing or wrong-length or non-finite column)."""
    if not isinstance(res, dict):
        return None
    arr = res.get('predicted_valid')
    if arr is None:
        return None
    vec = np.asarray(arr, dtype=float).ravel()
    if vec.shape[0] != n_expected or not np.all(np.isfinite(vec)):
        return None
    return vec


def _fit_stats(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """(Pearson r, MSE) of `p` against `y`; ``(nan, nan)`` for fewer than 3 rows or non-finite
    values; r is ``nan`` for a constant vector. Pure."""
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape or y.size < 3 or not (np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
        return float('nan'), float('nan')
    mse = float(np.mean((y - p) ** 2))
    if float(np.std(y)) == 0.0 or float(np.std(p)) == 0.0:
        return float('nan'), mse
    return float(np.corrcoef(y, p)[0, 1]), mse


def alpha_grid(cap: float) -> Optional[List[float]]:
    """ver4-69 (R2): the shrinkage levels ``0, .1, ... , cap`` of the member-weight path, or
    ``None`` (the default grid ``0, .1, ... 1``, ver4-68 behaviour) when ``cap >= 1``. The cap
    itself is always a level, so ``cap=0.25`` gives ``0, .1, .2, .25``. Pure; a cap that is not a
    finite number in (0, 1] counts as 1."""
    try:
        cap = float(cap)
    except (TypeError, ValueError):
        return None
    if not (0.0 < cap < 1.0):
        return None
    grid = [round(0.1 * i, 10) for i in range(int(cap * 10 + 1e-9) + 1)]
    if grid[-1] < cap - 1e-9:
        grid.append(round(cap, 10))
    return grid


def _run_two_phase(members: Sequence[IterMember], refit_members: Sequence[IterMember],
                   y_valid: np.ndarray, *, settings: dict, runner, RESULT_NAME: str,
                   runner_ctx: dict, valid_score_kwargs: Optional[dict],
                   notes: List[str]) -> IterResult:
    """The ``mode='two_phase'`` snapshot library for ONE task and ONE ensemble group (R1).

    `members` are the group's Step 9 fits (>= 2); `refit_members` the marker-channel members that
    share the marker pool (the only ones that are refitted, I6). Notes are appended to `notes`.

    Algorithm (``p1`` = ``p1_rounds``, ``p2`` = ``p2_rounds``, ``F`` = ``oof_folds``)::

        OOF0  = out-of-fold predictions of all K members on the training rows   (K*F fits)
        w0    = shrinkage_path_weights(OOF0 + validation rows)   ('factorised'; else equal)
        lib[m] = [Step 9 fit of m]
        phase 1, r = 1..p1:  t_i = (1-lam)*y + lam*mean_{j != i} OOF_j ; refit i on t_i
                             (extra snapshot; the OOF of that round is rebuilt for r < p1)
        phase 2, r = 1..p2:  rel_i = |corr(markers, member i's last train predictions)|
                             masks = cma_masks(rel) ; refit i on its masked markers
        pred = sum_m w0_m * mean(lib[m])            ('equal': the mean of ALL library columns)

    Why this shape (design rules D1-D8, module docstring): the output is a pool whose only
    validation-fitted quantity is the K-vector w0; no member is deployed after seeing ensemble
    information; nothing is selected between rounds.

    Fits are grouped into dependency waves (``pipeline_utils.iter_ens_wave_groups``); each wave
    is one ``_run_units`` call. The result does not depend on the grouping (R2). Out-of-fold units
    use the parity-guarded 1-row test stub for rrBLUP / GBLUP / BayesB / RKHS and, with
    ``oof_engine='exact'``, the closed-form kernel engine for rrBLUP / RKHS (R3). Early-stopping
    models see the real phenotype of the held-out rows, the same precedent as
    ``joint_tuning._oof_pooled_matrix``.

    Degradation, never an exception (a NOTE is appended): a group with a bio-prior member, or a
    training split below ``10 * F`` rows, cannot be fitted out of fold - ``w0`` then comes from
    the validation rows (``cv_shrunk_simplex_weights``) and phase 1 is skipped; a failed round-0
    out-of-fold unit does the same; a failed refit leaves that snapshot out of the library; fewer
    than 2 usable relevance vectors skip phase 2; ``max_fits`` stops after the last complete wave.
    Raises ValueError only on a contract violation (frames of different row counts).

    ver4-69: ``w0_alpha_max`` < 1 limits the member-weight path to ``0, .1, ... , w0_alpha_max``
    (a cap on how far w0 may move from equal weights; 1 = ver4-68 behaviour exactly), and the
    result carries ``attribution`` (test predictions of the naive ensemble, the Step 9 columns at
    w0, the equal-member snapshot pool and the final row) plus per-member out-of-fold and
    validation fit statistics in ``diag`` - reporting only, no decision reads them.

    The returned :class:`IterResult` has ``kept=None`` and ``effect`` = the Step 9 consensus
    (zeros when fewer than 2 members report a usable effect); ``member_weights`` is each member's
    share of ``pred``."""
    t_start = time.time()
    cfg = dict(_ITER_ENS_DEFAULTS)
    cfg.update(settings)
    members = list(members)
    refit_members = list(refit_members)
    names = [m.name for m in members]
    n_members = len(members)
    n_train, n_valid, n_test = (members[0].train.shape[0], members[0].valid.shape[0],
                                members[0].test.shape[0])
    for m in members:
        if (m.train.shape[0], m.valid.shape[0], m.test.shape[0]) != (n_train, n_valid, n_test):
            raise ValueError(
                f"iterative ensemble (two_phase): member {m.name!r} has frames of "
                f"{m.train.shape[0]}/{m.valid.shape[0]}/{m.test.shape[0]} train/valid/test rows, "
                f"expected {n_train}/{n_valid}/{n_test}")
    y_valid = np.asarray(y_valid, dtype=float).ravel()
    seed = int(cfg['seed'])
    n_folds = int(cfg['oof_folds'])
    lam = float(cfg['p1_distill'])
    rule = str(cfg['weight_rule'])
    fit_cap = int(cfg['max_fits'] or 0)
    tier = iter_ens_tier_of(cfg)
    a_max = float(cfg.get('w0_alpha_max', 1.0))
    a_grid = alpha_grid(a_max)                 # None = the default grid (ver4-68)

    from model_registry import exact_kernel_search_capable, search_test_stub_safe

    pool_cols = list(refit_members[0].train.columns[:-1])
    n_markers = len(pool_cols)
    X_tr = refit_members[0].train.iloc[:, :-1].to_numpy(dtype=float)
    y_tr = refit_members[0].train.iloc[:, -1].to_numpy(dtype=float)
    s9 = {m.name: {s: np.asarray(m.pred[s], dtype=float).ravel() for s in _SPLITS}
          for m in members}
    lib: Dict[str, List[Dict[str, np.ndarray]]] = {name: [s9[name]] for name in names}

    # ---- what can run ----------------------------------------------------------------------
    oof_ok, oof_reasons = _oof_possible(cfg, members)
    p1_req = int(cfg['p1_rounds'])
    if not oof_ok:
        consequence = []
        if rule == 'factorised':
            consequence.append("the member weights come from the validation rows")
        if p1_req >= 1:
            consequence.append("phase 1 is skipped")
        notes.append("two_phase: no out-of-fold fits are possible - " + "; ".join(oof_reasons)
                     + (" - so " + " and ".join(consequence) if consequence else ""))
    need_oof = oof_ok and (rule == 'factorised' or p1_req >= 1)
    p1_active = oof_ok and p1_req >= 1
    waves = iter_ens_wave_groups(cfg, n_members, len(refit_members), oof_ok)
    if valid_score_kwargs and (need_oof or rule == 'factorised'):
        notes.append("two_phase: BLUE sample weights are not used when the member weights w0 are "
                     "estimated (shrinkage_path_weights is unweighted) - the weights are "
                     "unweighted estimates; the validation Pearson in the rounds file is weighted.")

    # ---- member weights -------------------------------------------------------------------
    w0: Optional[np.ndarray] = None
    w0_source = 'equal' if rule == 'equal' else None
    alpha_star = float('nan')
    w0_path: Optional[dict] = None             # ver4-69 (R1): the alpha path of the weight CV
    oof_fit: Optional[Dict[str, Tuple[float, float]]] = None   # per member (Pearson, MSE) of OOF0

    def _w0_from_valid_rows(reason: str) -> None:
        nonlocal w0, w0_source, alpha_star
        valid_mat = np.column_stack([s9[n]['valid'] for n in names])
        weights, alpha, _diag = cv_shrunk_simplex_weights(valid_mat, y_valid, seed=seed,
                                                          alpha_cap=min(a_max, 1.0))
        vec = np.asarray(weights, dtype=float).ravel()
        if vec.shape[0] != n_members or not np.all(np.isfinite(vec)) or vec.sum() <= 0.0:
            notes.append(f"two_phase: the validation-row weights were unusable ({reason}) - equal "
                         f"member weights are used")
            vec = np.full(n_members, 1.0 / n_members)
            alpha = float('nan')
        w0, w0_source, alpha_star = vec / vec.sum(), 'valid_rows', float(alpha)
        notes.append(f"two_phase: member weights w0 estimated on the validation rows only "
                     f"({reason}); alpha*={alpha_star:.2f}")

    if rule == 'factorised' and not need_oof:
        _w0_from_valid_rows("no out-of-fold fits were possible")

    # ---- phase-2 relevance, restarted from the Step 9 fits -----------------------------------
    rel_cur: Dict[str, Optional[np.ndarray]] = {
        m.name: prediction_relevance(X_tr, s9[m.name]['train']) for m in refit_members}
    p2_active = sum(1 for v in rel_cur.values() if v is not None) >= 2
    if not p2_active:
        notes.append("two_phase: fewer than 2 members have a usable (finite, non-constant) "
                     "training prediction - no relevance vectors, phase 2 skipped")

    folds: List[np.ndarray] = []
    if need_oof:
        folds = [np.sort(f) for f in _positional_folds(n_train, n_folds, seed + _OOF_SEED_OFFSET)]
    all_pos = np.arange(n_train)

    def _oof_unit(m: IterMember, held: np.ndarray, tag: str,
                  target: Optional[np.ndarray]) -> _FitUnit:
        """One out-of-fold unit: fit on the training rows outside `held`, predict `held`."""
        tr_pos = np.setdiff1d(all_pos, held)
        tr = m.train.iloc[tr_pos]
        if target is not None:
            tr = tr.copy()
            tr.iloc[:, -1] = target[tr_pos]
        exact = cfg['oof_engine'] == 'exact' and exact_kernel_search_capable(m.base)
        return _FitUnit(model=m.base, params=_disable_explainability(m.base, m.params),
                        split_tag=tag, train=tr, valid=m.train.iloc[held], fidelity='oof',
                        keep_full=False, engine='exact_kernel' if exact else 'runner',
                        test_stub=bool(search_test_stub_safe(m.base)), test_frame=m.test)

    def _refit_unit(m: IterMember, tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame,
                    tag: str) -> _FitUnit:
        """One refit unit whose columns enter the pool (a real fit, explainability off)."""
        return _FitUnit(model=m.base, params=_disable_explainability(m.base, m.params),
                        split_tag=tag, train=tr, valid=va, fidelity='full', keep_full=True,
                        test_frame=te)

    # ---- the waves --------------------------------------------------------------------------
    oof_cur: Dict[str, np.ndarray] = {}
    p1_snaps: List[dict] = []
    p2_snaps: List[dict] = []
    n_fits = 0
    n_waves_run = 0
    for wave_no, wave in enumerate(waves, start=1):
        ids = [gid for gid, _count in wave]
        units: List[_FitUnit] = []
        tags: List[tuple] = []          # (kind, member name, held-out positions | mask | None)
        p1_targets: Dict[str, np.ndarray] = {}
        r1 = wave_no - 1                # the phase-1 round this wave refits (w >= 2)

        if 'oof0' in ids:
            for f, held in enumerate(folds):
                for m in members:
                    units.append(_oof_unit(m, held, f'tp0f{f}', None))
                    tags.append(('oof0', m.name, held))
        if f'p1r{r1}' in ids and p1_active:
            for m in refit_members:
                others = [oof_cur[o.name] for o in members if o.name != m.name]
                y_m = m.train.iloc[:, -1].to_numpy(dtype=float)
                p1_targets[m.name] = (1.0 - lam) * y_m + lam * np.mean(others, axis=0)
                tr = m.train.copy()
                tr.iloc[:, -1] = p1_targets[m.name]
                units.append(_refit_unit(m, tr, m.valid, m.test, f'tp1r{r1}'))
                tags.append(('p1', m.name, None))
            if f'p1r{r1}oof' in ids:
                for f, held in enumerate(folds):
                    for m in refit_members:
                        units.append(_oof_unit(m, held, f'tp1r{r1}f{f}', p1_targets[m.name]))
                        tags.append(('p1oof', m.name, held))
        if f'p2r{wave_no}' in ids and p2_active:
            usable = [m for m in refit_members if rel_cur[m.name] is not None]
            if len(usable) < 2:
                notes.append(f"two_phase: phase-2 round {wave_no}: fewer than 2 usable relevance "
                             f"vectors - phase 2 stopped")
                p2_active = False
            else:
                masks = cma_masks([rel_cur[m.name] for m in usable], float(cfg['cma_core']),
                                  float(cfg['cma_unique']))
                for m, mask in zip(usable, masks):
                    tr, va, te = member_frames(m, mask)
                    units.append(_refit_unit(m, tr, va, te, f'tp2r{wave_no}'))
                    tags.append(('p2', m.name, mask))
        if not units:
            continue
        if fit_cap > 0 and n_fits + len(units) > fit_cap:
            notes.append(f"two_phase: wave {wave_no} needs {len(units)} fit(s) with {n_fits} done "
                         f"- above max_fits={fit_cap} - stopped; the library holds what is done")
            break

        t_wave = time.time()
        results = _run_wave(units, runner, RESULT_NAME, members[0].test, runner_ctx)
        n_fits += len(units)
        n_waves_run += 1
        wave_secs = time.time() - t_wave

        oof0_buf = {n: np.full(n_train, np.nan) for n in names} if 'oof0' in ids else {}
        p1oof_buf = {m.name: np.full(n_train, np.nan) for m in refit_members}
        failed_oof0: List[str] = []
        failed_p1oof: List[str] = []
        p1_cols: Dict[str, Dict[str, np.ndarray]] = {}
        p2_cols: Dict[str, Dict[str, np.ndarray]] = {}
        p2_masks: Dict[str, np.ndarray] = {}
        for (kind, name, extra), unit, res in zip(tags, units, results):
            if kind in ('oof0', 'p1oof'):
                vec = _oof_vector(res, extra.shape[0])
                buf, bad = (oof0_buf, failed_oof0) if kind == 'oof0' else (p1oof_buf, failed_p1oof)
                if vec is None:
                    if name not in bad:
                        bad.append(name)
                else:
                    buf[name][extra] = vec
            else:
                cols = _columns_from_result(res, unit, members[0].test)
                if cols is None:
                    phase_txt = 'phase-1' if kind == 'p1' else 'phase-2'
                    notes.append(f"two_phase: {phase_txt} refit of {name!r} (wave {wave_no}) "
                                 f"failed, returned columns of the wrong length or non-finite "
                                 f"values - that snapshot is left out of the library")
                elif kind == 'p1':
                    p1_cols[name] = cols
                else:
                    p2_cols[name] = cols
                    p2_masks[name] = extra

        # -- round-0 out-of-fold columns -> member weights ------------------------------------
        if 'oof0' in ids:
            complete = not failed_oof0 and all(np.all(np.isfinite(b)) for b in oof0_buf.values())
            if not complete:
                notes.append(f"two_phase: the round-0 out-of-fold fit of {failed_oof0 or names!r} "
                             f"failed or left rows unfilled - phase 1 skipped"
                             + ("; member weights from the validation rows"
                                if rule == 'factorised' else ""))
                p1_active = False
                if rule == 'factorised':
                    _w0_from_valid_rows("a round-0 out-of-fold fit failed")
            else:
                oof_cur = {n: oof0_buf[n] for n in names}
                oof_fit = {n: _fit_stats(y_tr, oof_cur[n]) for n in names}
                if rule == 'factorised':
                    pooled = np.vstack([np.column_stack([oof_cur[n] for n in names]),
                                        np.column_stack([s9[n]['valid'] for n in names])])
                    target = np.concatenate([y_tr, y_valid])
                    weights, alpha, wdiag = shrinkage_path_weights(
                        pooled, target, direction='analytic', rule='min', seed=seed,
                        alphas=a_grid)
                    w0 = np.asarray(weights, dtype=float)
                    w0_source, alpha_star = 'oof_cv', float(alpha)
                    w0_path = dict(cv_mse={float(k): float(v)
                                           for k, v in dict(wdiag.get('cv_mse', {})).items()},
                                   direction_w=[float(x) for x in wdiag.get('direction_w', [])],
                                   fold_spread=wdiag.get('fold_spread'),
                                   n_rows=int(wdiag.get('n_rows', pooled.shape[0])))
                    if wdiag.get('note'):
                        notes.append(f"two_phase: member weights - {wdiag['note']}")

        # -- phase 1 snapshot -----------------------------------------------------------------
        if f'p1r{r1}' in ids and any(k == 'p1' for k, _n, _e in tags):
            if p1_cols:
                p1_snaps.append(dict(cols=p1_cols, mask=None, n_markers=n_markers,
                                     n_fits=n_fits, seconds=wave_secs))
                for name, cols in p1_cols.items():
                    lib[name].append({s: cols[s] for s in _SPLITS})
            else:
                notes.append(f"two_phase: phase-1 round {r1}: no member could be refitted - "
                             f"phase 1 stopped")
                p1_active = False
            if f'p1r{r1}oof' in ids:
                if failed_p1oof:
                    notes.append(f"two_phase: phase-1 round {r1}: the out-of-fold fit of "
                                 f"{failed_p1oof!r} failed - phase 1 stops after this round")
                    p1_active = False
                else:
                    for m in refit_members:
                        oof_cur[m.name] = p1oof_buf[m.name]

        # -- phase 2 snapshot -----------------------------------------------------------------
        if any(k == 'p2' for k, _n, _e in tags):
            if p2_cols:
                union = np.zeros(n_markers, dtype=bool)
                for mask in p2_masks.values():
                    union |= mask
                p2_snaps.append(dict(
                    cols=p2_cols, mask=union, n_fits=n_fits, seconds=wave_secs,
                    n_markers=int(round(float(np.mean([mk.sum() for mk in p2_masks.values()]))))))
                for name, cols in p2_cols.items():
                    lib[name].append({s: cols[s] for s in _SPLITS})
                    new_rel = prediction_relevance(X_tr, cols['train'])
                    if new_rel is not None:
                        rel_cur[name] = new_rel          # chained (D7); else unchanged
            else:
                notes.append(f"two_phase: phase-2 round {wave_no}: no member could be refitted - "
                             f"phase 2 stopped")
                p2_active = False

    if rule == 'factorised' and w0 is None:
        _w0_from_valid_rows("the out-of-fold wave did not run")

    # ---- output: pooled prediction ---------------------------------------------------------
    if rule == 'factorised':
        weights_final = np.asarray(w0, dtype=float)
    else:
        counts = np.array([len(lib[n]) for n in names], dtype=float)
        weights_final = counts / counts.sum()
    pooled_member = {s: np.column_stack([np.mean([c[s] for c in lib[n]], axis=0) for n in names])
                     for s in _SPLITS}
    pred = {s: pooled_member[s] @ weights_final for s in _SPLITS}
    w_diag = weights_final if rule == 'factorised' else np.full(n_members, 1.0 / n_members)

    def _snapshot_pred(cols_by_member: Mapping[str, Dict[str, np.ndarray]], split: str
                       ) -> np.ndarray:
        """The ensemble of ONE snapshot round at the diagnostic weights (a member without a
        column in this round contributes its Step 9 column)."""
        mat = np.column_stack([(cols_by_member.get(n) or s9[n])[split] for n in names])
        return mat @ w_diag

    def _row(round_no: int, phase: str, cols_by_member: Mapping[str, Dict[str, np.ndarray]],
             n_kept: int, mask: Optional[np.ndarray], fits: int,
             secs: float) -> Tuple[dict, np.ndarray]:
        valid_r = _score(y_valid, _snapshot_pred(cols_by_member, 'valid'), valid_score_kwargs)
        return (dict(round=round_no, phase=phase, n_markers_kept=int(n_kept), n_reentered=0,
                     accepted=True, selected=True, valid_pearson=valid_r,
                     valid_gain=float('nan'), valid_gain_sd=float('nan'), n_fits=int(fits),
                     seconds=float(secs), kept_mask=None if mask is None else mask.copy()),
                _snapshot_pred(cols_by_member, 'test'))

    rounds: List[dict] = []
    round_test_pred: List[np.ndarray] = []
    row, test_pred = _row(0, 'r0', {}, n_markers, None, 0, 0.0)
    rounds.append(row)
    round_test_pred.append(test_pred)
    for phase, snaps in (('p1', p1_snaps), ('p2', p2_snaps)):
        for snap in snaps:
            row, test_pred = _row(len(rounds), phase, snap['cols'], snap['n_markers'],
                                  snap['mask'], snap['n_fits'], snap['seconds'])
            rounds.append(row)
            round_test_pred.append(test_pred)

    consensus = consensus_relevance({m.name: m.effect for m in refit_members}, n_markers,
                                    'mean_effect')
    effect = pd.Series(np.zeros(n_markers) if consensus is None else consensus,
                       index=pool_cols, dtype=float)
    diag = dict(mode='two_phase', tier=tier, weight_rule=rule, w0_source=w0_source,
                alpha_star=alpha_star, n_waves_planned=len(waves), n_waves_run=n_waves_run,
                p1_rounds_done=len(p1_snaps), p2_rounds_done=len(p2_snaps),
                seconds=time.time() - t_start, oof_engine=str(cfg['oof_engine']),
                # ver4-69 (R1): what the weights were fitted on and how each member looked there
                alpha_max=a_max, w0_path=w0_path, n_weight_rows=(
                    None if w0_path is None else w0_path['n_rows']),
                oof_fit=oof_fit,
                valid_fit={n: _fit_stats(y_valid, s9[n]['valid']) for n in names},
                lib_columns={n: len(lib[n]) for n in names})
    # ver4-69 (R1): intermediate ensembles of this task, test side, for the attribution log
    equal_w = np.full(n_members, 1.0 / n_members)
    s9_test = np.column_stack([s9[n]['test'] for n in names])
    attribution = {'step9_equal': s9_test @ equal_w, 'pool_equal': pooled_member['test'] @ equal_w,
                   'final': pred['test'].copy()}
    if w0 is not None:
        attribution['step9_w0'] = s9_test @ np.asarray(w0, dtype=float)
    return IterResult(
        pred=pred, members=names, best_round=len(rounds) - 1, kept=None, effect=effect,
        interactions=None, rounds=rounds, round_test_pred=round_test_pred, n_fits=n_fits,
        notes=notes, effect_refit=False,
        member_weights={n: float(w) for n, w in zip(names, weights_final)}, diag=diag,
        attribution=attribution)
