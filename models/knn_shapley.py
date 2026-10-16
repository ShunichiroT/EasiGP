"""models/knn_shapley.py  (ver4-67)
================================

Exact-game Shapley marker effects for k-nearest-neighbour regression, fast
enough for panels of 10,000+ markers, and the closed-form evaluator that makes
KNN's `baseline_shap` pair interactions O(n_train) per hybrid row instead of
O(n_train * M).

Zero project imports (NumPy only), like the other Layer 3d authorities, so any
module - and the offline bench - can import it without circularity risk.

Why KNN needs its own engine
----------------------------
Before ver4-67 KNN's marker *effect* (`'kernel_shap'`, i.e.
`pipeline_utils.knn_relief_effects`) was a Relief-style neighbour-distance
decomposition, while its marker-pair *interactions* (`'baseline_shap'`) were a
background-referenced Shapley-type second difference. Two different estimators,
on two different scales: the Relief effect does not sum to the prediction, and
a single-marker knock-out of a discrete k-NN prediction is almost always
exactly zero once the panel is wide (the ver4-43 note), so the generic
`baseline_shap_effects` route could not simply be reused.

The structural fact this module exploits is that the k-NN prediction depends on
the markers ONLY through the training-row distances

    D_t(S) = sum_{m in S} d_m(x, t)  +  sum_{m not in S} d_m(z, t),
    d_m(a, t) = |a_m - T[t, m]|**p            (rank distance, monotone in the
                                               Minkowski distance)

which is ADDITIVE over markers. So the game used by interventional ("baseline")
Shapley values,

    v_x(S) = E_{z in background} f( x_S , z_{not S} ),

can be walked one player at a time by adding a length-n_train vector
(`d_m(x, .) - d_m(z, .)`) to a running distance vector and re-ranking - O(n_train)
per step, independent of the number of markers - instead of re-predicting a
full-width row (O(n_train * M)).

What is computed
----------------
phi_m(x) is the Shapley value of marker m in the game v_x, estimated by
permutation sampling with antithetic (reversed) permutations and one real
training row z per walk (LD between non-swapped markers is therefore preserved,
as in `baseline_shap_interactions`). Every walk telescopes exactly:
sum_m phi_m(x) = f(x) - f(z), so efficiency holds per walk, with no
renormalisation. The reported score uses the same cross-row reduction as every
other effect route (`reduce='abs_then_sum'`: sum over explained rows of
|phi_m(x)|).

How it stays fast at M > 10,000 without giving up the Shapley definition
-----------------------------------------------------------------------
1. Incremental distance walk (above): per-step cost is independent of M.
2. Marker groups as players (stage 1): contiguous windows in genome order
   (`marker_order`), `n_players` of them (default 256). A group is a single
   Shapley player, so a walk has 256 steps instead of M and each step moves the
   distances enough to actually change the neighbour set (single-marker steps
   almost never do - the variance problem of KNN Shapley at large M).
3. Refinement (stage 2): groups are ranked by their stage-1 mass and the top
   ones (covering `refine_mass` of the total, capped at `refine_markers`
   markers) are split into single-marker players; the remaining groups stay as
   (merged) context players. This is again a genuine Shapley value - of the
   mixed-granularity game - for every refined marker.
4. Adaptive stopping on the relative standard error of the top players.
5. float32 distance arithmetic whenever it is exact (0/1/2 genotypes), tie rule
   identical to `models.KNN.DeterministicKNNPredict` (ties at the k-th distance
   go to the lower training-row index).

Unrefined markers (the low-mass tail) receive their group's stage-1 score
divided equally over the group - flagged in the returned details as
`resolved=False`. They are not individually estimated Shapley values.

Public API
----------
KNNDistanceGame        the additive-distance game (values(D), term_sum(...)).
knn_shapley_effects    the hierarchical estimator described above.
knn_shapley_exact      brute-force enumeration reference (M <= 16), used by the bench.
KNNBaselineShapFastEval  closed-form hybrid-row evaluator for
                       `interaction_extraction.baseline_shap_interactions`.
"""

from __future__ import annotations

import time

import numpy as np

__all__ = [
    'KNNDistanceGame', 'knn_shapley_effects', 'knn_shapley_exact', 'make_marker_blocks',
    'KNNBaselineShapFastEval', 'KNNShapleyDetails',
]

#: Largest integer a float32 holds exactly. Distance sums stay below this for
#: 0/1/2 genotypes up to ~2.0M markers at p=2, so float32 is exact there.
_FLOAT32_EXACT_LIMIT = float(2 ** 24)

#: Upper bound on the number of elements in one stacked (rows, n_train) distance
#: block handed to `KNNDistanceGame.values` (bounds its temporaries).
_VALUES_BLOCK_ELEMENTS = 4_000_000

#: Elements of one (walks, n_train) walk batch (D, the two gathers, masks ...).
_WALK_BATCH_ELEMENTS = 3_000_000


def _exact_dtype(arrays, n_markers, p):
    """float32 when every distance sum is an exactly representable integer
    (integer-coded markers), else float64. Never raises."""
    try:
        if float(p) != int(p):
            return np.float64
        vmax = 0.0
        for a in arrays:
            a = np.asarray(a, dtype=float)
            if a.size == 0:
                continue
            if not np.all(np.isfinite(a)) or not np.all(a == np.rint(a)):
                return np.float64
            vmax = max(vmax, float(np.max(np.abs(a))))
        # intermediate walk states can exceed the final distance by one player's
        # terms (add before subtract), hence the factor 2 headroom.
        bound = 2.0 * n_markers * (2.0 * vmax) ** float(p)
        return np.float32 if bound < _FLOAT32_EXACT_LIMIT else np.float64
    except Exception:
        return np.float64


class KNNDistanceGame:
    """k-NN regression as a function of training-row RANK distances.

    Mirrors `models.KNN.DeterministicKNNPredict` exactly (same neighbour
    selection, same tie rule, same 'distance'-weights exact-hit rule, same BLUE
    `sample_weight` folding, same mean(y) fallback) but takes the distances
    directly, so a caller can maintain them incrementally.

    Parameters
    ----------
    X_train : (n, M) - training genotypes (the columns the game is played on).
    y_train : (n,)
    n_neighbors : int in [1, n]
    weights : 'uniform' | 'distance'
    p : float > 0 - Minkowski power; the rank distance is sum |.|**p.
    sample_weight : (n,) or None - BLUE fit weights.
    dtype : numpy dtype or None - distance dtype; None picks float32 when exact.
    extra_arrays : optional iterable of arrays (e.g. explained rows) taken into
        account when deciding whether float32 is exact.
    """

    def __init__(self, X_train, y_train, n_neighbors, weights='uniform', p=2,
                 sample_weight=None, dtype=None, extra_arrays=()):
        T = np.ascontiguousarray(np.asarray(X_train, dtype=float))
        y = np.asarray(y_train, dtype=float).reshape(-1)
        if T.ndim != 2 or T.shape[0] != y.shape[0]:
            raise ValueError(
                f"KNNDistanceGame: X_train {T.shape} and y_train {y.shape} must describe the "
                f"same training rows.")
        if not 1 <= int(n_neighbors) <= T.shape[0]:
            raise ValueError(
                f"KNNDistanceGame: n_neighbors={n_neighbors} must be between 1 and the number "
                f"of training rows ({T.shape[0]}).")
        if weights not in ('uniform', 'distance'):
            raise ValueError(f"KNNDistanceGame: weights must be 'uniform' or 'distance', got {weights!r}.")
        if not float(p) > 0:
            raise ValueError(f"KNNDistanceGame: p must be positive, got {p!r}.")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
            if sample_weight.shape[0] != T.shape[0]:
                raise ValueError(
                    f"KNNDistanceGame: sample_weight has {sample_weight.shape[0]} entries for "
                    f"{T.shape[0]} training rows.")
        self.T = T
        self.n, self.M = T.shape
        self.y = y
        self.k = int(n_neighbors)
        self.weights = weights
        self.p = float(p)
        self.sample_weight = sample_weight
        self._w = sample_weight if sample_weight is not None else np.ones(self.n)
        self._wy = self._w * y
        self._y_mean = float(y.mean()) if y.size else 0.0
        self.dtype = np.dtype(dtype) if dtype is not None else np.dtype(
            _exact_dtype([T, *extra_arrays], self.M, self.p))

    # ---- distance terms ------------------------------------------------
    def _pow(self, absdiff):
        if self.p == 1.0:
            return absdiff
        if self.p == 2.0:
            return absdiff * absdiff
        return absdiff ** self.p

    def term_sum(self, R, idx):
        """(r, n) array of sum_{m in idx} |R[:, m] - T[:, m]|**p for rows `R`."""
        idx = np.asarray(idx, dtype=np.intp).reshape(-1)
        R = np.asarray(R, dtype=float)
        Rb = R[:, idx]
        Tb = self.T[:, idx]
        if self.p == 2.0 and idx.size > 4:
            D = ((Rb * Rb).sum(axis=1)[:, None] + (Tb * Tb).sum(axis=1)[None, :]
                 - 2.0 * (Rb @ Tb.T))
            np.maximum(D, 0.0, out=D)
        else:
            D = np.zeros((R.shape[0], self.n))
            for j in range(idx.size):
                D += self._pow(np.abs(Rb[:, j][:, None] - Tb[:, j][None, :]))
        return D.astype(self.dtype, copy=False)

    def player_terms(self, R, players):
        """(r, P, n) per-player distance terms for rows `R`. `players` is a
        sequence of 1-D marker-index arrays. Runs of single-marker players are
        computed in vectorised chunks."""
        R = np.asarray(R, dtype=float)
        r, P, n = R.shape[0], len(players), self.n
        out = np.empty((r, P, n), dtype=self.dtype)
        sizes = np.fromiter((len(pl) for pl in players), dtype=np.int64, count=P)
        chunk_c = max(1, int(_WALK_BATCH_ELEMENTS // max(1, r * n)))
        g = 0
        while g < P:
            if sizes[g] == 1:
                h = g
                while h < P and sizes[h] == 1 and h - g < chunk_c:
                    h += 1
                cols = np.fromiter((players[j][0] for j in range(g, h)), dtype=np.intp, count=h - g)
                diff = np.abs(R[:, cols][:, :, None] - self.T[:, cols].T[None, :, :])
                out[:, g:h, :] = self._pow(diff).astype(self.dtype, copy=False)
                g = h
            else:
                out[:, g, :] = self.term_sum(R, players[g])
                g += 1
        return out

    def rank_distances(self, R):
        """(r, n) full rank distances from rows `R` to every training row."""
        return self.term_sum(R, np.arange(self.M))

    # ---- prediction from distances ------------------------------------
    def values(self, D):
        """Predictions for stacked rank-distance rows `D` (m, n) -> (m,).
        The value of a row never depends on the other rows of the batch."""
        D = np.asarray(D)
        if D.ndim == 1:
            D = D[None, :]
        m, n = D.shape
        if n != self.n:
            raise ValueError(f"KNNDistanceGame.values: expected {self.n} distance columns, got {n}.")
        out = np.empty(m, dtype=float)
        step = max(1, _VALUES_BLOCK_ELEMENTS // n)
        for start in range(0, m, step):
            out[start:start + step] = self._values_block(D[start:start + step])
        return out

    def _values_block(self, D):
        m, n = D.shape
        k = self.k
        kth = np.partition(D, k - 1, axis=1)[:, k - 1]
        less = D < kth[:, None]
        tie = D == kth[:, None]
        need = k - less.sum(axis=1)
        ambiguous = tie.sum(axis=1) > need
        if ambiguous.any():                       # lowest training-row index wins the tie
            amb = np.flatnonzero(ambiguous)
            ta = tie[amb]
            tie[amb] = ta & (np.cumsum(ta, axis=1) <= need[amb, None])
        if self.weights == 'uniform':
            sel = less | tie
            num = sel @ self._wy
            den = (sel @ self._w) if self.sample_weight is not None else np.full(m, float(k))
        else:
            sel = less | tie
            rows, cols = np.nonzero(sel)
            cols = cols.reshape(m, k)
            dr = D[rows, cols.reshape(-1)].astype(float).reshape(m, k)
            dist = np.sqrt(dr) if self.p == 2.0 else dr ** (1.0 / self.p)
            with np.errstate(divide='ignore'):
                inv = 1.0 / dist
            exact = ~np.isfinite(inv)
            has_exact = exact.any(axis=1)
            w = np.where(np.isfinite(inv), inv, 0.0)
            if has_exact.any():
                w[has_exact] = exact[has_exact].astype(float)
            if self.sample_weight is not None:
                w = w * self.sample_weight[cols]
            num = (w * self.y[cols]).sum(axis=1)
            den = w.sum(axis=1)
        safe = np.where(den > 0, den, 1.0)
        return np.where(den > 0, num / safe, self._y_mean)


# ---------------------------------------------------------------------------
# players
# ---------------------------------------------------------------------------
def make_marker_blocks(n_markers, n_players, order=None):
    """Contiguous marker groups ("players").

    `order` is the genome-order permutation of the column indices (identity
    when None) - groups are contiguous windows of that order, so neighbouring
    markers (and so, mostly, markers in LD) share a player. `n_players` None,
    'all' or >= n_markers gives one player per marker.
    Returns a list of 1-D intp arrays of column indices."""
    order = np.arange(n_markers) if order is None else np.asarray(order, dtype=np.intp).reshape(-1)
    if order.shape[0] != n_markers:
        raise ValueError(f"make_marker_blocks: order has {order.shape[0]} entries for {n_markers} markers.")
    if n_players in (None, 'all') or int(n_players) >= n_markers:
        return [order[i:i + 1] for i in range(n_markers)]
    g = max(1, int(n_players))
    return [chunk for chunk in np.array_split(order, g) if chunk.size]


class KNNShapleyDetails(dict):
    """Diagnostics dict returned by `knn_shapley_effects(return_details=True)`."""


# ---------------------------------------------------------------------------
# permutation walks
# ---------------------------------------------------------------------------
def _run_walk_batch(game, AX, AZ, DZ, FZ, FX, row_ids, z_ids, perms):
    """One batch of antithetic walks. Each walk = (explained row, background
    row, permutation of the P players); the reversed permutation is walked too
    and the two marginal-contribution vectors are averaged. Returns (Wp, P)
    marginal contributions (column = player id). Efficiency holds exactly:
    each row sums to FX[row] - FZ[z]."""
    Wp, P = perms.shape
    out = np.zeros((Wp, P), dtype=float)
    ar = np.arange(Wp)
    for direction in (0, 1):
        order = perms if direction == 0 else perms[:, ::-1]
        D = DZ[z_ids].copy()
        prev = FZ[z_ids]
        for s in range(P):
            g = order[:, s]
            D += AX[row_ids, g]
            D -= AZ[z_ids, g]
            cur = FX[row_ids] if s == P - 1 else game.values(D)
            out[ar, g] += cur - prev
            prev = cur
    out *= 0.5
    return out


def _shapley_stage(game, X_explain, X_background, players, *, max_pairs, min_pairs, rel_tol,
                   top_q, rng, max_bytes, log_prefix, label):
    """Permutation-sampling Shapley values of `players` for every explained row.
    Returns (phi (r,P) signed mean marginal, se (r,P), pairs_used (r,), meta)."""
    r, B, P, n = X_explain.shape[0], X_background.shape[0], len(players), game.n
    itemsize = np.dtype(game.dtype).itemsize
    az_bytes = B * P * n * itemsize
    if az_bytes > 0.5 * max_bytes:
        raise MemoryError(
            f"{label}: the background table needs {az_bytes / 2 ** 30:.2f} GiB "
            f"({B} background x {P} players x {n} training rows) - above the "
            f"{0.5 * max_bytes / 2 ** 30:.2f} GiB budget. Use fewer marker groups, fewer "
            f"background rows or a smaller training panel.")
    AZ = game.player_terms(X_background, players)                       # (B, P, n)
    DZ = AZ.sum(axis=1, dtype=np.float64).astype(game.dtype)            # (B, n)
    FZ = game.values(DZ)                                                # (B,)
    FX_all = game.values(game.rank_distances(X_explain))                # (r,)

    row_budget = max(1, int((max_bytes - az_bytes) // max(1, P * n * itemsize)))
    chunk_rows = int(min(r, row_budget))
    if chunk_rows < 1:
        raise MemoryError(f"{label}: not enough memory budget for even one explained row.")

    phi = np.zeros((r, P))
    se = np.zeros((r, P))
    pairs_used = np.zeros(r, dtype=np.int64)
    z_perm = rng.permutation(B)
    z_offset = 0
    rel_err_last = float('nan')
    for c0 in range(0, r, chunk_rows):
        c1 = min(r, c0 + chunk_rows)
        rc = c1 - c0
        AX = game.player_terms(X_explain[c0:c1], players)                # (rc, P, n)
        FX = FX_all[c0:c1]
        sums = np.zeros((rc, P))
        sumsq = np.zeros((rc, P))
        done = 0
        q_cap = max(1, int(_WALK_BATCH_ELEMENTS // max(1, rc * n)))      # pairs per vectorised batch
        round_target = min_pairs                                         # then doubling
        while done < max_pairs:
            target = min(max_pairs, round_target)
            while done < target:
                q = min(q_cap, target - done)
                Wp = rc * q
                row_ids = np.repeat(np.arange(rc), q)
                z_ids = z_perm[(z_offset + np.arange(Wp)) % B]
                z_offset += Wp
                perms = rng.permuted(np.tile(np.arange(P), (Wp, 1)), axis=1)
                res = _run_walk_batch(game, AX, AZ, DZ, FZ, FX, row_ids, z_ids, perms)
                res = res.reshape(rc, q, P)
                sums += res.sum(axis=1)
                sumsq += (res * res).sum(axis=1)
                done += q
            if done >= max_pairs:
                break
            mean = sums / done
            var = np.maximum(sumsq / done - mean * mean, 0.0) * (done / max(done - 1.0, 1.0))
            score = np.abs(mean).sum(axis=0)
            err = np.sqrt((var / done).sum(axis=0))
            if score.max() <= 0.0:
                break
            top = np.argsort(-score)[:min(top_q, P)]
            rel_err_last = float(np.median(err[top] / np.maximum(score[top], 1e-300)))
            if rel_err_last <= rel_tol:
                break
            round_target = done * 2
        mean = sums / done
        var = np.maximum(sumsq / done - mean * mean, 0.0) * (done / max(done - 1.0, 1.0))
        phi[c0:c1] = mean
        se[c0:c1] = np.sqrt(var / done)
        pairs_used[c0:c1] = done
        del AX
    meta = {'players': P, 'pairs': int(pairs_used.max()), 'pairs_min': int(pairs_used.min()),
            'rel_err': rel_err_last, 'row_chunks': int(-(-r // chunk_rows)),
            'bytes_AZ': az_bytes}
    return phi, se, pairs_used, meta


def _stage_scores(phi, se, reduce, denoise=True):
    """(P,) cross-row scores and a (P,) standard-error estimate for them.

    With `denoise` the folded-normal inflation of |phi_hat| is removed: for a
    noisy estimate E[phi_hat**2] = phi**2 + se**2, so |phi| is estimated by
    sqrt(max(phi_hat**2 - se**2, 0)). Without it a marker with NO effect still
    scores ~0.8 * se per explained row (the noise floor), which would rank
    ahead of a marker that was never resolved. `denoise=False` reproduces the
    plain sum of |phi_hat|."""
    var = se * se
    if reduce == 'abs_then_sum':
        if denoise:
            score = np.sqrt(np.maximum(phi * phi - var, 0.0)).sum(axis=0)
        else:
            score = np.abs(phi).sum(axis=0)
    else:
        tot = phi.sum(axis=0)
        if denoise:
            score = np.sqrt(np.maximum(tot * tot - var.sum(axis=0), 0.0))
        else:
            score = np.abs(tot)
    err = np.sqrt(var.sum(axis=0))
    return score, err


def _significance(phi, se, z):
    """(P,) chi-square style statistic of 'every explained row has phi == 0'
    and a boolean flag: statistic above its null mean (= rows with a usable
    standard error) by `z` null standard deviations."""
    usable = se > 0.0
    t = np.where(usable, (phi / np.where(usable, se, 1.0)) ** 2, 0.0)
    stat = t.sum(axis=0)
    n_use = usable.sum(axis=0)
    flag = stat > (n_use + float(z) * np.sqrt(2.0 * np.maximum(n_use, 1)))
    return stat, flag


# ---------------------------------------------------------------------------
# public estimator
# ---------------------------------------------------------------------------
def knn_shapley_effects(X_train, y_train, X_explain, X_background, *, n_neighbors,
                        weights='uniform', p=2, sample_weight=None, max_walks=200,
                        min_walks=24, n_players=256, refine_markers=1000, branching=8,
                        z_split=3.0, rel_tol=0.10, top_q=50, denoise=True,
                        reduce='abs_then_sum', marker_order=None, seed=0, max_bytes=2.0e9,
                        log_prefix='[KNN]', return_details=False, zero_null_markers=True):
    """Per-marker Shapley effects of a k-NN regressor (see the module docstring).

    Parameters
    ----------
    X_train (n, M), y_train (n,) : the training panel the k-NN is fitted on.
    X_explain (r, M) : rows to explain.
    X_background (B, M) : REAL training rows used as the "absent" reference.
    n_neighbors, weights, p, sample_weight : the fitted k-NN's own settings.
    max_walks : permutation walks per explained row and level (forward +
        reversed counted separately); an upper bound - adaptive stopping may use
        fewer.
    min_walks : walks always run before the adaptive criterion is consulted.
    n_players : number of marker groups at level 0 ('all'/None/>=M: one player
        per marker, a single flat level - exact marker-level Shapley, slowest).
    refine_markers : budget of extra single-player slots - the hierarchy stops
        splitting once the player count would exceed `n_players + refine_markers`.
    branching : a significant group is split into this many contiguous sub-groups
        per level (8: a 40-marker group reaches single markers in two levels).
    z_split : a group is split only if its chi-square statistic exceeds the null
        mean by `z_split` null standard deviations (see `_significance`).
    rel_tol : stop a level once the median relative standard error of its
        top-`top_q` players is below this (0 disables adaptive stopping).
    denoise : remove the noise-floor inflation of sum |phi_hat| (see
        `_stage_scores`).
    reduce : 'abs_then_sum' | 'sum_then_abs' (cross-row reduction).
    marker_order : genome-order permutation of the columns (groups are
        contiguous windows of it); identity when None.
    seed : walk RNG seed (deterministic for fixed inputs).
    max_bytes : working-memory budget for the distance tables.
    zero_null_markers : True (default) - a marker that provably cannot change any
        distance ranking (constant across the training, explained and background
        rows; or, for `weights='uniform'`, constant across the training rows) is a
        NULL player, so its Shapley value is exactly 0. Without this, a null
        marker inside an unrefined group inherited an equal share of that group's
        score and showed up as a spurious effect (circos: coloured windows in the
        KNN ring that every other model leaves blank). The group's mass is
        re-shared over its non-null members only. False reproduces the previous
        (ver4-67) output exactly.

    Returns
    -------
    scores (M,) float ndarray (never NaN); with `return_details=True` also a
    `KNNShapleyDetails` dict: `resolved` (markers that were individually
    estimated), `score_se`, per-level metadata, timings, dtype.
    """
    t0 = time.time()
    if reduce not in ('abs_then_sum', 'sum_then_abs'):
        raise ValueError(f"knn_shapley_effects: reduce must be 'abs_then_sum' or 'sum_then_abs', got {reduce!r}.")
    Xe = np.atleast_2d(np.asarray(X_explain, dtype=float))
    Xb = np.atleast_2d(np.asarray(X_background, dtype=float))
    game = KNNDistanceGame(X_train, y_train, n_neighbors, weights, p, sample_weight,
                           extra_arrays=(Xe, Xb))
    M = game.M
    if Xe.shape[1] != M or Xb.shape[1] != M:
        raise ValueError(
            f"knn_shapley_effects: X_train has {M} columns, X_explain {Xe.shape[1]}, "
            f"X_background {Xb.shape[1]} - all three must describe the SAME markers.")
    if Xe.shape[0] == 0 or Xb.shape[0] == 0 or M == 0:
        z = np.zeros(M)
        return (z, KNNShapleyDetails(mode='empty')) if return_details else z
    rng = np.random.default_rng(seed)
    max_pairs = max(1, -(-int(max_walks) // 2))
    min_pairs = max(1, min(max_pairs, -(-int(min_walks) // 2)))
    tol = float(rel_tol) if rel_tol else -1.0          # -1: never stop early
    branching = max(2, int(branching))

    players = make_marker_blocks(M, n_players, marker_order)
    flat = len(players) == M
    player_cap = len(players) + max(0, int(refine_markers))
    det = KNNShapleyDetails(mode='flat' if flat else 'hierarchical', dtype=str(game.dtype),
                            n_markers=M, explained_rows=int(Xe.shape[0]),
                            background_rows=int(Xb.shape[0]), train_rows=int(game.n),
                            groups=len(players), levels=[])
    score = np.zeros(M)
    err = np.zeros(M)
    resolved = np.zeros(M, dtype=bool)
    level = 0
    while True:
        pairs = max_pairs if (flat or level > 0) else max(min_pairs, -(-max_pairs // 4))
        phi, se, _, meta = _shapley_stage(
            game, Xe, Xb, players, max_pairs=pairs, min_pairs=min(min_pairs, pairs),
            rel_tol=tol, top_q=top_q, rng=rng, max_bytes=max_bytes, log_prefix=log_prefix,
            label=f'knn_shapley[level {level}]')
        sc, er = _stage_scores(phi, se, reduce, denoise)
        for g, idx in enumerate(players):               # latest level wins
            score[idx] = sc[g] / idx.size
            err[idx] = er[g] / idx.size
            resolved[idx] = idx.size == 1
        stat, flag = _significance(phi, se, z_split)
        sizes = np.fromiter((idx.size for idx in players), dtype=np.int64, count=len(players))
        cand = np.flatnonzero(flag & (sizes > 1))
        meta.update(level=level, significant=int(flag.sum()), splittable=int(cand.size))
        det['levels'].append(meta)
        if flat or cand.size == 0:
            break
        cand = cand[np.argsort(-stat[cand])]
        n_players_now = len(players)
        chosen = []
        for g in cand:                                  # strongest first, within the player budget
            extra = min(branching, int(sizes[g])) - 1
            if n_players_now + extra > player_cap:
                break
            chosen.append(int(g))
            n_players_now += extra
        if not chosen:
            break
        chosen_set = set(chosen)
        nxt = []
        for g, idx in enumerate(players):
            if g in chosen_set:
                nxt.extend(c for c in np.array_split(idx, min(branching, idx.size)) if c.size)
            else:
                nxt.append(idx)
        players = nxt
        level += 1

    n_null = 0
    if zero_null_markers:
        _Xt = np.atleast_2d(np.asarray(X_train, dtype=float))
        _joint_const = np.ptp(np.vstack([_Xt, Xe, Xb]), axis=0) == 0.0
        _train_const = np.ptp(_Xt, axis=0) == 0.0
        # Constant over the training rows adds the SAME |a_m - c|**p to every
        # training-row distance: ranking and ties are unchanged for uniform
        # weights (distance weights 1/D would change, so they need `_joint_const`).
        null_mask = _joint_const | (_train_const if weights == 'uniform' else False)
        n_null = int(null_mask.sum())
        if n_null:
            score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
            for idx in players:
                nm = null_mask[idx]
                if not nm.any():
                    continue
                mass = float(score[idx].sum())
                score[idx] = 0.0
                err[idx] = 0.0
                keep = idx[~nm]
                if keep.size and idx.size > 1:               # re-share a group's mass over non-null members
                    score[keep] = mass / keep.size
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    det.update(null_markers=n_null, resolved=resolved, resolved_markers=int(resolved.sum()), score_se=err,
               seconds=time.time() - t0, final_players=len(players))
    if return_details:
        return score, det
    return score


# ---------------------------------------------------------------------------
# exact reference (small M) - used by the bench and the unit checks
# ---------------------------------------------------------------------------
def knn_shapley_exact(X_train, y_train, X_explain, X_background, *, n_neighbors,
                      weights='uniform', p=2, sample_weight=None, max_markers=16):
    """Exact Shapley values phi (r, M) of the game v_x(S) = mean_z f(x_S, z_notS),
    by enumeration of all 2**M coalitions. For testing only (M <= `max_markers`)."""
    Xe = np.atleast_2d(np.asarray(X_explain, dtype=float))
    Xb = np.atleast_2d(np.asarray(X_background, dtype=float))
    game = KNNDistanceGame(X_train, y_train, n_neighbors, weights, p, sample_weight,
                           dtype=np.float64)
    M = game.M
    if M > max_markers:
        raise ValueError(f"knn_shapley_exact: M={M} > {max_markers} - enumeration is infeasible.")
    n_coal = 1 << M
    idx = np.arange(n_coal)
    popcount = np.zeros(n_coal, dtype=np.int64)
    for m in range(M):
        popcount += (idx >> m) & 1
    from math import factorial
    wt = np.array([factorial(s) * factorial(M - s - 1) / factorial(M) for s in range(M)])
    players = [np.array([m]) for m in range(M)]
    AZ = game.player_terms(Xb, players)                                    # (B, M, n)
    DZ = AZ.sum(axis=1)
    phi = np.zeros((Xe.shape[0], M))
    for i in range(Xe.shape[0]):
        AX = game.player_terms(Xe[i:i + 1], players)[0]                    # (M, n)
        v = np.zeros(n_coal)
        for b in range(Xb.shape[0]):
            delta = AX - AZ[b]                                             # (M, n)
            D = DZ[b][None, :].repeat(1, axis=0)                           # (1, n)
            for m in range(M):                                             # doubling: bit m of the index
                D = np.concatenate([D, D + delta[m][None, :]], axis=0)
            v += game.values(D)
        v /= Xb.shape[0]
        for m in range(M):
            without = idx[((idx >> m) & 1) == 0]
            phi[i, m] = float(np.sum(wt[popcount[without]] * (v[without | (1 << m)] - v[without])))
    return phi


# ---------------------------------------------------------------------------
# closed-form hybrid-row evaluator for baseline_shap_interactions (KNN)
# ---------------------------------------------------------------------------
class KNNBaselineShapFastEval:
    """Drop-in `kernel_fast_eval` for
    `models.interaction_extraction.baseline_shap_interactions` /
    `baseline_shap_effects`, for a k-NN regressor.

    A hybrid row `x^{i->z}` (explained row x with marker i taken from background
    row z) changes the training-row distances by exactly

        D'(t) = D_x(t) - d_i(x, t) + d_i(z, t),

    so its prediction costs O(n_train) instead of the O(n_train * M) of
    materialising the row and calling `predict`. `begin_row` computes D_x once
    per explained row. Every method returns what
    `models.KNN.DeterministicKNNPredict` returns for the same hybrid rows
    (identical for integer-coded markers; floating-point-rounding level
    otherwise). Picklable (plain arrays) so it travels to worker processes.

    Parameters mirror `DeterministicKNNPredict`, plus `X_background` - the SAME
    background rows passed to `baseline_shap_interactions` (`bg_rows` are
    positional indices into it).
    """

    def __init__(self, X_ref, y_ref, X_background, n_neighbors, weights='uniform', p=2,
                 sample_weight=None):
        # dtype left to the game's own exactness rule (float32 when every
        # distance sum is exactly representable - integer genotypes - else
        # float64): half the memory traffic of the (n_eval, n) work arrays.
        self.game = KNNDistanceGame(X_ref, y_ref, n_neighbors, weights, p, sample_weight,
                                    extra_arrays=(np.asarray(X_background, dtype=float),))
        # marker-major copy: gathering the rows of (M, n) is a contiguous copy,
        # gathering columns of (n, M) is a strided one (the dominant cost).
        self._TT = np.ascontiguousarray(self.game.T.T.astype(self.game.dtype))
        self.X_background = np.ascontiguousarray(np.asarray(X_background, dtype=float))
        if self.X_background.ndim != 2 or self.X_background.shape[1] != self.game.M:
            raise ValueError("KNNBaselineShapFastEval: X_background must have the same columns as X_ref.")

    def working_bytes_per_eval(self):
        """Approximate peak working memory (bytes) of ONE hybrid evaluation: a few
        (n_ref,) float64 temporaries plus the comparison masks."""
        return 10 * 8 * int(self.game.n)

    # per-row cache of the explained row's own per-marker distance terms
    # (M, n) is kept only while it is small (<= this many elements).
    _TX_MAX_ELEMENTS = 40_000_000

    def begin_row(self, x):
        """Opaque per-explained-row cache: the full rank-distance row D_x and (when
        small enough) the (M, n) table of x's own per-marker terms."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        D = self.game.rank_distances(x)[0]
        Tx = None
        if self.game.M * self.game.n <= self._TX_MAX_ELEMENTS:
            xv = np.asarray(x[0], dtype=self._TT.dtype)[:, None]
            Tx = self.game._pow(np.abs(xv - self._TT))
        return (D, Tx)

    def full(self, x, row_cache):
        return float(self.game.values(row_cache[0][None, :])[0])

    def _term(self, vals, cols):
        # (n_eval, n): |vals[t] - T[:, cols[t]]|**p
        diff = np.abs(np.asarray(vals, dtype=self._TT.dtype)[:, None] - self._TT[cols])
        return self.game._pow(diff)

    def _x_term(self, x, row_cache, cols):
        Tx = row_cache[1]
        return Tx[cols] if Tx is not None else self._term(x[cols], cols)

    def single_knockout(self, x, row_cache, cols, bg_rows):
        x = np.asarray(x, dtype=float)
        cols = np.asarray(cols, dtype=np.intp)
        bg_rows = np.asarray(bg_rows, dtype=np.intp)
        D = row_cache[0][None, :] - self._x_term(x, row_cache, cols)
        D += self._term(self.X_background[bg_rows, cols], cols)
        return self.game.values(D)

    def joint_knockout(self, x, row_cache, i_arr, j_arr, bg_rows):
        x = np.asarray(x, dtype=float)
        i_arr = np.asarray(i_arr, dtype=np.intp)
        j_arr = np.asarray(j_arr, dtype=np.intp)
        bg_rows = np.asarray(bg_rows, dtype=np.intp)
        D = row_cache[0][None, :] - self._x_term(x, row_cache, i_arr)
        D -= self._x_term(x, row_cache, j_arr)
        D += self._term(self.X_background[bg_rows, i_arr], i_arr)
        D += self._term(self.X_background[bg_rows, j_arr], j_arr)
        return self.game.values(D)
