"""
tools/knn_shapley_bench.py
==========================
ver4-67 offline checks for models/knn_shapley.py (never imported by the engine).

  [K1] KNNDistanceGame.values == models.KNN.DeterministicKNNPredict (uniform/
       distance, p=1/2/3, with/without BLUE weights, 0/1/2 genotypes)
  [K2] knn_shapley_effects (flat) vs exhaustive enumeration (knn_shapley_exact)
       on a small marker panel: error relative to the largest score
  [K3] telescoping: every walk's contributions sum to f(x) - f(z) (checked through
       the estimator on a 1-row, 1-background problem: sum(phi) == f(x) - f(z))
  [K4] KNNBaselineShapFastEval vs DeterministicKNNPredict inside
       baseline_shap_interactions (dedup on/off): max |difference|
  [K5] timing of the hierarchical estimator at --m markers (default 12000)

Run from the project root:

    python tools/knn_shapley_bench.py --quick
    python tools/knn_shapley_bench.py --m 16000 --n 800 --bg 30 --explain 20

Exit status is 1 when a parity/exactness check exceeds its tolerance.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np                                                    # noqa: E402

from models.knn_shapley import (                                      # noqa: E402
    KNNDistanceGame, KNNBaselineShapFastEval, knn_shapley_effects, knn_shapley_exact,
)

FAILED = []


def _check(name, value, tol):
    ok = value <= tol
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {value:.3e} (tol {tol:.1e})")
    if not ok:
        FAILED.append(name)


def _geno(rng, n, m):
    return rng.integers(0, 3, size=(n, m)).astype(float)


def k1(rng):
    print("[K1] distance-game predictions vs DeterministicKNNPredict")
    from models.KNN import DeterministicKNNPredict
    worst = 0.0
    for weights in ('uniform', 'distance'):
        for p in (1, 2, 3):
            for sw in (None, rng.uniform(0.5, 1.5, 200)):
                X, y, Q = _geno(rng, 200, 60), rng.normal(size=200), _geno(rng, 50, 60)
                ref = DeterministicKNNPredict(X, y, 7, weights, p, sw)(Q)
                g = KNNDistanceGame(X, y, 7, weights, p, sw)
                worst = max(worst, float(np.abs(ref - g.values(g.rank_distances(Q))).max()))
    _check("max |diff|", worst, 1e-9)


def k2(rng):
    print("[K2] flat estimator vs exhaustive enumeration (10 markers)")
    n, M = 150, 10
    X = _geno(rng, n, M)
    y = X[:, 0] + 0.7 * X[:, 1] + 0.8 * X[:, 2] * X[:, 3] - 0.5 * X[:, 5] + rng.normal(0, .4, n)
    Xe, Xb = _geno(rng, 6, M), X[rng.choice(n, 12, replace=False)]
    exact = np.abs(knn_shapley_exact(X, y, Xe, Xb, n_neighbors=7)).sum(0)
    sc = knn_shapley_effects(X, y, Xe, Xb, n_neighbors=7, max_walks=1600, n_players='all', rel_tol=0)
    _check("max err / max score", float(np.abs(sc - exact).max() / exact.max()), 0.05)


def k3(rng):
    print("[K3] efficiency: sum_i phi_i = f(x) - f(z)")
    n, M = 120, 12
    X, y = _geno(rng, n, M), rng.normal(size=n)
    x, z = _geno(rng, 1, M), _geno(rng, 1, M)
    g = KNNDistanceGame(X, y, 5)
    fx, fz = g.values(g.rank_distances(x))[0], g.values(g.rank_distances(z))[0]
    ex = knn_shapley_exact(X, y, x, z, n_neighbors=5)
    _check("|sum phi - (f(x)-f(z))|", abs(float(ex.sum()) - float(fx - fz)), 1e-9)


def k4(rng):
    print("[K4] closed-form evaluator vs predict_fn in baseline_shap_interactions")
    from models.KNN import DeterministicKNNPredict
    from models.interaction_extraction import baseline_shap_interactions
    X = _geno(rng, 150, 12)
    y = X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(0, .5, 150)
    Xe, Xb = X[:6], X[20:32]
    names = [f"m{i}" for i in range(12)]
    pairs = [(a, b) for a in range(12) for b in range(a + 1, 12)]
    worst = 0.0
    for weights in ('uniform', 'distance'):
        pf = DeterministicKNNPredict(X, y, 5, weights, 2, None)
        fe = KNNBaselineShapFastEval(X, y, Xb, 5, weights=weights, p=2)
        for dedup in (True, False):
            a = baseline_shap_interactions(pf, Xe, Xb, names, pairs=pairs, n_jobs=1, dedup=dedup)
            b = baseline_shap_interactions(pf, Xe, Xb, names, pairs=pairs, n_jobs=1, dedup=dedup,
                                           kernel_fast_eval=fe)
            ia, ib = a.set_index(['marker1', 'marker2'])['value'], b.set_index(['marker1', 'marker2'])['value']
            worst = max(worst, float(np.abs(ia - ib.reindex(ia.index)).max()))
    _check("max |diff|", worst, 1e-9)


def k5(rng, n, m, bg, explain, quick):
    print(f"[K5] timing: n={n} train rows, M={m} markers, {explain} explained rows x {bg} background rows")
    block = 25
    X = _geno(rng, n + explain, m).astype(np.float32)
    for s in range(0, m - block, block):                 # LD-like blocks
        keep = rng.random((n + explain, block - 1)) < 0.7
        X[:, s + 1:s + block] = np.where(keep, X[:, [s]], X[:, s + 1:s + block])
    q = rng.choice(m, 15, replace=False)
    y = X[:n, q] @ rng.normal(size=15) + rng.normal(size=n)
    t = time.time()
    sc, det = knn_shapley_effects(X[:n], y, X[n:], X[rng.choice(n, bg, replace=False)],
                                  n_neighbors=7, max_walks=100 if quick else 200,
                                  return_details=True)
    print(f"  {time.time() - t:.1f}s; {int(np.sum(det['resolved']))} marker(s) individually resolved; "
          f"levels: {[(l['players'], l['pairs']) for l in det['levels']]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--m', type=int, default=12000)
    ap.add_argument('--bg', type=int, default=30)
    ap.add_argument('--explain', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    k1(rng); k2(rng); k3(rng); k4(rng)
    if a.quick:
        k5(rng, 300, 3000, 20, 8, True)
    else:
        k5(rng, a.n, a.m, a.bg, a.explain, False)
    print("FAILED: " + ", ".join(FAILED) if FAILED else "all checks passed")
    sys.exit(1 if FAILED else 0)


if __name__ == '__main__':
    main()
