"""gat_search_check.py - ver4-46: offline equivalence / speed check of the lean GAT search fit.

Why: `models/gat_search.py` re-implements the TRAINING LOOP of the four non-biological GAT variants
so a hyperparameter-search candidate does not pay for outputs the search never reads. The claim is
that its validation predictions equal the normal fit's (same layers, same random draws in the same
order), and that the topology cache of `GAT_prior_knowledge` is exact. This script checks both on a
small synthetic problem and, with ``--time``, reports how much faster the lean fit is on a problem
size you choose. It fits no real data, writes nothing and is never imported by the engine.

Usage (from the EasiGP source directory):

  python tools/gat_search_check.py                       # equivalence checks only
  python tools/gat_search_check.py --time --n-train 150 --markers 60 --epochs 10

Exit status 0 = every check passed. One line per check on stdout.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from pipeline_utils import search_fit_context  # noqa: E402


class _W:
    """Stand-in for a BlueWeightContext: only ``w_fit`` is read by the GAT models."""

    def __init__(self, w_fit):
        self.w_fit = w_fit


def _data(n_train, n_valid, n_test, markers, seed):
    rng = np.random.default_rng(seed)
    n = n_train + n_valid + n_test
    geno = rng.integers(0, 3, size=(n, markers)).astype(float)
    beta = rng.normal(size=markers)
    pheno = geno @ beta / np.sqrt(markers) + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame(geno, columns=[f'm{i}' for i in range(markers)])
    frame['y'] = pheno
    return (frame.iloc[:n_train].reset_index(drop=True),
            frame.iloc[n_train:n_train + n_valid].reset_index(drop=True),
            frame.iloc[n_train + n_valid:].reset_index(drop=True))


def _params(model, epochs, bsize, heads, neuron=8, dropout=0.1):
    base = [neuron, dropout, 0.01, 0.0005, epochs, bsize, heads]
    if model == 'GAT_prior_knowledge':
        return base + [20, False, 5, False]            # top_rate, marker_effect, samples, emit
    return base + [False, 5]                           # marker_effect, samples


def _run(model, train, valid, test, params, w_fit, lean):
    from models.GAT_fully_connected import GAT_fully_connected
    from models.GAT_infinitesimal import GAT_infinitesimal
    from models.GAT_prior_knowledge import GAT_prior_knowledge
    fn = {'GAT_fully_connected': GAT_fully_connected, 'GAT_infinitesimal': GAT_infinitesimal,
          'GAT_prior_knowledge': GAT_prior_knowledge}[model]
    ctx = _W(w_fit) if w_fit is not None else None
    torch.manual_seed(0)
    np.random.seed(0)
    started = time.perf_counter()
    if lean:
        with search_fit_context(lean=True):
            out = fn(train, valid, test, list(params), sample_weight_ctx=ctx)
    else:
        out = fn(train, valid, test, list(params), sample_weight_ctx=ctx)
    return out, time.perf_counter() - started


def _valid_of(model, out):
    return np.asarray(out[4], dtype=float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--time', action='store_true', help='also time lean vs normal fits')
    ap.add_argument('--n-train', type=int, default=40)
    ap.add_argument('--markers', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--atol', type=float, default=1e-5)
    args = ap.parse_args(argv)

    train, valid, test = _data(args.n_train, max(8, args.n_train // 4), 8, args.markers, 1)
    failures = 0

    def check(name, ok, detail=''):
        nonlocal failures
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
        failures += 0 if ok else 1

    models = ['GAT_fully_connected', 'GAT_infinitesimal', 'GAT_prior_knowledge']
    for model in models:
        for bsize, heads, weighted in ((8, 1, False), (7, 2, False), (8, 2, True)):
            w = (np.linspace(0.5, 1.5, len(train)) if weighted else None)
            params = _params(model, args.epochs, bsize, heads)
            normal, t_normal = _run(model, train, valid, test, params, w, lean=False)
            lean, t_lean = _run(model, train, valid, test, params, w, lean=True)
            a, b = _valid_of(model, normal), _valid_of(model, lean)
            ok = a.shape == b.shape and np.allclose(a, b, atol=args.atol, rtol=1e-4)
            err = float(np.max(np.abs(a - b))) if a.shape == b.shape else float('nan')
            lean_shape_ok = (len(lean[3]) == 0 and len(lean[5]) == 0 and np.isnan(lean[0]))
            check(f"{model} bsize={bsize} heads={heads}{' weighted' if weighted else ''}: "
                  f"lean valid predictions == normal", ok and lean_shape_ok,
                  f"(max |diff| = {err:.2e}, lean returns no test/train output: {lean_shape_ok})")

    # topology cache is exact
    from models.gat_search import compute_topology, topology_cache_context, topology_for
    from models.GAT_prior_knowledge import _RF_N_ESTIMATORS, _SHAP_INTERACTION_SAMPLE_SIZE
    tx, ty = train.iloc[:, :-1], train.iloc[:, -1]
    plain = compute_topology(tx, ty, None, 20, 1, _RF_N_ESTIMATORS, _SHAP_INTERACTION_SAMPLE_SIZE)
    with topology_cache_context(True):
        first = topology_for(tx, ty, None, 20, n_jobs=1, rf_trees=_RF_N_ESTIMATORS,
                             shap_samples=_SHAP_INTERACTION_SAMPLE_SIZE)
        second = topology_for(tx, ty, None, 20, n_jobs=1, rf_trees=_RF_N_ESTIMATORS,
                              shap_samples=_SHAP_INTERACTION_SAMPLE_SIZE)
        other_rate = topology_for(tx, ty, None, 50, n_jobs=1, rf_trees=_RF_N_ESTIMATORS,
                                  shap_samples=_SHAP_INTERACTION_SAMPLE_SIZE)
    same = (np.array_equal(plain.pos1, first.pos1) and np.array_equal(plain.pos2, first.pos2)
            and np.allclose(plain.value, first.value))
    check("topology: cached result == fresh computation", same)
    check("topology: second call is a cache hit (same object)", second is first)
    check("topology: a different top_rate is not a hit", other_rate is not first
          and len(other_rate.value) > len(first.value))

    if args.time:
        print("\n  timing (seconds per fit; one fit each, after a warm-up fit):")
        for model in models:
            params = _params(model, args.epochs, 8, 2, neuron=16)
            _run(model, train, valid, test, params, None, lean=True)      # warm-up
            if model == 'GAT_prior_knowledge':
                with topology_cache_context(True):
                    _run(model, train, valid, test, params, None, lean=True)
                    _, t_lean = _run(model, train, valid, test, params, None, lean=True)
            else:
                _, t_lean = _run(model, train, valid, test, params, None, lean=True)
            _, t_normal = _run(model, train, valid, test, params, None, lean=False)
            print(f"  {model:<22} normal {t_normal:7.2f}s   lean {t_lean:7.2f}s   "
                  f"speed-up {t_normal / max(t_lean, 1e-9):5.1f}x")

    print(f"\n{'ALL CHECKS PASSED' if not failures else str(failures) + ' CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
