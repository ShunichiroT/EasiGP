"""
tools/baseline_shap_bench.py
============================
ver4-43 (R1/R2) offline parity + timing bench for the `baseline_shap`
explanations of SVR, RKHS and KNN. Never imported by the engine.

What it checks (the blueprint's acceptance IDs in brackets), on synthetic
0/1/2 (or inbred 0/2) genotype fixtures:

  [P2]    level structure of the background (distinct values per column, the
          mean live level-combinations per pair c-bar, the dedup factor B/c-bar)
  [P4/S5] how often sklearn's KNN prediction depends on batch composition
  [A1.2]  `baseline_shap_interactions(dedup=True, <closed-form evaluator>)` vs
          the legacy loop (`dedup=False`): SVR rbf/poly/sigmoid, RKHS, KNN
  [A1.4]  n_jobs=1 vs n_jobs=N give the same numbers
  [A1.6]  a forced failure of the dedup engine / of the RKHS evaluator falls
          back with a NOTE (`--force-fail`)
  [A1.7]  DeterministicKNNPredict vs sklearn: explained rows and random
          hybrids; every mismatch must have a tie at the k-th distance
  [A2.1]  `baseline_shap_effects(dedup=True, fast)` vs `(dedup=False,
          predict_fn)`: SVR rbf/poly/sigmoid/linear, RKHS
  [A2.3]  speed of the new effect vs the per-row KernelExplainer (`--time-kernel-shap`)
  [P3/A2.6] the gated RKHS reconstruction is deterministic and the memo does
          not change `Marker_effect` / `Interaction` (needs the project's full
          import stack; skipped with a message when it is unavailable;
          `--memo-off` runs the reference without the memo and prints both)

Run from the project root:

    python tools/baseline_shap_bench.py --quick
    python tools/baseline_shap_bench.py --n 750 --m 500 --bg 50 --explain 10 --jobs 4

Exit status is 1 when any parity check exceeds its tolerance, else 0.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from models import interaction_extraction as ie           # noqa: E402
from models import kernel_effects as ke                   # noqa: E402
from models.KNN import DeterministicKNNPredict, WeightedKNeighborsRegressor  # noqa: E402

REL_TOL = 1e-6          # A1.2 / A2.1 tolerance (relative to the largest legacy value)
STRICT_TOL = 1e-12      # A1.4 / A2.6 tolerance


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def make_genotypes(n, m, coding, rng):
    """(n, m) float genotype matrix: independent 0/1/2 (coding '012') or inbred
    0/2 (coding '02'), plus a few blocks of correlated markers (LD)."""
    top = 3 if coding == '012' else 2
    X = rng.integers(0, top, size=(n, m)).astype(float)
    if coding == '02':
        X *= 2.0
    for start in range(0, m - 3, 25):                      # copy a marker into its neighbours
        for k in (1, 2):
            flip = rng.random(n) < 0.15
            X[:, start + k] = np.where(flip, X[:, start + k], X[:, start])
    return X


def make_phenotype(X, rng):
    y = 0.8 * X[:, 0] + 0.5 * X[:, 1] + 0.7 * X[:, 2] * X[:, 3] + 0.4 * X[:, 5] * X[:, 6]
    return y + rng.normal(0.0, 0.5, X.shape[0])


def fit_svr(kernel, X, y):
    from sklearn.svm import SVR
    return SVR(kernel=kernel, C=5.0, gamma=0.5 / X.shape[1], coef0=0.5, degree=2).fit(X, y)


def svr_fast_eval(svr, kernel, X_bg):
    return ke.SVRBaselineShapFastEval(
        kernel, svr.dual_coef_.ravel(), svr.support_vectors_, X_bg,
        intercept=float(np.asarray(svr.intercept_).reshape(-1)[0]),
        gamma=float(svr._gamma), coef0=float(svr.coef0), degree=int(svr.degree))


def make_rkhs(X_train, rng, h=1.0):
    """A fitted-looking `RKHSPredictor` (u = K @ z gives a well-conditioned alpha)."""
    d_scale = float(np.mean(ke.squared_euclidean_distance(X_train)))
    K = np.exp(-h * ke.squared_euclidean_distance(X_train) / d_scale)
    u = K @ rng.normal(0.0, 0.1, X_train.shape[0])
    return ke.RKHSPredictor(X_train, u, 0.3, h, d_scale)


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------
_FAILURES = []


def rel_diff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = max(float(np.abs(a).max()), 1e-300)
    return float(np.abs(a - b).max() / scale)


def top_set(values, frac):
    k = max(1, int(round(frac * len(values))))
    return set(np.argsort(-np.asarray(values), kind='stable')[:k].tolist())


def selection_identical(legacy, fast, frac=0.05):
    """Top-`frac` selection equal, ignoring pairs within REL_TOL*max of the cut."""
    legacy = np.asarray(legacy, dtype=float)
    fast = np.asarray(fast, dtype=float)
    k = max(1, int(round(frac * len(legacy))))
    cut = np.sort(legacy)[::-1][k - 1]
    band = REL_TOL * float(np.abs(legacy).max())
    ambiguous = np.abs(legacy - cut) <= band
    a = {i for i in top_set(legacy, frac) if not ambiguous[i]}
    b = {i for i in top_set(fast, frac) if not ambiguous[i]}
    return a == b


def check(name, value, tol):
    ok = value <= tol
    print(f"    {name:<46s} {value:10.3e}  (tol {tol:.0e})  {'OK' if ok else 'FAIL'}")
    if not ok:
        _FAILURES.append(f"{name}: {value:.3e} > {tol:.0e}")
    return ok


def header(text):
    print(f"\n=== {text} ===")


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def section_p2(X_bg, pairs, X_explain):
    header("P2  background level structure")
    lv = ie._background_levels(X_bg)
    counts = [(~np.isnan(row)).sum() for row in lv.values] if lv.mode == 'levels' else None
    print(f"    mode={lv.mode!r}, K={lv.values.shape[1]}, "
          f"distinct levels per column: "
          f"{'min %d / max %d' % (min(counts), max(counts)) if counts else 'n/a (rows mode)'}")
    i_arr = np.array([i for i, _ in pairs])
    j_arr = np.array([j for _, j in pairs])
    ctx = ie._DedupContext(
        lv, i_arr, j_arr,
        ie._build_joint_table(lv, i_arr, j_arr) if lv.mode == 'levels' else None,
        None, None, X_bg)
    print("    " + ie._dedup_cost_summary(ctx, X_explain, len(set(i_arr) | set(j_arr))))


def section_p4_a17(X, y, X_bg, rng, k=10):
    header("P4/S5 + A1.7  KNN: sklearn batch dependence and DeterministicKNNPredict")
    knn = WeightedKNeighborsRegressor(n_neighbors=k, weights='distance', p=2).fit(X, y)
    det = DeterministicKNNPredict(X, y, k, 'distance', 2)
    hybrids = X[rng.integers(0, X.shape[0], 10_000)].copy()
    donors = X_bg[rng.integers(0, X_bg.shape[0], 10_000)]
    cols = rng.integers(0, X.shape[1], 10_000)
    hybrids[np.arange(10_000), cols] = donors[np.arange(10_000), cols]
    batch = knn.predict(hybrids)
    alone = np.array([knn.predict(hybrids[i:i + 1])[0] for i in range(0, 2_000)])
    dep = float(np.mean(np.abs(batch[:2_000] - alone) > 1e-9))
    print(f"    sklearn: {100 * dep:.2f}% of 2,000 hybrids differ when predicted alone vs in a batch")
    mism = np.abs(knn.predict(hybrids) - det(hybrids)) > 1e-9
    d2 = ke.squared_euclidean_distance(hybrids[mism], X) if mism.any() else np.empty((0, X.shape[0]))
    srt = np.sort(d2, axis=1)
    no_tie = int((srt[:, k - 1] != srt[:, k]).sum()) if mism.any() else 0
    print(f"    det vs sklearn: {int(mism.sum())}/10,000 hybrids differ "
          f"({100 * mism.mean():.2f}%), of which WITHOUT a k-th-distance tie: {no_tie}")
    batch_dep = float(np.abs(det(hybrids[:500]) - np.array([det(hybrids[i:i + 1])[0] for i in range(500)])).max())
    check("DeterministicKNNPredict batch dependence", batch_dep, STRICT_TOL)
    check("mismatches without a k-th-distance tie", float(no_tie), 0.0)
    explained = X[:200] + 0.0
    check("explained rows (no exact-hit ties) rel diff",
          0.0 if not np.any(np.abs(knn.predict(explained) - det(explained)) > 1e-9)
          else rel_diff(knn.predict(explained), det(explained)), 1.0)


def run_interactions(label, predict_fn, X_explain, X_bg, names, pairs, fast_eval, legacy_eval, n_jobs):
    """A1.2 + A1.4 for one model. `legacy_eval` is what ver4-42 passed as
    kernel_fast_eval (SVR: the evaluator; RKHS/KNN: None)."""
    t0 = time.time()
    legacy = ie.baseline_shap_interactions(
        predict_fn, X_explain, X_bg, names, pairs=pairs, dedup=False, cost_log=False,
        kernel_fast_eval=legacy_eval, n_jobs=n_jobs)
    t_legacy = time.time() - t0
    t0 = time.time()
    fast = ie.baseline_shap_interactions(
        predict_fn, X_explain, X_bg, names, pairs=pairs, dedup=True, cost_log=False,
        kernel_fast_eval=fast_eval, n_jobs=n_jobs)
    t_fast = time.time() - t0
    print(f"  {label}: legacy {t_legacy:6.1f}s, fast {t_fast:6.1f}s  ({t_legacy / max(t_fast, 1e-9):.0f}x)")
    check(f"A1.2 {label} max|fast-legacy|/max|legacy|", rel_diff(legacy.value, fast.value), REL_TOL)
    ok = selection_identical(legacy.value.to_numpy(), fast.value.to_numpy())
    print(f"    {'A1.2 ' + label + ' top-5% selection identical':<46s} {'yes' if ok else 'NO':>10s}  "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        _FAILURES.append(f"{label}: top-5% selection differs")
    return legacy, fast


def section_interactions(args, rng):
    header("A1.2 / A1.4  baseline_shap interactions: legacy vs fast")
    X = make_genotypes(args.n, args.m, args.coding, rng)
    y = make_phenotype(X, rng)
    idx_bg = rng.choice(args.n, args.bg, replace=False)
    X_bg = X[idx_bg]
    X_explain = X[rng.choice(args.n, args.explain, replace=False)]
    names = [f"m{i}" for i in range(args.m)]
    pairs = [(i, j) for i in range(args.m) for j in range(i + 1, args.m)]
    print(f"  fixture: n={args.n}, M'={args.m}, B={args.bg}, n_explain={args.explain}, "
          f"pairs={len(pairs):,}, coding={args.coding}")
    section_p2(X_bg, pairs, X_explain)

    for kernel in ('rbf', 'poly', 'sigmoid'):
        svr = fit_svr(kernel, X, y)
        fe = svr_fast_eval(svr, kernel, X_bg)
        run_interactions(f"SVR-{kernel}", svr.predict, X_explain, X_bg, names, pairs, fe, fe, args.jobs)

    rkhs = make_rkhs(X, rng)
    baseline = X.mean(axis=0)
    shortlist = np.arange(args.m)
    masked = ke.RKHSMaskedPredictor(rkhs, baseline, shortlist)
    fe_rkhs = ke.rkhs_baseline_shap_fast_eval(rkhs, baseline, shortlist, X_bg)
    run_interactions("RKHS", masked.predict, X_explain, X_bg, names, pairs, fe_rkhs, None, args.jobs)

    det = DeterministicKNNPredict(X, y, 10, 'distance', 2)
    run_interactions("KNN(det)", det, X_explain, X_bg, names, pairs, None, None, args.jobs)
    knn = WeightedKNeighborsRegressor(n_neighbors=10, weights='distance', p=2).fit(X, y)
    a = ie.baseline_shap_interactions(knn.predict, X_explain, X_bg, names, pairs=pairs, dedup=False,
                                      cost_log=False, n_jobs=args.jobs)
    b = ie.baseline_shap_interactions(det, X_explain, X_bg, names, pairs=pairs, dedup=True,
                                      cost_log=False, n_jobs=args.jobs)
    print(f"    (informational) KNN legacy+sklearn vs fast+deterministic, rel diff = "
          f"{rel_diff(a.value, b.value):.3e}  - nonzero is the S5 tie noise removed")

    if args.jobs != 1:
        header("A1.4  n_jobs invariance (fast engine)")
        svr = fit_svr('rbf', X, y)
        fe = svr_fast_eval(svr, 'rbf', X_bg)
        one = ie.baseline_shap_interactions(svr.predict, X_explain, X_bg, names, pairs=pairs, dedup=True,
                                            cost_log=False, kernel_fast_eval=fe, n_jobs=1)
        many = ie.baseline_shap_interactions(svr.predict, X_explain, X_bg, names, pairs=pairs, dedup=True,
                                             cost_log=False, kernel_fast_eval=fe, n_jobs=args.jobs)
        check(f"A1.4 n_jobs=1 vs {args.jobs}", rel_diff(one.value, many.value), STRICT_TOL)

    if args.force_fail:
        header("A1.6  forced failures")
        svr = fit_svr('rbf', X, y)
        fe = svr_fast_eval(svr, 'rbf', X_bg)
        legacy = ie.baseline_shap_interactions(svr.predict, X_explain, X_bg, names, pairs=pairs, dedup=False,
                                               cost_log=False, kernel_fast_eval=fe)
        original = ie._dedup_row_deltas

        def _boom(*a, **k):
            raise RuntimeError("forced failure (bench --force-fail)")
        ie._dedup_row_deltas = _boom
        try:
            forced = ie.baseline_shap_interactions(svr.predict, X_explain, X_bg, names, pairs=pairs,
                                                   dedup=True, cost_log=False, kernel_fast_eval=fe)
        finally:
            ie._dedup_row_deltas = original
        check("A1.6 forced dedup failure -> legacy values", rel_diff(legacy.value, forced.value), 0.0)
        original_factory = ke.rkhs_baseline_shap_fast_eval
        try:
            ke.rkhs_baseline_shap_fast_eval = lambda *a, **k: (_ for _ in ()).throw(
                ValueError("forced evaluator failure (bench --force-fail)"))
            try:
                ke.rkhs_baseline_shap_fast_eval(rkhs, baseline, shortlist, X_bg)
            except ValueError as exc:
                print(f"    forced RKHS evaluator failure raised as expected: {exc}")
            # callers catch it and pass kernel_fast_eval=None; dedup still holds:
            no_fe = ie.baseline_shap_interactions(masked.predict, X_explain, X_bg, names, pairs=pairs,
                                                  dedup=True, cost_log=False, kernel_fast_eval=None)
        finally:
            ke.rkhs_baseline_shap_fast_eval = original_factory
        ref = ie.baseline_shap_interactions(masked.predict, X_explain, X_bg, names, pairs=pairs,
                                            dedup=False, cost_log=False)
        check("A1.6 evaluator failure, dedup on predict_fn", rel_diff(ref.value, no_fe.value), REL_TOL)
    return X, y, X_bg


def section_effects(args, rng):
    header("A2.1  baseline_shap effects: fast+dedup vs literal (dedup=False, predict_fn)")
    n_eff_markers = args.m_effect
    X = make_genotypes(args.n, n_eff_markers, args.coding, rng)
    y = make_phenotype(X, rng)
    X_bg = X[rng.choice(args.n, args.bg, replace=False)]
    X_explain = X[rng.choice(args.n, args.explain, replace=False)]
    print(f"  fixture: n={args.n}, M={n_eff_markers}, B={args.bg}, n_explain={args.explain}")
    for kernel in ('rbf', 'poly', 'sigmoid'):
        svr = fit_svr(kernel, X, y)
        fe = svr_fast_eval(svr, kernel, X_bg)
        t0 = time.time()
        fast = ie.baseline_shap_effects(svr.predict, X_explain, X_bg, kernel_fast_eval=fe, dedup=True,
                                        n_jobs=args.jobs, cost_log=False)
        t_fast = time.time() - t0
        t0 = time.time()
        literal = ie.baseline_shap_effects(svr.predict, X_explain, X_bg, dedup=False, n_jobs=args.jobs,
                                           cost_log=False)
        print(f"  SVR-{kernel}: fast {t_fast:.2f}s, literal {time.time() - t0:.2f}s")
        check(f"A2.1 SVR-{kernel}", rel_diff(literal, fast), REL_TOL)
    svr = fit_svr('linear', X, y)
    lin = ke.linear_baseline_shap_effects(svr.coef_.ravel(), X_explain, X_bg)
    literal = ie.baseline_shap_effects(svr.predict, X_explain, X_bg, dedup=False, cost_log=False)
    check("A2.1 SVR-linear (closed form)", rel_diff(literal, lin), REL_TOL)
    rkhs = make_rkhs(X, rng)
    baseline = X.mean(axis=0)
    shortlist = np.arange(n_eff_markers)
    masked = ke.RKHSMaskedPredictor(rkhs, baseline, shortlist)
    fe = ke.rkhs_baseline_shap_fast_eval(rkhs, baseline, shortlist, X_bg)
    fast = ie.baseline_shap_effects(masked.predict, X_explain, X_bg, kernel_fast_eval=fe, dedup=True,
                                    n_jobs=args.jobs, cost_log=False)
    literal = ie.baseline_shap_effects(masked.predict, X_explain, X_bg, dedup=False, n_jobs=args.jobs,
                                       cost_log=False)
    check("A2.1 RKHS", rel_diff(literal, fast), REL_TOL)
    check("A2.1 effects finite (count of non-finite)", float((~np.isfinite(fast)).sum()), 0.0)

    if args.time_kernel_shap:
        header("A2.3  baseline_shap effect vs per-row KernelExplainer (M <= 10,000)")
        import shap
        svr = fit_svr('rbf', X, y)
        fe = svr_fast_eval(svr, 'rbf', X_bg)
        t0 = time.time()
        ie.baseline_shap_effects(svr.predict, X_explain, X_bg, kernel_fast_eval=fe, dedup=True,
                                 n_jobs=args.jobs, cost_log=False)
        t_new = time.time() - t0
        background = shap.kmeans(pd.DataFrame(X), min(args.bg, 50))
        explainer = shap.KernelExplainer(svr.predict, background)
        t0 = time.time()
        explainer.shap_values(X_explain[:1], nsamples=200, l1_reg=False, silent=True)
        t_old = (time.time() - t0) * args.explain
        print(f"  baseline_shap {t_new:.2f}s vs kernel_shap ~{t_old:.1f}s (1 row timed x {args.explain}) "
              f"-> {t_old / max(t_new, 1e-9):.0f}x")
        check("A2.3 speed-up shortfall (20x target)", max(0.0, 20.0 - t_old / max(t_new, 1e-9)), 0.0)


def section_memo(args, rng):
    header("P3 / A2.6  gated RKHS reconstruction: determinism and memo equivalence")
    try:
        import genomic_prediction as gp
    except Exception as exc:                                  # noqa: BLE001 - optional import stack
        print(f"  skipped: genomic_prediction could not be imported here ({exc!r}).")
        return
    n_train, n_test, m = 120, 20, 60
    X = make_genotypes(n_train + n_test, m, args.coding, rng)
    y = make_phenotype(X, rng)
    cols = [f"m{i}" for i in range(m)]
    frame = pd.DataFrame(X, columns=cols)
    frame['y'] = y
    train, test, valid = frame.iloc[:n_train], frame.iloc[n_train:], frame.iloc[:0]
    d_scale = float(np.mean(ke.squared_euclidean_distance(train[cols].to_numpy())))
    K = np.exp(-1.0 * ke.squared_euclidean_distance(train[cols].to_numpy()) / d_scale)
    u = K @ rng.normal(0.0, 0.1, n_train)
    result = {'r_u': u, 'r_mu': np.array([0.3]), 'r_d_scale': np.array([d_scale]),
              'r_effect': pd.DataFrame(), 'r_y_predicted': None}
    recon = ke.RKHSPredictor(train[cols].to_numpy(), u, 0.3, 1.0, d_scale)
    predicted_test = recon.predict(test[cols].to_numpy())
    result['r_y_predicted'] = predicted_test
    params = [1200, 200, 1.0, True, 10, 'all', 1, 1, 0, 1, True, 'all', 20, 'all', 'off', 2.0, 3,
              'baseline_shap', 'baseline_shap', '2', 'k-SII', 4096, 20, 200, True]

    a, _ = gp._build_gated_rkhs_predictor(result, params, train, test, predicted_test, valid=valid)
    b, _ = gp._build_gated_rkhs_predictor(result, params, train, test, predicted_test, valid=valid)
    same = all(np.array_equal(np.asarray(getattr(a, k)), np.asarray(getattr(b, k)))
               for k in ('X_ref', 'alpha_ref', 'mu', 'h', 'd_scale'))
    print(f"  P3  two builds identical (X_ref, alpha_ref, mu, h, d_scale): {'yes' if same else 'NO'}")
    if not same:
        _FAILURES.append("P3: gated reconstruction is not deterministic")

    import pipeline_utils as pu
    pu.set_active_compute_resources(pu.resolve_compute_resources(
        {'ACCELERATION': {'baseline_shap_engine': 'fast'}}))
    outputs = {}
    for label, memo in (('memo', {}), ('no-memo', None)):
        eff = gp._apply_rkhs_shapiq_predictor_effect(
            None, result, params, train, valid, test, 'bench', predicted_test, {}, n_jobs=1,
            predictor_memo=memo)
        inter = gp._apply_rkhs_predictor_interaction(
            result, params, train, test, None, predicted_test, n_jobs=1, effect_values=eff,
            valid=valid, predictor_memo=memo)
        outputs[label] = (eff, inter.sort_values(['marker1', 'marker2']).reset_index(drop=True))
        if args.memo_off and label == 'no-memo':
            print("  (--memo-off: reference run without the memo shown above/below)")
    eff_diff = rel_diff(outputs['memo'][0].to_numpy(), outputs['no-memo'][0].to_numpy())
    int_diff = rel_diff(outputs['memo'][1]['value'], outputs['no-memo'][1]['value'])
    same_pairs = outputs['memo'][1][['marker1', 'marker2']].equals(outputs['no-memo'][1][['marker1', 'marker2']])
    check("A2.6 Marker_effect memo vs no-memo", eff_diff, 0.0)
    check("A2.6 Interaction memo vs no-memo", int_diff, 0.0)
    print(f"    A2.6 identical pair sets: {'yes' if same_pairs else 'NO'}; "
          f"effect finite columns: {int(np.isfinite(outputs['memo'][0].to_numpy()).sum())}/{m}")
    if not same_pairs:
        _FAILURES.append("A2.6: interaction pair sets differ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n', type=int, default=750, help='individuals in the fixture')
    parser.add_argument('--m', type=int, default=200, help="interaction shortlist size M' (all pairs scored)")
    parser.add_argument('--m-effect', type=int, default=2000, help='markers for the effect fixture')
    parser.add_argument('--bg', type=int, default=50, help='background rows B')
    parser.add_argument('--explain', type=int, default=10, help='explained rows')
    parser.add_argument('--jobs', type=int, default=1, help='n_jobs for the fan-out')
    parser.add_argument('--coding', choices=('012', '02'), default='012')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quick', action='store_true', help='small fixture (M\'=60, M=400, n=300)')
    parser.add_argument('--force-fail', action='store_true', help='run the A1.6 forced-failure checks')
    parser.add_argument('--memo-off', action='store_true', help='also report the no-memo RKHS reference run')
    parser.add_argument('--time-kernel-shap', action='store_true', help='time the A2.3 KernelExplainer baseline')
    args = parser.parse_args(argv)
    if args.quick:
        args.n, args.m, args.m_effect, args.explain, args.bg = 300, 60, 400, 6, 40

    rng = np.random.default_rng(args.seed)
    X, y, X_bg = section_interactions(args, rng)
    section_p4_a17(X, y, X_bg, rng)
    section_effects(args, rng)
    section_memo(args, rng)

    header("summary")
    if _FAILURES:
        print("  FAILED checks:")
        for line in _FAILURES:
            print(f"    - {line}")
        return 1
    print("  all parity checks within tolerance.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
