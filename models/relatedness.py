"""
models/relatedness.py
=====================
ver4-94 (R0 / R3, design blueprint ``EasiGP_ver4-94_Design_Blueprint.md``): the single place
that knows how related one set of lines is to another, and that builds the "source" lines the
population-pooled Super learner borrows (``SUPER_LEARNER['pop_sources']``).

Why this module exists
----------------------
* ``mean_kinship_score`` ranks candidate source lines by their mean genomic relationship to the
  target population's own train+valid lines (``pop_select='related'``). The blueprint's first
  formula standardised the markers on the train+valid rows alone. That makes the column means of
  the standardised train+valid rows exactly zero, so every score is exactly zero and the ranking
  is arbitrary (found in simulation ``ice_sim40``, amendment A11). The markers are therefore
  standardised on the UNION of source and train+valid lines, as
  ``ice_sim40_p1_pooled.py::relatedness_to_target`` does.
* ``build_population_source`` turns the other populations' labelled lines into one frame with
  the same layout as a task's own frames (markers first, phenotype last, I4), applying the
  guards that keep it from ever touching the task's own validation or test lines (I7).

Contract
--------
Pure functions: numpy / pandas only, no project imports (I2), nothing printed, no state kept.
The genotype reader is injected (``genotype_loader``), so the CSV and PLINK paths of
``genomic_prediction`` share this code and it can be tested without a ``GP()`` run.
"""

from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Fewer source lines than this cannot carry a useful pooled fit; the caller decides what to do
# (genomic_prediction.py writes a NOTE and no pooled row). Recorded here so every caller and the
# docstrings agree on the number.
MIN_SOURCE_LINES = 30


def standardise_on(ref: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Standardise genotype matrices with the mean and SD of the `ref` rows only.

    Columns whose SD in `ref` is zero are dropped from `ref` and from every matrix in `others`
    (a column with no variation carries no relationship information and would divide by zero).
    Returns ``(Z_ref, *Z_others)`` as float arrays. The values for the `ref` rows do not depend
    on `others` (blueprint AC0.3).

    Raises ValueError, naming the shapes, when a matrix is not 2-D, the column counts differ, or
    no column has variation."""
    ref = np.asarray(ref, dtype=float)
    mats = [np.asarray(o, dtype=float) for o in others]
    if ref.ndim != 2 or any(m.ndim != 2 for m in mats):
        raise ValueError("standardise_on needs 2-D matrices, got shapes "
                         f"{[ref.shape] + [m.shape for m in mats]}")
    for m in mats:
        if m.shape[1] != ref.shape[1]:
            raise ValueError(f"standardise_on: column counts differ, reference "
                             f"{ref.shape[1]} vs {m.shape[1]}")
    mean = ref.mean(axis=0)
    sd = ref.std(axis=0)
    ok = sd > 0
    if not ok.any():
        raise ValueError(f"standardise_on: no column has variation in the {ref.shape[0]} "
                         f"reference row(s)")
    out = [(ref[:, ok] - mean[ok]) / sd[ok]]
    out.extend((m[:, ok] - mean[ok]) / sd[ok] for m in mats)
    return tuple(out)


def max_kinship(Zq: np.ndarray, Zr: np.ndarray, *, mask: Optional[np.ndarray] = None,
                chunk: int = 2048, exclude_self: bool = False) -> np.ndarray:
    """Largest genomic relationship of every query line to the reference lines.

    ``k_i = max_j  z_i . z_j / M`` over the allowed reference lines j, with `Zq` (n_q x M) and
    `Zr` (n_r x M) ALREADY standardised on the reference rows (see :func:`standardise_on`; a
    value of about 1 means a near-duplicate, about 0 an unrelated line). Only genotypes enter,
    never a phenotype. The product is formed `chunk` query rows at a time, so memory is
    O(chunk x n_r), not O(n_q x n_r).

    Args:
        mask: optional boolean (n_q x n_r) array, True = that pair is NOT allowed (blueprint
            ver4-94 R0).
        exclude_self: when True, `Zq` and `Zr` must be the same lines in the same order and the
            pair (i, i) is skipped, giving the leave-one-out value k_i of a reference line.

    Returns a float vector of length n_q; a query line with no allowed reference line gets NaN.
    Raises ValueError on non-2-D input, a column-count mismatch, an empty set, non-finite
    values, a mask of the wrong shape, or ``exclude_self`` with different row counts."""
    Zq = np.asarray(Zq, dtype=float)
    Zr = np.asarray(Zr, dtype=float)
    if Zq.ndim != 2 or Zr.ndim != 2 or Zq.shape[1] != Zr.shape[1]:
        raise ValueError(f"max_kinship: expected two 2-D matrices with the same columns, got "
                         f"{Zq.shape} and {Zr.shape}")
    if Zq.shape[0] == 0 or Zr.shape[0] == 0 or Zq.shape[1] == 0:
        raise ValueError("max_kinship: an empty query set, reference set or marker set")
    if not (np.all(np.isfinite(Zq)) and np.all(np.isfinite(Zr))):
        raise ValueError("max_kinship: non-finite values (standardise first)")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (Zq.shape[0], Zr.shape[0]):
            raise ValueError(f"max_kinship: mask shape {mask.shape} does not match "
                             f"({Zq.shape[0]}, {Zr.shape[0]})")
    if exclude_self and Zq.shape[0] != Zr.shape[0]:
        raise ValueError("max_kinship: exclude_self needs the same lines in Zq and Zr")
    chunk = max(1, int(chunk))
    M = float(Zq.shape[1])
    out = np.empty(Zq.shape[0], dtype=float)
    for start in range(0, Zq.shape[0], chunk):
        stop = min(start + chunk, Zq.shape[0])
        sim = Zq[start:stop] @ Zr.T / M
        if mask is not None:
            sim[mask[start:stop]] = -np.inf
        if exclude_self:
            rows = np.arange(stop - start)
            sim[rows, start + rows] = -np.inf
        best = sim.max(axis=1)
        best[~np.isfinite(best)] = np.nan
        out[start:stop] = best
    return out


def mean_kinship_score(X_src: np.ndarray, X_tv: np.ndarray) -> np.ndarray:
    """Mean genomic relationship of every source line to the target's train+valid lines.

    `X_src` (n_src x M) and `X_tv` (n_tv x M) are RAW additive genotypes (0 / 1 / 2) over the
    same M marker columns. Markers are standardised on the union of both sets (the reason is in
    the module docstring), then ``score_j = z_j . mean_i(z_i) / M_used`` with ``z_i`` the
    train+valid rows. Larger = more related. Returns a float vector of length n_src. Only
    train+valid genotypes are used, never test genotypes or any phenotype.

    Raises ValueError on non-finite input, a column-count mismatch, an empty set or no column
    with variation."""
    X_src = np.asarray(X_src, dtype=float)
    X_tv = np.asarray(X_tv, dtype=float)
    if X_src.ndim != 2 or X_tv.ndim != 2 or X_src.shape[1] != X_tv.shape[1]:
        raise ValueError(f"mean_kinship_score: expected two 2-D matrices with the same columns, "
                         f"got {X_src.shape} and {X_tv.shape}")
    if X_src.shape[0] == 0 or X_tv.shape[0] == 0:
        raise ValueError("mean_kinship_score: no source or no train+valid rows")
    if not (np.all(np.isfinite(X_src)) and np.all(np.isfinite(X_tv))):
        raise ValueError("mean_kinship_score: non-finite genotype values")
    both = np.vstack([X_src, X_tv])
    mean = both.mean(axis=0)
    sd = both.std(axis=0)
    ok = sd > 0
    if not ok.any():
        raise ValueError("mean_kinship_score: no marker has variation")
    z_src = (X_src[:, ok] - mean[ok]) / sd[ok]
    z_tv_mean = ((X_tv[:, ok] - mean[ok]) / sd[ok]).mean(axis=0)
    return z_src @ z_tv_mean / float(ok.sum())


def build_population_source(pheno_all: pd.DataFrame, phenotype: str, own_pop: str,
                            own_ids: Dict[str, Set[str]], marker_cols: List[str],
                            y_tv_mean: float, X_tv: Optional[np.ndarray], settings: dict,
                            seed: int, genotype_loader: Callable[[List[str]], pd.DataFrame],
                            score_cols: Optional[List[str]] = None,
                            y_tv_sd: Optional[float] = None
                            ) -> Tuple[pd.DataFrame, dict]:
    """The other populations' labelled lines for one task, as a frame of ``marker_cols`` plus
    the phenotype column last (same layout as a task's own frames, I4).

    Args:
        pheno_all: phenotype table with columns ``ID``, ``population`` and `phenotype`.
        phenotype: name of the phenotype column of the task.
        own_pop: the task's own population (its lines are never source lines).
        own_ids: sets of str IDs under ``'train'``, ``'valid'``, ``'test'``; every one of them is
            dropped from the source (I7: a line of the task's own split never enters a fit
            through the source, whatever population label it carries).
        marker_cols: the marker columns the frame must carry, in order.
        y_tv_mean: mean phenotype of the task's own train+valid lines (the centring target).
        X_tv: raw genotypes of the task's own train+valid lines over `score_cols` (default
            `marker_cols`); needed only for ``pop_select='related'``.
        settings: a resolved SUPER_LEARNER dict; reads ``pop_select`` ('all' | 'related'),
            ``pop_max_lines`` (0 = no cap) and ``pop_center`` (bool).
        seed: the already combined seed (the caller adds the module offset and the replicate)
            used for the capped random draw of ``pop_select='all'``.
        genotype_loader: ``loader(ids) -> DataFrame`` with column ``ID`` plus every
            `marker_cols` column, free of NaN, one row per ID it could find.
        score_cols: the markers the relatedness score is computed on, a subset of `marker_cols`
            in the column order of `X_tv` (the markers every member shares); default
            `marker_cols`.
        y_tv_sd: standard deviation (ddof=1) of the task's own train+valid phenotype (pool-fix);
            read only when ``settings['pop_scale']`` is true (default false = no rescaling).

    Guards, in order, each counted in the returned ``diag``: other populations only and a
    non-missing phenotype; drop every own train / valid / test ID; keep the first of duplicated
    IDs; load genotypes (IDs without a genotype are dropped); centre each source population's
    phenotype to `y_tv_mean` (population mean shifts are trial effects, not marker effects);
    with ``pop_scale`` (pool-fix) first rescale each source population's phenotype about its own
    mean to the spread `y_tv_sd` of the own lines (a population measured on another scale or in a
    noisier trial would otherwise pull the shared marker effects towards its own scale; the mean
    is unchanged by the rescaling, so it composes with the centring); select (seeded cap, or the most related lines); sort by (population, ID) so the result does
    not depend on input order.

    Returns ``(src, diag)``. ``src`` has a fresh RangeIndex and float columns. ``diag`` holds
    ``n_candidates``, ``n_missing_pheno``, ``n_dropped_own``, ``n_duplicate``, ``n_no_genotype``,
    ``n_populations``, ``n_selected``, ``mode``, ``centre_shift`` {population: shift}, ``ids``
    and ``populations`` (lists in the row order of ``src``), ``note`` / ``reason`` when
    something worth saying happened and, for 'related', ``score_mean`` / ``score_top`` /
    ``score_bottom``. Having fewer than MIN_SOURCE_LINES lines left is NOT an error here: the
    frame is returned and ``diag['reason']`` says so, because the caller owns that decision.

    Raises ValueError only on a contract violation: a missing column in `pheno_all` or in what
    the loader returned, or ``pop_select='related'`` without `X_tv`."""
    for col in ('ID', 'population', phenotype):
        if col not in pheno_all.columns:
            raise ValueError(f"build_population_source: the phenotype table has no column "
                             f"{col!r} (columns: {list(pheno_all.columns)})")
    mode = str(settings.get('pop_select', 'all'))
    cap = int(settings.get('pop_max_lines', 0) or 0)
    centre = bool(settings.get('pop_center', True))
    scale = bool(settings.get('pop_scale', False))
    if mode == 'related' and X_tv is None:
        raise ValueError("build_population_source: pop_select='related' needs the train+valid "
                         "genotypes (X_tv)")
    diag: dict = {'mode': mode, 'n_candidates': 0, 'n_missing_pheno': 0, 'n_dropped_own': 0,
                  'n_duplicate': 0, 'n_no_genotype': 0, 'n_populations': 0, 'n_selected': 0,
                  'centre_shift': {}, 'scale_factor': {}, 'ids': [], 'populations': []}
    empty = pd.DataFrame(columns=list(marker_cols) + [phenotype])

    # 1. other populations, phenotype present
    table = pheno_all[['ID', 'population', phenotype]].copy()
    table['ID'] = table['ID'].astype(str)
    table['population'] = table['population'].astype(str)
    table = table[table['population'] != str(own_pop)]
    diag['n_candidates'] = int(table.shape[0])
    has_y = table[phenotype].notna()
    diag['n_missing_pheno'] = int((~has_y).sum())
    table = table[has_y]

    # 2. never a line of the task's own split
    own_all: Set[str] = set()
    for key in ('train', 'valid', 'test'):
        own_all |= {str(v) for v in own_ids.get(key, ())}
    is_own = table['ID'].isin(own_all)
    diag['n_dropped_own'] = int(is_own.sum())
    table = table[~is_own]

    # 3. one row per ID, first occurrence wins
    dup = table['ID'].duplicated(keep='first')
    diag['n_duplicate'] = int(dup.sum())
    table = table[~dup].sort_values(['population', 'ID'], kind='mergesort')
    if table.shape[0] == 0:
        diag['reason'] = "no source line is left after the guards"
        return empty, diag

    # 4. genotypes
    geno = genotype_loader(table['ID'].tolist())
    for col in ['ID'] + list(marker_cols):
        if col not in geno.columns:
            raise ValueError(f"build_population_source: the genotype loader did not return "
                             f"column {col!r}")
    geno = geno.copy()
    geno['ID'] = geno['ID'].astype(str)
    geno = geno.drop_duplicates(subset='ID', keep='first')
    merged = table.merge(geno[['ID'] + list(marker_cols)], on='ID', how='inner')
    diag['n_no_genotype'] = int(table.shape[0] - merged.shape[0])
    if merged.shape[0] == 0:
        diag['reason'] = "no source line has a genotype"
        return empty, diag

    # 5. per-population centring to the target's own mean
    y = merged[phenotype].to_numpy(dtype=float).copy()
    if scale:
        sd_own = None if y_tv_sd is None else float(y_tv_sd)
        if sd_own is None or not np.isfinite(sd_own) or sd_own <= 0.0:
            diag['scale_note'] = ("pop_scale is on but the own phenotype has no usable spread - "
                                  "the source phenotypes are not rescaled")
        else:
            for pop in sorted(merged['population'].unique()):
                rows = (merged['population'] == pop).to_numpy()
                sd_pop = float(y[rows].std(ddof=1)) if int(rows.sum()) > 1 else float('nan')
                if not np.isfinite(sd_pop) or sd_pop <= 0.0:
                    diag['scale_factor'][str(pop)] = float('nan')    # left as it is
                    continue
                factor = sd_own / sd_pop
                mean_pop = float(y[rows].mean())
                y[rows] = (y[rows] - mean_pop) * factor + mean_pop
                diag['scale_factor'][str(pop)] = float(factor)
    if centre:
        for pop in sorted(merged['population'].unique()):
            rows = (merged['population'] == pop).to_numpy()
            shift = float(y_tv_mean - y[rows].mean())
            y[rows] = y[rows] + shift
            diag['centre_shift'][str(pop)] = shift
    merged[phenotype] = y
    diag['n_populations'] = int(merged['population'].nunique())

    # 6. selection
    keep = np.arange(merged.shape[0])
    if mode == 'related':
        cols = list(score_cols) if score_cols is not None else list(marker_cols)
        score = mean_kinship_score(merged[cols].to_numpy(dtype=float), X_tv)
        diag['score_mean'] = float(score.mean())
        diag['score_top'] = float(np.sort(score)[-cap:].mean()) if cap > 0 else float(score.mean())
        diag['score_bottom'] = float(np.sort(score)[:cap].mean()) if cap > 0 else float(score.mean())
        if cap > 0:
            keep = np.sort(np.argsort(-score, kind='stable')[:cap])
        else:
            diag['note'] = ("related selection needs pop_max_lines > 0; all source lines are "
                            "used")
    elif cap > 0 and merged.shape[0] > cap:
        keep = np.sort(np.random.default_rng(int(seed)).choice(merged.shape[0], size=cap,
                                                               replace=False))
    merged = merged.iloc[keep]

    # 7. deterministic order
    merged = merged.sort_values(['population', 'ID'], kind='mergesort').reset_index(drop=True)
    src = merged[list(marker_cols) + [phenotype]].astype(float).reset_index(drop=True)
    diag['n_selected'] = int(src.shape[0])
    diag['ids'] = merged['ID'].tolist()
    diag['populations'] = merged['population'].tolist()
    if src.shape[0] < MIN_SOURCE_LINES:
        diag['reason'] = (f"only {src.shape[0]} source line(s) remain (fewer than "
                          f"{MIN_SOURCE_LINES})")
    return src, diag
