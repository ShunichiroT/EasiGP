"""stability_summary.py - stability of the selected markers and marker pairs across iterations.

REBUILT 2026-10-15 (ver4-103 recovery patch). The ver4-99 original of this module was not
available, so this is a re-implementation from the recorded design (ver4-93 R2 blueprint and the
ver4-99 change record), the call sites already in the tree (`main_app.py`, five report sites call
``run_stability_report(cfg, result_name, effect_fn, log)``), and the interface that
``tool_diagnostics.py`` / ``tools/consensus_stability.py`` expect. Names, columns and edge-case
rules below are therefore this rebuild's, not necessarily the original's; see the notes at the end
of this docstring.

Question answered
-----------------
EasiGP runs the same model list on many train/test splits ("iterations", the ``sample`` column).
Which markers (and which marker pairs) does the ensemble call important IN EVERY SPLIT, and which
ones only appear in a few? This is stability selection (Meinshausen & Buhlmann 2010) applied to
the consensus of the models: the share of iterations in which a marker is among the top K is its
selection frequency; frequencies near 1 are stable.

Definitions
-----------
* Task = one iteration = ``(asreml_option, population, phenotype, ratio, sample)``; a group =
  the same without ``sample``. Stability is computed inside a group, over its samples.
* Consensus relevance of a marker in a task = the mean over the chosen models of that model's
  absolute marker effect, L1-normalised within the model's row (so models on different scales
  count equally; the same normalisation the ensemble interaction routes use). The chosen models
  are, by default, every base model present: rows whose label is a combined prediction
  (``ensemble``, ``Super learner``, ``Iterative ensemble``, a weighting method) are left out so a
  model is not counted twice. A run that only has combined rows falls back to ``ensemble``.
* Top K: the K markers with the largest consensus relevance (ties broken by column order, so the
  result is deterministic). A marker the task does not report (NaN) is not a candidate there.
* Selection frequency = (number of tasks of the group whose top K contains the marker) / (number of
  tasks of the group); a marker never selected counts 0 and is not listed.
* Stability of a group = the mean pairwise Jaccard index of its tasks' top-K sets (1 = identical
  every iteration) and the mean Kuncheva consistency index (the same, corrected for the overlap
  expected by chance among the candidates; about 0 for random sets), plus how many markers were
  selected at least once and at frequency >= 0.5, >= 0.9 and = 1.
* Pairs: ``Interaction.csv`` rows of the chosen models are L1-normalised within (task, model),
  the pair is made order-free (marker1 <= marker2), and its task score is the mean over the
  models that reported it. Top K pairs per task, frequencies and Jaccard as for markers. The file
  is read in chunks; the result does not depend on the chunk size (sums are exact, nothing is
  pruned), only the peak memory does.
* ``bin_size`` > 1 (markers only) pools that many consecutive marker columns into one unit (a
  crude LD bin when the columns are in map order); the unit is named ``first:last``.

Outputs (derived reports in the result folder; they are not result files of the engine):
``Consensus_stability.csv`` (one row per group and level), ``Marker_selection_frequency.csv``,
``Pair_selection_frequency.csv`` and, when asked (X03), ``Interaction_selection_frequency.csv``:
the rows of ``Interaction.csv`` (every model) with three extra columns, ``pair_frequency`` (the
share of iterations of the row's group in which the pair was among the top-K consensus pairs),
``marker1_frequency`` and ``marker2_frequency`` (the same for each marker of the pair among the
top-K consensus markers); 0 when never selected. ``Interaction.csv`` itself is never changed and
the original columns and values are copied through as text unchanged.

Limits (read before relying on the numbers)
-------------------------------------------
* With few iterations a frequency can only take the values 0, 1/n ... 1 (4 iterations: steps of
  0.25); use 10 or more.
* The Jaccard index of top-K sets depends on K and on how many markers there are; it is a
  stability measure, not an accuracy measure (a marker can be stable and wrong).
* The splits overlap (they resample the same lines), so the iterations are not independent;
  frequencies are descriptive, not p-values.
* Pairs use the model's own pair table, which is already filtered by each model's top-N / top-%
  setting, so a pair that was filtered out in an iteration counts as not selected there.

Import discipline (I2): numpy / pandas at import time; the engine modules are imported lazily and
only by `run_stability_report`.
"""
from __future__ import annotations

import itertools
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Columns of Marker_effect.csv / Interaction.csv that are not markers, recognised BY NAME (the
#: optional leading ``asreml_option`` column of a multi-option run moves the marker block).
ID_COLUMNS = ('asreml_option', 'population', 'phenotype', 'model', 'ratio', 'sample')
TASK_COLUMNS = ('asreml_option', 'population', 'phenotype', 'ratio', 'sample')
GROUP_COLUMNS = ('asreml_option', 'population', 'phenotype', 'ratio')

CONSENSUS_FILE = 'Consensus_stability.csv'
MARKER_FREQUENCY_FILE = 'Marker_selection_frequency.csv'
PAIR_FREQUENCY_FILE = 'Pair_selection_frequency.csv'
INTERACTION_FREQUENCY_FILE = 'Interaction_selection_frequency.csv'     # X03

DEFAULT_TOP_K = 50
DEFAULT_CHUNKSIZE = 500_000

#: Labels of combined predictions (not base models). Mirrors ``pipeline_utils.is_wopt_label`` and
#: ``is_naive_ensemble_label`` without importing the engine.
_COMBINED_PREFIXES = ('ensemble', 'Linear transformation', 'Nelder Mead', 'Bayesian optimisation',
                      'Analytic least-squares', 'Joint Bayesian', 'Iterative ensemble',
                      'Super learner')


def is_combined_label(label) -> bool:
    """True for a combined prediction label (``ensemble``, ``Super learner__analytic`` ...)."""
    if not isinstance(label, str):
        return False
    return any(label == p or label.startswith(p + '__') or label.startswith(p + ' ')
               for p in _COMBINED_PREFIXES)


def select_models(present: Iterable, wanted: Optional[Sequence[str]] = None) -> List[str]:
    """The model labels used for the consensus.

    `wanted` (explicit) keeps the labels that are present, in the order given; otherwise every
    base (non-combined) label, and when there is none the naive-ensemble labels."""
    present = [str(m) for m in dict.fromkeys(present)]
    if wanted:
        return [m for m in wanted if m in present]
    base = [m for m in present if not is_combined_label(m)]
    if base:
        return base
    return [m for m in present if m == 'ensemble' or m.startswith('ensemble__')]


def _task_columns(frame: pd.DataFrame) -> List[str]:
    return [c for c in TASK_COLUMNS if c in frame.columns]


def _group_columns(task_cols: Sequence[str]) -> List[str]:
    return [c for c in task_cols if c != 'sample']


def _jaccard_stats(sets: List[frozenset], n_candidates: int = 0
                   ) -> Tuple[float, float, float, float]:
    """(mean Jaccard, SD of Jaccard, mean overlap count, mean Kuncheva index) over all pairs of
    sets. Kuncheva's consistency index corrects the overlap for the overlap expected by chance
    among `n_candidates` candidates: (inter * n - k^2) / (k * (n - k)), k = mean set size; it is 1
    for identical sets, about 0 for random ones, and blank (NaN) when k = 0 or k = n."""
    nan = float('nan')
    if len(sets) < 2:
        return nan, nan, nan, nan
    jac, ov, kun = [], [], []
    n = int(n_candidates)
    for a, b in itertools.combinations(sets, 2):
        inter = len(a & b)
        union = len(a) + len(b) - inter
        jac.append(inter / union if union else nan)
        ov.append(inter)
        k = (len(a) + len(b)) / 2.0
        kun.append((inter * n - k * k) / (k * (n - k)) if n > 0 and 0 < k < n else nan)
    jac_arr = np.asarray(jac, dtype=float)
    kun_arr = np.asarray(kun, dtype=float)
    return (float(np.nanmean(jac_arr)),
            float(np.nanstd(jac_arr, ddof=1)) if np.isfinite(jac_arr).sum() > 1 else nan,
            float(np.mean(ov)),
            float(np.nanmean(kun_arr)) if np.isfinite(kun_arr).any() else nan)


def _top_indices(score: np.ndarray, k: int) -> np.ndarray:
    """Indices of the K largest finite scores, ties broken by position (deterministic)."""
    valid = np.flatnonzero(np.isfinite(score))
    if valid.size == 0:
        return valid
    order = np.lexsort((valid, -score[valid]))      # primary: -score, secondary: position
    return valid[order[:k]]


def _summarise(records: List[dict], group_cols: Sequence[str], level: str, top_k: int,
               item_cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group tasks -> (stability rows, frequency rows).

    `records`: one dict per task: ``group`` (tuple), ``items`` (list of hashable items, each a
    tuple of `item_cols` values) and ``scores`` (matching list of floats), ``n_candidates``."""
    by_group: Dict[tuple, List[dict]] = {}
    for rec in records:
        by_group.setdefault(rec['group'], []).append(rec)
    stab_rows, freq_rows = [], []
    for group, recs in by_group.items():
        n_tasks = len(recs)
        sets = [frozenset(r['items']) for r in recs]
        n_cand = int(max(r['n_candidates'] for r in recs))
        jac_mean, jac_sd, overlap, kuncheva = _jaccard_stats(sets, n_cand)
        count: Dict[tuple, int] = {}
        score_sum: Dict[tuple, float] = {}
        for r in recs:
            for item, s in zip(r['items'], r['scores']):
                count[item] = count.get(item, 0) + 1
                score_sum[item] = score_sum.get(item, 0.0) + float(s)
        freq = {item: c / n_tasks for item, c in count.items()}
        base = dict(zip(group_cols, group))
        stab_rows.append({
            'level': level, **base, 'top_k': int(top_k), 'n_tasks': n_tasks,
            'n_candidates': n_cand,
            'jaccard_mean': jac_mean, 'jaccard_sd': jac_sd, 'overlap_mean': overlap,
            'kuncheva_mean': kuncheva,
            'n_selected_ever': len(count),
            'n_freq_ge_0.5': int(sum(f >= 0.5 for f in freq.values())),
            'n_freq_ge_0.9': int(sum(f >= 0.9 for f in freq.values())),
            'n_freq_1': int(sum(f >= 1.0 for f in freq.values())),
        })
        for item in sorted(count, key=lambda it: (-count[it], it)):
            freq_rows.append({**base, **dict(zip(item_cols, item)), 'n_tasks': n_tasks,
                              'n_selected': count[item], 'frequency': freq[item],
                              'consensus_mean': score_sum[item] / count[item]})
    return pd.DataFrame(stab_rows), pd.DataFrame(freq_rows)


# ----------------------------------------------------------------------------------- markers
def _bin_columns(values: np.ndarray, names: List[str], bin_size: int) -> Tuple[np.ndarray, List[str]]:
    """Pool `bin_size` consecutive columns (sum; NaN only if every member is NaN)."""
    if bin_size <= 1:
        return values, names
    n = values.shape[1]
    starts = np.arange(0, n, bin_size)
    filled = np.nan_to_num(values, nan=0.0)
    present = (~np.isnan(values)).astype(float)
    sums = np.add.reduceat(filled, starts, axis=1)
    counts = np.add.reduceat(present, starts, axis=1)
    out = np.where(counts > 0, sums, np.nan)
    labels = [f"{names[s]}:{names[min(s + bin_size, n) - 1]}" if min(s + bin_size, n) - 1 > s
              else names[s] for s in starts]
    return out, labels


def marker_stability(effect: pd.DataFrame, top_k: int = DEFAULT_TOP_K,
                     models: Optional[Sequence[str]] = None, bin_size: int = 1
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Stability of the top-K markers over the tasks of each group.

    Returns ``(stability, frequency, notes)``; both frames are empty (and a note says why) when
    there is nothing to evaluate. Raises ValueError for a table that is not a Marker_effect."""
    if int(top_k) < 1:
        raise ValueError("top_k must be at least 1")
    if int(bin_size) < 1:
        raise ValueError("bin_size must be at least 1")
    notes: List[str] = []
    if effect is None or effect.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame(), ["no Marker_effect rows"]
    if 'model' not in effect.columns or 'sample' not in effect.columns:
        raise ValueError("the effect table needs 'model' and 'sample' columns")
    marker_cols = [c for c in effect.columns if c not in ID_COLUMNS]
    if not marker_cols:
        raise ValueError("the effect table has no marker column")
    task_cols = _task_columns(effect)
    chosen = select_models(effect['model'].astype(str), models)
    if not chosen:
        return pd.DataFrame(), pd.DataFrame(), ["none of the requested models is in the table"]
    sub = effect[effect['model'].astype(str).isin(chosen)]
    vals = np.abs(sub[marker_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float))
    total = np.nansum(vals, axis=1)
    keep = np.isfinite(total) & (total > 0.0)
    if not keep.any():
        return pd.DataFrame(), pd.DataFrame(), ["every effect row is empty or all zero"]
    vals = vals[keep] / total[keep][:, None]
    sub = sub.loc[keep]
    vals, unit_names = _bin_columns(vals, [str(c) for c in marker_cols], int(bin_size))
    task_key = sub[task_cols].astype(str).agg('|'.join, axis=1).to_numpy()
    codes, uniques = pd.factorize(task_key)
    # mean over the chosen models of each task, ignoring a model that did not report the marker
    sums = np.zeros((len(uniques), vals.shape[1]))
    cnts = np.zeros_like(sums)
    np.add.at(sums, codes, np.nan_to_num(vals, nan=0.0))
    np.add.at(cnts, codes, (~np.isnan(vals)).astype(float))
    with np.errstate(invalid='ignore', divide='ignore'):
        score = np.where(cnts > 0, sums / cnts, np.nan)
    first_row = pd.Series(np.arange(len(codes))).groupby(codes).first().to_numpy()
    task_frame = sub[task_cols].iloc[first_row].reset_index(drop=True)
    group_cols = _group_columns(task_cols)
    records = []
    for t in range(len(uniques)):
        idx = _top_indices(score[t], int(top_k))
        row = task_frame.iloc[t]
        records.append({'group': tuple(str(row[c]) for c in group_cols),
                        'items': [(unit_names[i],) for i in idx],
                        'scores': [score[t, i] for i in idx],
                        'n_candidates': int(np.isfinite(score[t]).sum())})
    if any(len(r['items']) < int(top_k) for r in records):
        notes.append(f"some tasks have fewer than {int(top_k)} candidate markers; their sets are "
                     f"shorter")
    stab, freq = _summarise(records, group_cols, 'marker', int(top_k), ['marker'])
    if not stab.empty and (stab['n_tasks'] < 2).any():
        notes.append("a group has a single iteration: its Jaccard stability is blank (NaN)")
    return stab, freq, notes


# -------------------------------------------------------------------------------------- pairs
def _interaction_chunks(source, chunksize: int, usecols: List[str]) -> Iterable[pd.DataFrame]:
    """Chunks of an Interaction table: a path (read in chunks) or a DataFrame / iterable of
    DataFrames (used as is, which is how the tests exercise different chunk sizes)."""
    if isinstance(source, (str, os.PathLike)):
        yield from pd.read_csv(source, usecols=lambda c: c in usecols, chunksize=int(chunksize),
                               dtype={c: str for c in usecols if c != 'value'})
    elif isinstance(source, pd.DataFrame):
        for start in range(0, source.shape[0], int(chunksize)):
            yield source.iloc[start:start + int(chunksize)]
    else:
        yield from source


def pair_stability(source, top_k: int = DEFAULT_TOP_K, models: Optional[Sequence[str]] = None,
                   chunksize: int = DEFAULT_CHUNKSIZE
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Stability of the top-K marker pairs; `source` is a path to ``Interaction.csv[.gz]`` or a
    DataFrame. Same return shape as `marker_stability`."""
    if int(top_k) < 1:
        raise ValueError("top_k must be at least 1")
    need = ['population', 'phenotype', 'model', 'marker1', 'marker2', 'value']
    usecols = need + ['asreml_option', 'ratio', 'sample']

    def chunks():
        return _interaction_chunks(source, chunksize, usecols)

    # pass 1: L1 totals per (task, model) and the models present
    total_parts: List[pd.Series] = []
    task_cols: Optional[List[str]] = None
    for chunk in chunks():
        missing = [c for c in need if c not in chunk.columns]
        if missing:
            raise ValueError(f"the interaction table lacks column(s) {', '.join(missing)}")
        if 'sample' not in chunk.columns:
            return pd.DataFrame(), pd.DataFrame(), ["the interaction table has no 'sample' column"]
        if task_cols is None:
            task_cols = _task_columns(chunk)
        val = pd.to_numeric(chunk['value'], errors='coerce').abs().fillna(0.0)
        key = chunk[task_cols + ['model']].astype(str)
        total_parts.append(val.groupby([key[c].to_numpy() for c in task_cols + ['model']]).sum())
    if task_cols is None or not total_parts:
        return pd.DataFrame(), pd.DataFrame(), ["no interaction rows"]
    totals = pd.concat(total_parts).groupby(level=list(range(len(task_cols) + 1))).sum()
    totals.index.names = task_cols + ['model']
    totals = totals.rename('tot').reset_index()
    chosen = select_models(totals['model'], models)
    if not chosen:
        return pd.DataFrame(), pd.DataFrame(), ["none of the requested models has interactions"]
    chosen_set = set(chosen)
    # pass 2: per (task, pair) sum and count of normalised values over the chosen models
    partial: List[pd.DataFrame] = []
    merged: Optional[pd.DataFrame] = None
    keycols = task_cols + ['m1', 'm2']

    def merge(frames: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames).groupby(level=list(range(len(keycols)))).sum()

    for chunk in chunks():
        c = chunk[chunk['model'].astype(str).isin(chosen_set)]
        if c.empty:
            continue
        keys = c[task_cols + ['model']].astype(str).reset_index(drop=True)
        tot = keys.merge(totals, on=task_cols + ['model'], how='left')['tot'].to_numpy(dtype=float)
        val = pd.to_numeric(c['value'], errors='coerce').abs().to_numpy(dtype=float)
        ok = np.isfinite(val) & np.isfinite(tot) & (tot > 0)
        if not ok.any():
            continue
        a, b = c['marker1'].astype(str).to_numpy(), c['marker2'].astype(str).to_numpy()
        lo, hi = np.where(a <= b, a, b), np.where(a <= b, b, a)
        frame = keys[task_cols].copy()
        frame['m1'], frame['m2'] = lo, hi
        frame['s'] = val / np.where(tot > 0, tot, 1.0)
        frame['n'] = 1.0
        frame = frame[ok]
        partial.append(frame.groupby(keycols)[['s', 'n']].sum())
        if len(partial) >= 16:
            merged = merge(([merged] if merged is not None else []) + partial)
            partial = []
    if partial:
        merged = merge(([merged] if merged is not None else []) + partial)
    if merged is None or merged.empty:
        return pd.DataFrame(), pd.DataFrame(), ["no interaction row of the chosen models"]
    merged = merged.reset_index()
    merged['score'] = merged['s'] / merged['n']
    group_cols = _group_columns(task_cols)
    records = []
    for key, part in merged.groupby(task_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        n_candidates = int(part.shape[0])
        part = part.sort_values(['score', 'm1', 'm2'], ascending=[False, True, True]).head(int(top_k))
        row = dict(zip(task_cols, key))
        records.append({'group': tuple(str(row[c]) for c in group_cols),
                        'items': list(zip(part['m1'], part['m2'])),
                        'scores': part['score'].tolist(),
                        'n_candidates': n_candidates})
    stab, freq = _summarise(records, group_cols, 'pair', int(top_k), ['marker1', 'marker2'])
    notes: List[str] = []
    if not stab.empty and (stab['n_tasks'] < 2).any():
        notes.append("a group has a single iteration: its Jaccard stability is blank (NaN)")
    return stab, freq, notes


# -------------------------------------------------------------------------------------- driver
def build_report(effect: Optional[pd.DataFrame], interactions=None, top_k: int = DEFAULT_TOP_K,
                 models: Optional[Sequence[str]] = None, bin_size: int = 1, pairs: bool = True,
                 chunksize: int = DEFAULT_CHUNKSIZE) -> dict:
    """The three tables as DataFrames: ``{'consensus', 'marker_frequency', 'pair_frequency',
    'notes'}``. Pairs are computed only when `pairs` and `interactions` is given."""
    notes: List[str] = []
    stab_parts, marker_freq, pair_freq = [], pd.DataFrame(), pd.DataFrame()
    if effect is not None:
        s, marker_freq, n = marker_stability(effect, top_k, models, bin_size)
        notes += [f"markers: {x}" for x in n]
        if not s.empty:
            stab_parts.append(s)
    if pairs and interactions is not None:
        s, pair_freq, n = pair_stability(interactions, top_k, models, chunksize)
        notes += [f"pairs: {x}" for x in n]
        if not s.empty:
            stab_parts.append(s)
    consensus = pd.concat(stab_parts, ignore_index=True) if stab_parts else pd.DataFrame()
    return {'consensus': consensus, 'marker_frequency': marker_freq,
            'pair_frequency': pair_freq, 'notes': notes}


def write_report(report: dict, out_dir: str, frequencies: bool = True) -> List[str]:
    """Write the tables of `build_report` into `out_dir`; returns the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    plan = [(CONSENSUS_FILE, report['consensus'])]
    if frequencies:
        plan += [(MARKER_FREQUENCY_FILE, report['marker_frequency']),
                 (PAIR_FREQUENCY_FILE, report['pair_frequency'])]
    for name, frame in plan:
        if frame is not None and not frame.empty:
            path = os.path.join(out_dir, name)
            frame.to_csv(path, index=False)
            written.append(path)
    return written


def write_interaction_frequency(interactions, out_path: str, report: dict,
                                chunksize: int = DEFAULT_CHUNKSIZE) -> int:
    """X03: copy the rows of ``Interaction.csv`` (a path) to `out_path` with the columns
    ``pair_frequency``, ``marker1_frequency`` and ``marker2_frequency`` appended, from the
    frequency tables of `report` (`build_report`). Returns the number of rows written.

    The frequency belongs to the pair within a GROUP (population, phenotype, ratio), not to a
    model, so every model's row of the same pair carries the same value. Cells are copied as text
    (nothing is re-formatted), so the original columns are byte-for-byte the same values; the file
    is written in chunks and is always plain CSV, even when the source is ``.csv.gz``."""
    if not isinstance(interactions, (str, os.PathLike)):
        raise ValueError("interactions must be the path of an Interaction.csv file")
    pair_freq, marker_freq = report['pair_frequency'], report['marker_frequency']
    written = 0
    first = True
    tmp = out_path + '.partial'
    try:
        for chunk in pd.read_csv(interactions, chunksize=int(chunksize), dtype=str,
                                 keep_default_na=False):
            groups = [c for c in GROUP_COLUMNS if c in chunk.columns]
            for table in (pair_freq, marker_freq):
                if table is not None and not table.empty and \
                        any(c not in table.columns for c in groups):
                    raise ValueError("the frequency tables and Interaction.csv do not share the "
                                     "same group columns")
            a, b = chunk['marker1'].to_numpy(), chunk['marker2'].to_numpy()
            frame = chunk[groups].copy()
            frame['m1'], frame['m2'] = np.where(a <= b, a, b), np.where(a <= b, b, a)
            frame['marker1'], frame['marker2'] = a, b
            out = chunk.copy()
            if pair_freq is not None and not pair_freq.empty:
                lookup = pair_freq[groups + ['marker1', 'marker2', 'frequency']].rename(
                    columns={'marker1': 'm1', 'marker2': 'm2', 'frequency': 'pair_frequency'})
                out['pair_frequency'] = frame.merge(lookup, on=groups + ['m1', 'm2'],
                                                    how='left')['pair_frequency'].to_numpy()
            else:
                out['pair_frequency'] = np.nan
            for col in ('marker1', 'marker2'):
                if marker_freq is not None and not marker_freq.empty:
                    lookup = marker_freq[groups + ['marker', 'frequency']].rename(
                        columns={'marker': col, 'frequency': f'{col}_frequency'})
                    out[f'{col}_frequency'] = frame[groups + [col]].merge(
                        lookup, on=groups + [col], how='left')[f'{col}_frequency'].to_numpy()
                else:
                    out[f'{col}_frequency'] = np.nan
            for col in ('pair_frequency', 'marker1_frequency', 'marker2_frequency'):
                out[col] = out[col].astype(float).fillna(0.0)
            out.to_csv(tmp, mode='w' if first else 'a', header=first, index=False)
            first = False
            written += out.shape[0]
        if written == 0:                # empty source: nothing to write
            return 0
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return written


def run_stability_report(cfg: dict, result_name: str, effect_fn: Callable[[], pd.DataFrame],
                         log: Callable[[str], None]) -> List[str]:
    """The report-site hook (ver4-99 call sites): does nothing unless
    ``cfg['STABILITY_SUMMARY_CREATE']`` is true, writes the three tables into the result folder
    (and, with ``cfg['STABILITY_INTERACTION_FREQUENCY']`` as well, the X03 file) and NEVER raises (a failure is one log line; the run's own files are never touched)."""
    if not bool(cfg.get('STABILITY_SUMMARY_CREATE', False)):
        return []
    try:
        from pipeline_utils import result_dir_path
        from checkpoint_utils import resolve_result_path
        top_k = max(1, int(cfg.get('STABILITY_TOP_K', DEFAULT_TOP_K) or DEFAULT_TOP_K))
        effect = effect_fn()
        interactions = resolve_result_path(result_name, 0, False, 'interactions')
        if interactions is not None and os.path.getsize(interactions) == 0:
            interactions = None
        report = build_report(effect, interactions, top_k=top_k, pairs=interactions is not None)
        out_dir = result_dir_path(result_name)
        written = write_report(report, out_dir)
        if bool(cfg.get('STABILITY_INTERACTION_FREQUENCY', False)):          # X03, off by default
            if interactions is None:
                log("[stability] NOTE: Interaction frequency requested but the run has no "
                    "Interaction table.")
            else:
                path = os.path.join(out_dir, INTERACTION_FREQUENCY_FILE)
                n_rows = write_interaction_frequency(interactions, path, report)
                if n_rows:
                    written.append(path)
        for note in report['notes']:
            log(f"[stability] NOTE: {note}")
        if written:
            log(f"[stability] wrote {', '.join(os.path.basename(p) for p in written)} "
                f"(top {top_k}).")
        else:
            log("[stability] NOTE: nothing to evaluate (no Marker_effect rows).")
        return written
    except Exception as exc:  # noqa: BLE001 - reporting must never stop a finished run
        log(f"[stability] NOTE: the stability summary was skipped: {exc!r}")
        return []
