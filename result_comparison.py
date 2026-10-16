"""
result_comparison.py
====================
Update ID ver4-54 (R2 / R3 / R4 / R5 / R6; Design Blueprint
`EasiGP_ver4-54_Design_Blueprint.md`) - the "Tool" mode's cross-run performance
comparison.

Layer 3c: a pure module (no Streamlit import, I2) that sits BESIDE
`metric_summary.py` / `metric_plot.py` and imports only downward
(`batch_reader`, `checkpoint_utils`, `metric_summary`, `metric_plot`,
`plot_style`, `pipeline_utils`, plus the Tool-mode key helpers of
`gui_config_loader`). It never runs `GP()`, never touches R, and never writes
into a run's own result folder (I8): every output goes to
`result_dir_path(cfg['RESULT_NAME'])`, the Tool's own folder.

What it does
------------
Given two or more finished runs ("sources"), for each of three sections
(`SECTIONS`: Pearson correlation + MSE, Adjusted prediction accuracy, Spearman
correlation) it

  * reads every source's `Metric` table - from the combined file when the run
    was assembled, or from its complete per-batch `Metric_<idx>.csv` files when
    it was not (R4) - through `batch_reader.ResultSet`, splitting a multi-option
    ASReml run into one source per option;
  * draws the existing violin figure with ONE violin per (model, result) pair:
    colour = result, a two-tier x axis (result name under each violin, model
    name under each group of violins). The drawing itself is
    `metric_plot._render_grid` / `_draw_metric_cell`, reused unchanged through
    two optional hooks, so every violin style the Tab-5 figure offers works here
    too (R2, R3);
  * writes a summary workbook laid out like the hand-built `example.xlsx` the
    blueprint (E9) is based on: one block per metric, phenotypes as rows, a
    two-row header (model name merged over its result columns, result names
    below), the best value of each phenotype row marked `*`, missing cells `-`.

Why a separate module (blueprint R2 "Rejected"): `metric_summary.
write_pivot_worksheet` has no merged two-row header, no `-` cell and no best
marker, and extending it would change `Metric_summary.xlsx` (I11). Only the
statistic authority (`metric_summary.six_statistics`) and the per-level long
sheets (`metric_summary.build_metric_summary`) are imported from there (I3).

A fourth, plot-only section - "Comparison of metrics between results"
(`SCATTER_SECTION`, ver4-60) - is the cross-run twin of Tab 5's "Comparison of
metrics between phenotype representation": for every metric and every pair of the
selected results it matches the same task (population, phenotype, model, ratio,
sample) in both and draws one centroid +/- 1 SE point per model against a 1:1 line
(`accuracy_metrics_plot.metric_across_results_plot`). It has no violin or workbook
settings, so it is kept OUT of `SECTIONS` / `TOOL_SECTIONS` (every loop over those
two is violin/workbook-shaped); a Tool config that never mentions it is unchanged
(I11).

Public entry points: `discover_result_sources`, `source_status`,
`load_source_metric`, `build_comparison_record`, `comparison_violin_plot`,
`build_comparison_pivot`, `write_comparison_summary`, `validate_tool_config`,
`run_tool`.
"""

from __future__ import annotations

import csv
import glob
import os
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import batch_reader as _br
import checkpoint_utils as _ckpt
import metric_plot as _mp
import metric_summary as _ms
import pipeline_utils as _pu
import plot_style as _ps
import tool_diagnostics as _td
from gui_config_loader import (
    SOURCE_SEP, TOOL_SECTIONS, TOOL_SCATTER_SECTION, make_source_id, default_source_label,
)
from pipeline_utils import canonical_model_order, is_naive_ensemble_label, result_dir_path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Joins model and result into the composite x key the shared violin drawing
#: code works with (`record['x'] = record['model'] = key`). Seaborn briefly
#: renders these keys as tick text, so the separator is a PRINTABLE character
#: every matplotlib font has (a control character raised a "missing glyph"
#: warning); `validate_tool_config` rejects a result name containing it.
KEY_SEP = '\u2016'

GATHER_MODES = ('auto', 'combined', 'batches')
SUMMARY_STATISTICS = ('mean', 'median')
X_LABEL_MODES = ('two_tier', 'combined', 'model_only')
REFERENCE_MODES = ('none', 'baseline')

OPTION_COLUMN = _br.OPTION_COLUMN  # 'asreml_option'
_STATS_ORDER = _ms.PIVOT_STATISTICS  # the six Metric_summary statistics

#: Direction of "better" per metric - drives the best-value marker.
_BETTER = {
    'Pearson correlation': 'higher',
    'MSE': 'lower',
    'Adjusted prediction accuracy': 'higher',
    'Spearman correlation': 'higher',
}

#: The three sections. `metrics` is the list of Metric columns the section
#: compares; `tag` names the workbook; `create_plot` is the legacy-preserving
#: default of the "violin plots" tick box (mirrors Tab 5: on for Pearson/MSE,
#: off for the two accuracy metrics).
SECTIONS = {
    'pearson_mse': {
        'title': 'Pearson correlation & MSE',
        'metrics': ('Pearson correlation', 'MSE'),
        'tag': 'Pearson_MSE',
        'create_plot': True,
    },
    'adjusted_accuracy': {
        'title': 'Adjusted prediction accuracy (adjusted Pearson correlation)',
        'metrics': ('Adjusted prediction accuracy',),
        'tag': 'Adjusted_accuracy',
        'create_plot': False,
    },
    'spearman': {
        'title': 'Spearman correlation',
        'metrics': ('Spearman correlation',),
        'tag': 'Spearman',
        'create_plot': False,
    },
}
for _s in SECTIONS.values():
    _s['better'] = {m: _BETTER[m] for m in _s['metrics']}
assert tuple(SECTIONS) == TOOL_SECTIONS, "SECTIONS and gui_config_loader.TOOL_SECTIONS must agree"

#: ver4-60 - the plot-only "Comparison of metrics between results" section. Its
#: metrics are the three Tab 5's across-options scatter offers (MSE is not on a
#: comparable scale between runs, exactly as there); `tag` prefixes nothing on
#: disk (the figure prefix is `SCATTER_PREFIX`), it only names the section in logs.
SCATTER_SECTION = TOOL_SCATTER_SECTION
SCATTER_METRICS = (
    'Pearson correlation',
    'Adjusted prediction accuracy',
    'Spearman correlation',
)
SCATTER_SPEC = {
    'title': 'Comparison of metrics between results',
    'metrics': SCATTER_METRICS,
    'tag': 'Across_results',
    'create_plot': True,
}
SCATTER_PREFIX = 'Across_results_'
#: Every section a Tool config may carry, in display order.
ALL_SECTIONS = tuple(SECTIONS) + (SCATTER_SECTION,)
DEFAULT_SCATTER_PLOT_CONFIG = {'font_size': 1, 'fig_size': 5}

#: Violin settings a section uses when its `plot_config` omits a key. They equal
#: the widget defaults of the Tool tab (and the Tab-5 violin defaults).
DEFAULT_PLOT_CONFIG = {
    'font_size': 1,
    'fig_size': 5,
    'style': 'auto',
    'min_n_violin': 5,
    'legend': 'auto',
    'between_label': 'test',
    'cut': 2,
    'show_mean_se': False,
    'show_median': False,
    'auto_width': True,
    'group_separators': True,
    'reference': 'none',
    'x_label_mode': 'two_tier',
}
DEFAULT_DPI = 300
DEFAULT_DECIMALS = 3

_PHENOTYPE_SHEETS = _ms.SHEET_NAMES
_DEFAULT_LOG: Callable[[str], None] = print


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #


@dataclass
class ResultSource:
    """One finished run (or one ASReml option of it) that can be compared.

    `folder` is the sub-folder of `Result/`; `option` is `None` for a single-
    option / non-ASReml run, else one `asreml_option` label; `label` is the
    editable display name (defaults to `default_source_label`)."""

    folder: str
    option: Optional[str] = None
    label: str = ''

    def __post_init__(self) -> None:
        if not self.label:
            self.label = default_source_label(self.folder, self.option)

    @property
    def source_id(self) -> str:
        return make_source_id(self.folder, self.option)


def _clean_label(value) -> str:
    """`1.0` -> `'1'`, everything else `str(value)` (mirrors
    `circos_plot._clean_population_label`, which is private to a heavy module
    and therefore not imported)."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def _clean_label_series(series: pd.Series) -> pd.Series:
    mapping = {v: _clean_label(v) for v in pd.unique(series)}
    return series.map(mapping)


def _result_dir(folder: str) -> str:
    return os.path.join(_pu.RESULT_BASE_DIR, folder)


def _is_hidden_or_private(name: str) -> bool:
    return name.startswith('.') or name.startswith('_')


def _read_option_labels(path: str) -> list:
    """Distinct `asreml_option` labels of one Metric file ([] if it has none)."""
    try:
        header = pd.read_csv(path, nrows=0).columns
        if OPTION_COLUMN not in header:
            return []
        col = pd.read_csv(path, usecols=[OPTION_COLUMN])[OPTION_COLUMN]
    except (pd.errors.EmptyDataError, OSError, ValueError, EOFError):
        return []
    return [str(v) for v in pd.unique(col.dropna())]


def _read_columns(path: str) -> list:
    try:
        return [str(c) for c in pd.read_csv(path, nrows=0).columns]
    except (pd.errors.EmptyDataError, OSError, ValueError, EOFError):
        return []


# --------------------------------------------------------------------------- #
# R4 - discovery, status and loading
# --------------------------------------------------------------------------- #


def _metric_files_in(directory: str) -> tuple:
    """`(combined_path_or_None, [batch Metric paths])` found in `directory`."""
    combined = None
    for name in ('Metric.csv.gz', 'Metric.csv'):
        p = os.path.join(directory, name)
        if os.path.isfile(p):
            combined = p
            break
    batches = sorted(glob.glob(os.path.join(directory, 'Metric_*.csv')))
    return combined, batches


def discover_result_sources(base_dir: Optional[str] = None) -> "list[ResultSource]":
    """Every comparable result under `base_dir` (default `RESULT_BASE_DIR`):
    the immediate sub-folders that hold a combined `Metric` file or at least one
    `Metric_<idx>.csv`. Folders whose name starts with `.` or `_`, or contains
    `SOURCE_SEP`, are skipped (the last with a NOTE - such a name would make a
    source id ambiguous). A multi-option ASReml folder yields ONE source per
    option, in the order the options appear in the file.

    Only file headers and the option column are read - cheap enough to call on
    every GUI rerun (the GUI additionally caches it)."""
    base = base_dir or _pu.RESULT_BASE_DIR
    if not os.path.isdir(base):
        return []
    found: "list[ResultSource]" = []
    for name in sorted(os.listdir(base)):
        directory = os.path.join(base, name)
        if not os.path.isdir(directory) or _is_hidden_or_private(name):
            continue
        if SOURCE_SEP in name:
            print(
                f"[tool] NOTE: result folder {name!r} is skipped - its name contains "
                f"{SOURCE_SEP!r}, which Tool mode uses to join a folder and an ASReml option. "
                f"Rename the folder to compare it."
            )
            continue
        combined, batches = _metric_files_in(directory)
        probe = combined or (batches[0] if batches else None)
        if probe is None:
            continue
        options = _read_option_labels(probe)
        if options:
            found.extend(ResultSource(name, opt) for opt in options)
        else:
            found.append(ResultSource(name, None))
    return found


def source_status(folder: str, gather_mode: str = 'auto') -> dict:
    """Read-only description of how `folder` would be read under `gather_mode`
    (`auto` / `combined` / `batches`, the three choices of Step 2):

    `read_from` ('combined' | 'batches' | None), `batch_ids` (the complete
    batches to read), `incomplete` / `missing` (batch ids), `stale` (a batch file
    is newer than the combined file), `has_combined`, `options` (ASReml option
    labels, [] for a single-option run), `columns` (Metric header) and `error`
    (None when the folder can be read, else one sentence naming the reason).

    Incomplete batches (a checkpoint remains) are never read - the same rule
    Step 2 applies."""
    out = {
        'folder': folder,
        'read_from': None,
        'batch_ids': [],
        'incomplete': [],
        'missing': [],
        'stale': False,
        'has_combined': False,
        'options': [],
        'columns': [],
        'error': None,
    }
    if gather_mode not in GATHER_MODES:
        out['error'] = f"unknown gather mode {gather_mode!r} (use one of {', '.join(GATHER_MODES)})"
        return out
    if not folder or not os.path.isdir(_result_dir(folder)):
        out['error'] = f"result folder {folder!r} does not exist under {_pu.RESULT_BASE_DIR}"
        return out
    combined = _ckpt.resolve_result_path(folder, 0, False, 'record')
    out['has_combined'] = combined is not None
    try:
        batch_status = _ckpt.check_batch_status(folder)
    except (OSError, ValueError) as exc:
        batch_status = {'complete': [], 'incomplete': [], 'missing': []}
        print(
            f"[tool] NOTE: could not check the batch status of {folder!r} ({exc!r}); "
            f"treating it as having no batch files."
        )
    complete = list(batch_status['complete'])
    out['incomplete'] = [b['batch_id'] for b in batch_status['incomplete']]
    out['missing'] = list(batch_status['missing'])

    if gather_mode == 'combined':
        use = 'combined' if combined else None
    elif gather_mode == 'batches':
        use = 'batches' if complete else None
    else:
        use = 'combined' if combined else ('batches' if complete else None)
    if use is None:
        need = {
            'combined': 'a combined Metric file',
            'batches': 'at least one complete batch',
            'auto': 'a combined Metric file or at least one complete batch',
        }[gather_mode]
        out['error'] = (
            f"{folder!r}: gather mode {gather_mode!r} needs {need}, but none was found"
            + (
                f" ({len(out['incomplete'])} incomplete batch(es) are not read)"
                if out['incomplete']
                else ''
            )
        )
        return out
    out['read_from'] = use
    if use == 'batches':
        out['batch_ids'] = complete
    if combined and complete:
        try:
            combined_mtime = os.path.getmtime(combined)
            for b in complete:
                bp = _ckpt.resolve_result_path(folder, b, True, 'record')
                if bp and os.path.getmtime(bp) > combined_mtime:
                    out['stale'] = True
                    break
        except OSError:
            pass
    probe = (
        combined
        if use == 'combined'
        else _ckpt.resolve_result_path(folder, complete[0], True, 'record')
    )
    if probe is None:
        out['error'] = f"{folder!r}: the Metric file disappeared while it was being checked"
        return out
    out['columns'] = _read_columns(probe)
    if use == 'combined':
        out['options'] = _read_option_labels(probe)
    else:
        seen: list = []
        for b in complete:
            bp = _ckpt.resolve_result_path(folder, b, True, 'record')
            for opt in (_read_option_labels(bp) if bp else []):
                if opt not in seen:
                    seen.append(opt)
        out['options'] = seen
    return out


def status_text(status: dict) -> str:
    """One short phrase for a source selector, e.g. `assembled`, `12 complete
    batches - not assembled (+1 incomplete excluded)`."""
    if status.get('error'):
        return 'not readable'
    if status['read_from'] == 'combined':
        text = 'assembled'
        if status.get('stale'):
            text += ' (combined file older than batch files)'
    else:
        n = len(status['batch_ids'])
        text = f"{n} complete batch{'es' if n != 1 else ''} - not assembled"
    if status.get('incomplete'):
        text += f" (+{len(status['incomplete'])} incomplete excluded)"
    return text


def load_source_metric(src: ResultSource, gather_mode: str = 'auto') -> "tuple[pd.DataFrame, dict]":
    """The `Metric` table of one source plus how it was read.

    Combined file when present (`auto`/`combined`), else the complete batches
    (`auto`/`batches`), through `batch_reader.ResultSet` - nothing is written
    (I8). For a multi-option run the frame is the per-option view (rows
    filtered, `asreml_option` dropped), exactly what a single-option run holds.

    Returns `(frame, info)`; `info` has `read_from`, `batch_ids`, `incomplete`,
    `missing`, `stale`, `n_rows`, `metric_weighting`, `mtime`.

    Raises ValueError (message names the folder) when the source cannot be read
    or its option is not in the run."""
    status = source_status(src.folder, gather_mode)
    if status['error']:
        raise ValueError(status['error'])
    options = status['options']
    if src.option is None and options:
        raise ValueError(
            f"{src.folder!r} holds several ASReml options ({', '.join(options)}); "
            f"select one option per source."
        )
    if src.option is not None and src.option not in options:
        raise ValueError(
            f"{src.folder!r} has no ASReml option {src.option!r} "
            f"(available: {', '.join(options) if options else 'none - single-option run'})."
        )
    try:
        if status['read_from'] == 'combined':
            rs = _br.ResultSet(src.folder, 'combined', option=src.option)
        else:
            rs = _br.ResultSet(
                src.folder,
                'batches',
                batch_ids=status['batch_ids'],
                missing_batches=status['missing'],
                incomplete_batches=status['incomplete'],
                option=src.option,
            )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"{src.folder!r}: could not read the Metric table ({exc})") from exc
    frame = rs.metric.copy()
    weighting = 'n/a'
    if 'metric_weighting' in frame.columns:
        values = sorted({str(v) for v in frame['metric_weighting'].dropna().unique()})
        weighting = ', '.join(values) if values else 'n/a'
    probe = (
        _ckpt.resolve_result_path(src.folder, 0, False, 'record')
        if status['read_from'] == 'combined'
        else _ckpt.resolve_result_path(src.folder, status['batch_ids'][0], True, 'record')
    )
    try:
        mtime = pd.Timestamp(os.path.getmtime(probe), unit='s').strftime('%Y-%m-%d %H:%M:%S')
    except (OSError, TypeError):
        mtime = ''
    info = {
        'read_from': status['read_from'],
        'batch_ids': list(status['batch_ids']),
        'incomplete': list(status['incomplete']),
        'missing': list(status['missing']),
        'stale': status['stale'],
        'n_rows': int(frame.shape[0]),
        'metric_weighting': weighting,
        'mtime': mtime,
    }
    return frame, info


def list_source_models(src: ResultSource, gather_mode: str = 'auto') -> "list[str]":
    """Model names present in one source, in `canonical_model_order`.

    Reads only the `model` (and `asreml_option`) columns of the same files
    `load_source_metric` would read, so the Tool tab can offer a model filter
    without loading whole Metric tables. Returns `[]` when the source cannot be
    read (the caller shows the status text instead)."""
    status = source_status(src.folder, gather_mode)
    if status['error']:
        return []
    if status['read_from'] == 'combined':
        paths = [_ckpt.resolve_result_path(src.folder, 0, False, 'record')]
    else:
        paths = [
            _ckpt.resolve_result_path(src.folder, b, True, 'record') for b in status['batch_ids']
        ]
    seen: list = []
    for path in paths:
        if not path:
            continue
        columns = [c for c in ('model', OPTION_COLUMN) if c in _read_columns(path)]
        if 'model' not in columns:
            continue
        try:
            frame = pd.read_csv(path, usecols=columns)
        except (pd.errors.EmptyDataError, OSError, ValueError, EOFError):
            continue
        if src.option is not None and OPTION_COLUMN in frame.columns:
            frame = frame[frame[OPTION_COLUMN].astype(str) == src.option]
        for name in pd.unique(frame['model'].dropna()):
            if str(name) not in seen:
                seen.append(str(name))
    return _pu.canonical_model_order(seen)


def _sources_of(section_cfg: dict) -> "list[ResultSource]":
    out = []
    for item in section_cfg.get('sources') or []:
        out.append(
            ResultSource(str(item['folder']), item.get('option'), str(item.get('label') or ''))
        )
    return out


def build_comparison_record(
    section_cfg: dict,
    gather_mode: str = 'auto',
    log: Callable[[str], None] = _DEFAULT_LOG,
    *,
    section: Optional[str] = None,
) -> "tuple[pd.DataFrame, list, list, pd.DataFrame]":
    """Read every source of one section and stack them into one table.

    `section_cfg` is one `TOOL_COMPARISON[<section>]` dict (`sources` =
    `[{folder, option, label}, ...]`, `models` = list or None). With `section`
    given, a source lacking (or entirely NaN in) any of that section's metrics
    is DROPPED with a NOTE (an older run, or the metric was not selected there).

    Returns `(record, models, results, provenance)`:
      * `record` - the concatenated Metric rows plus a `result` column (the
        label); `population` and `phenotype` are normalised to `str`;
      * `models` - the models in `canonical_model_order` of their union over the
        sources, restricted to `section_cfg['models']` when that is not None;
      * `results` - the kept labels, in selection order;
      * `provenance` - one row per kept source (the `sources` sheet).

    Never raises for a droppable source; raises ValueError for an unreadable
    one (the message names the folder)."""
    sources = _sources_of(section_cfg)
    required = SECTIONS[section]['metrics'] if section else ()
    frames, provenance, weightings, ratio_sets, scenarios = [], [], {}, {}, {}
    for src in sources:
        frame, info = load_source_metric(src, gather_mode)
        log(
            f"[tool] source {src.label!r}: read {info['n_rows']} Metric row(s) from "
            f"{src.folder!r}{' / ' + src.option if src.option else ''} "
            f"({info['read_from']}"
            + (f", batches {info['batch_ids']}" if info['read_from'] == 'batches' else '')
            + ")."
        )
        if info['stale']:
            log(
                f"[tool] NOTE: {src.folder!r}: a batch Metric file is newer than the combined "
                f"Metric file - the combined (older) file is used. Re-assemble the run, or choose "
                f"'batches' as the gather mode, if the batches hold newer results."
            )
        if info['incomplete']:
            log(
                f"[tool] NOTE: {src.folder!r}: incomplete batches {info['incomplete']} are excluded."
            )
        if info['missing']:
            log(f"[tool] NOTE: {src.folder!r}: batches {info['missing']} have no output at all.")
        if frame.shape[0] == 0 or 'model' not in frame.columns:
            log(f"[tool] NOTE: source {src.label!r} ({src.folder!r}) has no Metric rows - dropped.")
            continue
        lacking = [
            m
            for m in required
            if m not in frame.columns or pd.to_numeric(frame[m], errors='coerce').notna().sum() == 0
        ]
        if lacking:
            log(
                f"[tool] NOTE: source {src.label!r} ({src.folder!r}) lacks "
                + ', '.join(repr(m) for m in lacking)
                + " (an older run, or the metric was not selected there) - dropped from this section."
            )
            continue
        frame = frame.copy()
        frame['result'] = src.label
        frames.append(frame)
        weightings[src.label] = info['metric_weighting']
        ratio_sets[src.label] = (
            sorted({str(v) for v in frame['ratio'].unique()}) if 'ratio' in frame else []
        )
        pops = (
            frame['population'].astype(str) if 'population' in frame else pd.Series([], dtype=str)
        )
        scenarios[src.label] = (
            'between' if (len(pops) and pops.str.contains('->').any()) else 'within'
        )
        provenance.append(
            {
                'label': src.label,
                'folder': src.folder,
                'option': src.option or '',
                'read_from': info['read_from'],
                'batch_ids': ','.join(str(b) for b in info['batch_ids']),
                'incomplete_excluded': ','.join(str(b) for b in info['incomplete']),
                'n_rows': info['n_rows'],
                'metric_weighting': info['metric_weighting'],
                'Metric mtime': info['mtime'],
            }
        )
    prov_cols = [
        'label',
        'folder',
        'option',
        'read_from',
        'batch_ids',
        'incomplete_excluded',
        'n_rows',
        'metric_weighting',
        'Metric mtime',
    ]
    provenance_df = pd.DataFrame(provenance, columns=prov_cols)
    if not frames:
        return pd.DataFrame(), [], [], provenance_df

    if len(set(weightings.values())) > 1:
        log(
            "[tool] NOTE: the sources use different metric weightings "
            f"({'; '.join(f'{k}: {v}' for k, v in weightings.items())}) - the metrics are defined "
            f"differently, so the comparison is not like-for-like."
        )
    if len({tuple(v) for v in ratio_sets.values()}) > 1:
        log(
            "[tool] NOTE: the sources were run with different train/test ratios "
            f"({'; '.join(f'{k}: {v}' for k, v in ratio_sets.items())})."
        )
    if len(set(scenarios.values())) > 1:
        log(
            "[tool] NOTE: the sources mix 'within' and 'between' population scenarios "
            f"({'; '.join(f'{k}: {v}' for k, v in scenarios.items())}); facets are the union."
        )

    record = pd.concat(frames, ignore_index=True, sort=False)
    for col in ('population', 'phenotype'):
        if col in record.columns:
            record[col] = _clean_label_series(record[col])
    union = list(dict.fromkeys(record['model'].astype(str)))
    record['model'] = record['model'].astype(str)
    models = canonical_model_order(union)
    wanted = section_cfg.get('models')
    if wanted is not None:
        wanted_set = set(wanted)
        models = [m for m in models if m in wanted_set]
        record = record[record['model'].isin(wanted_set)].reset_index(drop=True)
    # Selection order (not first-appearance order); a source whose rows were all
    # removed by the model filter drops out of `results`.
    present = set(record['result'])
    results = [p['label'] for p in provenance if p['label'] in present]
    return record, models, results, provenance_df


# --------------------------------------------------------------------------- #
# R2 - violin plots
# --------------------------------------------------------------------------- #


def _slot_key(model: str, result: str) -> str:
    return f'{model}{KEY_SEP}{result}'


def _label_height_points(ax, rotation: float) -> float:
    """Height in points of the x tick labels as currently set on `ax` - measured
    with the figure's renderer; a character-count estimate when the backend has
    no `get_renderer` (so the second tier never overlaps the first)."""
    labels = [t for t in ax.get_xticklabels() if t.get_text()]
    if not labels:
        return 0.0
    fig = ax.figure
    try:
        renderer = fig.canvas.get_renderer()
        px = max(t.get_window_extent(renderer).height for t in labels)
        return float(px) * 72.0 / float(fig.dpi)
    except (AttributeError, RuntimeError, ValueError):
        longest = max(len(t.get_text()) for t in labels)
        size = float(labels[0].get_fontsize())
        return longest * 0.62 * size if abs(rotation) >= 45 else 1.6 * size


def _make_axis_hook(models: list, results: list, mode: str, shown: Optional[dict] = None) -> Callable:
    """The per-axis tick hook handed to `metric_plot._render_grid`.

    `_draw_metric_cell` puts the composite `model<US>result` key on the x axis;
    this replaces the tick text according to `mode` (see `X_LABEL_MODES`) and,
    for `two_tier`, adds the model names as a second tier under the bottom row
    of axes (a secondary x axis moved outward by the measured label height, so
    `tight_layout` reserves room for it)."""
    n_res = len(results)
    slots = [(m, r) for m in models for r in results]
    shown = shown or {}                       # ver4-95: {model: text to draw}; display only
    models_txt = [shown.get(m, m) for m in models]
    group_centres = [i * n_res + (n_res - 1) / 2.0 for i in range(len(models))]

    def hook(ax) -> None:
        if mode == 'combined':
            ax.set_xticks(list(range(len(slots))))
            ax.set_xticklabels(
                [f'{shown.get(m, m)} · {r}' for m, r in slots], rotation=45, ha='right', rotation_mode='anchor'
            )
            return
        if mode == 'model_only':
            ax.set_xticks(group_centres)
            ax.set_xticklabels(models_txt, rotation=45, ha='right', rotation_mode='anchor')
            return
        ax.set_xticks(list(range(len(slots))))
        ax.set_xticklabels(
            [r for _m, r in slots], rotation=90, ha='center', va='top', rotation_mode='default'
        )
        if not ax.get_subplotspec().is_last_row():
            return  # inner rows carry no x tick labels at all
        pad = _label_height_points(ax, 90) + 12.0
        tier = ax.secondary_xaxis('bottom')
        tier.set_xticks(group_centres)
        tier.set_xticklabels(models_txt)
        for label in tier.get_xticklabels():
            label.set_fontweight('bold')
        tier.tick_params(axis='x', length=0, pad=2)
        tier.spines['bottom'].set_position(('outward', pad))
        tier.spines['bottom'].set_visible(False)

    return hook


def _plot_options(plot_config: Optional[dict]) -> "tuple[dict, dict, str, str]":
    """`(metric_plot options, merged config, x_label_mode, reference)` for one
    section's `plot_config` (missing keys -> `DEFAULT_PLOT_CONFIG`)."""
    cfg = dict(DEFAULT_PLOT_CONFIG)
    for k, v in (plot_config or {}).items():
        if v is not None:
            cfg[k] = v
    mode = str(cfg.get('x_label_mode', 'two_tier')).strip().lower()
    if mode not in X_LABEL_MODES:
        mode = 'two_tier'
    reference = str(cfg.get('reference') or 'none').strip().lower()
    if reference not in REFERENCE_MODES:
        reference = 'none'
    opts = _mp._resolve_metric_options(cfg)
    opts['_cfg'] = cfg  # carries `_extra_formats` to save_figure
    opts['colour_by'] = 'auto'  # colour = RESULT; keeps the legacy style off 'colorblind'
    opts['reference'] = None  # metric_plot's own 'ensemble' reference is not used
    return opts, cfg, mode, reference


@_ps.themed(
    'ticks',
    lambda a: (a['METRIC_PLOT_CONFIG'] or DEFAULT_PLOT_CONFIG).get(
        'font_size', DEFAULT_PLOT_CONFIG['font_size']
    ),
    lambda a: a['PLOT_DPI'],
)
def comparison_violin_plot(
    record: pd.DataFrame,
    models: list,
    results: list,
    out_dir: str,
    metrics,
    *,
    METRIC_PLOT_CONFIG: Optional[dict] = None,
    PLOT_DPI: int = DEFAULT_DPI,
    prefix: str = 'Comparison_',
) -> "list[str]":
    """Draw one violin figure per metric (a per-population figure and a pooled
    `_total` one) with one violin per (model, result) pair.

    `record` needs `phenotype`, `population`, `model`, `result` and the metric
    columns. x positions are model-major (`slots = [(m, r) for m in models for r
    in results]`), colour is the result's colour (`plot_style.result_colour_map`),
    an empty slot leaves a gap so model groups stay aligned.

    `METRIC_PLOT_CONFIG` takes the `metric_plot` violin keys plus
    `x_label_mode` (`two_tier` | `combined` | `model_only`) and `reference`
    (`none` | `baseline`: a dashed line at the median of the FIRST result's naive
    ensemble). `legend` `off` draws no legend, any other value draws the result
    legend. Files: `<out_dir>/<prefix><metric>.png` and `..._total.png` (plus
    svg/pdf when `_extra_formats` is set in the config).

    Returns the paths written. The figure code is `metric_plot._render_grid`;
    this function only supplies the composite keys, the palette and two hooks."""
    opts, cfg, mode, reference = _plot_options(METRIC_PLOT_CONFIG)
    fig_size = cfg.get('fig_size', DEFAULT_PLOT_CONFIG['fig_size'])
    models = list(models)
    results = list(results)
    slots = [(m, r) for m in models for r in results]
    order = [_slot_key(m, r) for m, r in slots]
    colours = _ps.result_colour_map(results)
    palette = {_slot_key(m, r): colours[r] for m, r in slots}

    data = record.reset_index(drop=True).copy()
    data['model'] = data['model'].astype(str)
    data['result'] = data['result'].astype(str)
    data = data[data['model'].isin(models) & data['result'].isin(results)].copy()
    pops = data['population'].astype(str)
    if opts['between_label'] == 'test' and pops.str.contains('->').any():
        pops = pops.map(lambda p: p.split('->')[-1] if '->' in p else p)
    data['population'] = pops
    data['x'] = [_slot_key(m, r) for m, r in zip(data['model'], data['result'])]
    data['model'] = data['x']

    separators = None
    if opts['group_separators'] and len(models) > 1:
        separators = [i * len(results) - 0.5 for i in range(1, len(models))]
    reference_label = None
    if reference == 'baseline':
        naive = [m for m in models if is_naive_ensemble_label(m)]
        if naive:
            reference_label = _slot_key(naive[0], results[0])
        else:
            print(
                "[tool] NOTE: reference='baseline' requested but there is no naive ensemble "
                "among the models - no reference line drawn."
            )
    aspect = None
    if opts['auto_width']:
        aspect = max(1.0, len(slots) * _mp._AUTO_WIDTH_PER_MODEL / float(fig_size))
    handles = (
        {}
        if opts['legend'] == 'off'
        else {r: Patch(facecolor=colours[r], edgecolor='0.2', linewidth=0.5) for r in results}
    )
    hook = _make_axis_hook(models, results, mode, _ps.shown_labels(models, cfg))   # ver4-95
    _ps.log_resolved(
        'tool_plot',
        **{k: v for k, v in opts.items() if not k.startswith('_') and k != 'reference'},
        x_label_mode=mode,
        reference=reference,
        n_models=len(models),
        n_results=len(results),
    )

    os.makedirs(out_dir, exist_ok=True)
    written: list = []
    for metric in metrics:
        ax_share = False if metric == 'MSE' else True
        base = os.path.join(out_dir, f'{prefix}{metric}')
        for suffix, row in (('.png', 'population'), ('_total.png', None)):
            path = base + suffix
            _mp._render_grid(
                data,
                order,
                metric,
                ax_share,
                fig_size,
                path,
                row=row,
                opts=opts,
                palette=palette,
                separators=separators,
                reference_label=reference_label,
                aspect=aspect,
                axis_hook=hook,
                legend_handles=handles,
            )
            written.append(path)
            written.extend(
                os.path.splitext(path)[0] + '.' + f
                for f in _ps.extra_formats_of(opts['_cfg'])
                if os.path.isfile(os.path.splitext(path)[0] + '.' + f)
            )
    return written


# --------------------------------------------------------------------------- #
# R2 - summary workbook
# --------------------------------------------------------------------------- #


def _slot_columns(models: list, results: list) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [(m, r) for m in models for r in results], names=['model', 'result']
    )


def _statistic_tables(
    record: pd.DataFrame, models: list, results: list, metric: str, phenotypes: list
) -> "dict[str, pd.DataFrame]":
    """`{statistic: wide frame}` for one metric: rows = phenotypes, columns =
    `(model, result)` slots, every value from `metric_summary.six_statistics`
    (I3). Slots without data are NaN."""
    grouped = record.groupby(['phenotype', 'model', 'result'], sort=False, dropna=False)
    stats = _ms.six_statistics(grouped, metric)
    columns = _slot_columns(models, results)
    out = {}
    for name, series in stats.items():
        wide = series.unstack(['model', 'result'])
        out[name] = wide.reindex(index=phenotypes, columns=columns)
    return out


def build_comparison_pivot(
    record: pd.DataFrame, models: list, results: list, metric: str, statistic: str = 'mean'
) -> pd.DataFrame:
    """One summary block as a frame: rows = phenotypes (first-seen order),
    columns = a `(model, result)` MultiIndex in slot order (model-major).
    `statistic` is one of `metric_summary.PIVOT_STATISTICS`. Pure."""
    if statistic not in _STATS_ORDER:
        raise ValueError(f"unknown statistic {statistic!r} (use one of {', '.join(_STATS_ORDER)})")
    phenotypes = list(pd.unique(record['phenotype']))
    return _statistic_tables(record, models, results, metric, phenotypes)[statistic]


def _plus_table(value: pd.DataFrame, se: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """`'value±se'` text cells built from the values ROUNDED to `decimals` (the
    same convention as `metric_summary.six_statistics`); NaN where either part
    is undefined."""
    out = pd.DataFrame(index=value.index, columns=value.columns, dtype=object)
    for col in value.columns:
        for idx in value.index:
            v, s = value.at[idx, col], se.at[idx, col]
            out.at[idx, col] = (
                f'{v:.{decimals}f}±{s:.{decimals}f}' if pd.notna(v) and pd.notna(s) else np.nan
            )
    return out


def _prepare_block(
    table: pd.DataFrame,
    *,
    decimals: int,
    better: str,
    mark_best: bool,
    numeric: bool,
    rank_table: Optional[pd.DataFrame] = None,
) -> "list[tuple]":
    """`[(phenotype, [(value, best), ...]), ...]` for one block.

    `value` is a float (numeric block), a str (`value±SE` block) or None (missing
    -> the writer prints `-`). `best` marks, per phenotype row, every cell whose
    value - at display precision - equals the best one (max for `higher`, min for
    `lower`), judged on `rank_table` (default: the table itself) so a text block
    is ranked on its underlying value. Missing cells are ignored."""
    ranks = table if rank_table is None else rank_table
    rows = []
    for phenotype in table.index:
        cells = []
        best_cols: set = set()
        if mark_best:
            shown = {
                c: round(float(ranks.at[phenotype, c]), decimals)
                for c in table.columns
                if pd.notna(ranks.at[phenotype, c])
            }
            if shown:
                target = (max if better == 'higher' else min)(shown.values())
                best_cols = {c for c, d in shown.items() if d == target}
        for col in table.columns:
            v = table.at[phenotype, col]
            if pd.isna(v):
                cells.append((None, False))
            elif numeric:
                cells.append((float(v), col in best_cols))
            else:
                cells.append((str(v), col in best_cols))
        rows.append((phenotype, cells))
    return rows


def _text_of(value, best: bool, decimals: int) -> str:
    """Cell text for the CSV degrade (and tests): `-` for missing, the number at
    `decimals` or the text, `*` appended for the best value."""
    if value is None:
        return '-'
    text = f'{value:.{decimals}f}' if isinstance(value, float) else str(value)
    return text + ('*' if best else '')


def _summary_blocks(
    record: pd.DataFrame,
    models: list,
    results: list,
    metrics,
    *,
    statistic: str,
    decimals: int,
    all_statistics: bool,
) -> "list[dict]":
    """The ordered blocks of the `summary` sheet (headline statistic, one block
    per metric) or of `summary_all_statistics` (every metric x statistic).
    Each block: `title`, `rows` (see `_prepare_block`)."""
    phenotypes = list(pd.unique(record['phenotype']))
    blocks = []
    for metric in metrics:
        better = _BETTER.get(metric, 'higher')
        tables = _statistic_tables(record, models, results, metric, phenotypes)
        if not all_statistics:
            title = metric if statistic == 'mean' else f'{metric} ({statistic})'
            blocks.append(
                {
                    'title': title,
                    'rows': _prepare_block(
                        tables[statistic],
                        decimals=decimals,
                        better=better,
                        mark_best=True,
                        numeric=True,
                    ),
                }
            )
            continue
        for name in _STATS_ORDER:
            title = f'{metric} {name}'
            if name in ('median', 'mean'):
                rows = _prepare_block(
                    tables[name], decimals=decimals, better=better, mark_best=True, numeric=True
                )
            elif name.endswith('plus standard error'):
                base = name.split(' ')[0]
                text = _plus_table(tables[base], tables[f'{base} standard error'], decimals)
                rows = _prepare_block(
                    text,
                    decimals=decimals,
                    better=better,
                    mark_best=True,
                    numeric=False,
                    rank_table=tables[base],
                )
            else:
                rows = _prepare_block(
                    tables[name], decimals=decimals, better=better, mark_best=False, numeric=True
                )
            blocks.append({'title': title, 'rows': rows})
    return blocks


def _write_block(
    ws, top_row: int, title: str, rows: list, models: list, results: list, *, decimals: int
) -> int:
    """Write one block into worksheet `ws` starting at `top_row`; returns the
    first free row AFTER the block's trailing blank row.

    Layout (E9): row `top_row` the title (bold); `top_row+1` 'Phenotype' (merged
    with the row below) and each model name merged over its `len(results)`
    columns; `top_row+2` the result names; then one row per phenotype. Numeric
    cells keep their number (format `0.000`, or `0.000"*"` for the best value,
    which also gets a bold font); missing cells are the centred text `-`."""
    from openpyxl.styles import Alignment, Font

    bold, centre = Font(bold=True), Alignment(horizontal='center', vertical='center')
    n_res = len(results)
    ws.cell(row=top_row, column=1, value=title).font = bold
    ws.cell(row=top_row + 1, column=1, value='Phenotype')
    ws.merge_cells(start_row=top_row + 1, start_column=1, end_row=top_row + 2, end_column=1)
    ws.cell(row=top_row + 1, column=1).font = bold
    ws.cell(row=top_row + 1, column=1).alignment = centre
    for i, model in enumerate(models):
        c0 = 2 + i * n_res
        cell = ws.cell(row=top_row + 1, column=c0, value=model)
        cell.font, cell.alignment = bold, centre
        if n_res > 1:
            ws.merge_cells(
                start_row=top_row + 1,
                start_column=c0,
                end_row=top_row + 1,
                end_column=c0 + n_res - 1,
            )
        for j, result in enumerate(results):
            ws.cell(row=top_row + 2, column=c0 + j, value=result).alignment = centre
    base_format = '0.' + '0' * decimals if decimals > 0 else '0'
    r = top_row + 3
    for phenotype, cells in rows:
        ws.cell(row=r, column=1, value=phenotype)
        for k, (value, best) in enumerate(cells):
            cell = ws.cell(row=r, column=2 + k)
            if value is None:
                cell.value, cell.alignment = '-', centre
            elif isinstance(value, float):
                cell.value = value
                cell.number_format = base_format + ('"*"' if best else '')
                if best:
                    cell.font = bold
            else:
                cell.value = value + ('*' if best else '')
                if best:
                    cell.font = bold
        r += 1
    return r + 1


def _fit_columns(ws, results: list, models: list, phenotypes: list) -> None:
    from openpyxl.utils import get_column_letter

    ws.column_dimensions['A'].width = max([12] + [len(str(p)) + 2 for p in phenotypes])
    n_res = len(results)
    for i in range(len(models)):
        for j, result in enumerate(results):
            ws.column_dimensions[get_column_letter(2 + i * n_res + j)].width = max(
                10, len(result) + 2
            )


def _cell_value(value):
    """A value openpyxl can store: NaN -> None, numpy scalars -> Python, a tuple
    ratio -> its text."""
    if isinstance(value, (list, tuple, dict, set)):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_frame(ws, frame: pd.DataFrame) -> None:
    ws.append([str(c) for c in frame.columns])
    for row in frame.itertuples(index=False, name=None):
        ws.append([_cell_value(v) for v in row])


def _long_sheets(
    record: pd.DataFrame, models: list, results: list, metrics
) -> "dict[str, pd.DataFrame]":
    """The three `Metric_summary` long sheets, once per result, stacked with a
    leading `result` column (every statistic from `build_metric_summary`)."""
    extra = tuple(m for m in metrics if m not in ('Pearson correlation', 'MSE'))
    pieces: "dict[str, list]" = {name: [] for name in _PHENOTYPE_SHEETS}
    for result in results:
        part = record[record['result'] == result]
        sheets = _ms.build_metric_summary(part, model_order=models, extra_metrics=extra)
        for name in _PHENOTYPE_SHEETS:
            if name in sheets:
                frame = sheets[name].copy()
                frame.insert(0, 'result', result)
                pieces[name].append(frame)
    return {name: pd.concat(parts, ignore_index=True) for name, parts in pieces.items() if parts}


def _csv_rows(blocks: list, models: list, results: list, decimals: int) -> "list[list]":
    """Sheet-1 style rows for the CSV degrade: title, header (model name only in
    the first column of its group), result names, data, blank row."""
    rows: "list[list]" = []
    for block in blocks:
        rows.append([block['title']] + [''] * (len(models) * len(results)))
        rows.append(
            ['Phenotype'] + [(m if j == 0 else '') for m in models for j in range(len(results))]
        )
        rows.append([''] + [r for _m in models for r in results])
        for phenotype, cells in block['rows']:
            rows.append([phenotype] + [_text_of(v, b, decimals) for v, b in cells])
        rows.append([])
    return rows


def write_comparison_summary(
    record: pd.DataFrame,
    models: list,
    results: list,
    out_dir: str,
    tag: str,
    metrics,
    provenance: pd.DataFrame,
    *,
    summary_statistic: str = 'mean',
    decimals: int = DEFAULT_DECIMALS,
    mark_best: bool = True,
) -> "list[str]":
    """Write `Comparison_<tag>_summary.xlsx` (sheets in this order: `summary`,
    `summary_all_statistics`, `phenotype`, `phenotype_population`,
    `phenotype_population_ratio`, `sources`) - or, when `openpyxl` is not
    installed, the same tables as CSV (`Comparison_<tag>_summary.csv`,
    `..._summary_all_statistics.csv`, one CSV per long sheet and
    `..._sources.csv`), with a loud warning. Returns the paths written.

    `summary` is the layout of `example.xlsx` (see module docstring) for the
    headline statistic (`summary_statistic`, `mean` or `median`);
    `summary_all_statistics` repeats it for the six `Metric_summary` statistics.
    `mark_best=False` leaves the `*` out everywhere. Every number comes from
    `metric_summary.six_statistics`."""
    if summary_statistic not in SUMMARY_STATISTICS:
        raise ValueError(
            f"summary_statistic must be one of {SUMMARY_STATISTICS}, got {summary_statistic!r}"
        )
    decimals = max(0, min(int(decimals), 6))
    metrics = list(metrics)
    os.makedirs(out_dir, exist_ok=True)
    kw = dict(statistic=summary_statistic, decimals=decimals)
    headline = _summary_blocks(record, models, results, metrics, all_statistics=False, **kw)
    everything = _summary_blocks(record, models, results, metrics, all_statistics=True, **kw)
    if not mark_best:
        for blocks in (headline, everything):
            for block in blocks:
                block['rows'] = [(p, [(v, False) for v, _b in cells]) for p, cells in block['rows']]
    long_sheets = _long_sheets(record, models, results, metrics)
    stem = os.path.join(out_dir, f'Comparison_{tag}_summary')
    phenotypes = list(pd.unique(record['phenotype']))
    try:
        from openpyxl import Workbook
    except ImportError:
        print(
            f"[tool] WARNING: openpyxl is not installed, so {stem}.xlsx cannot be written "
            f"(install it with `pip install openpyxl`). The same tables are written as CSV instead."
        )
        written = []
        for suffix, blocks in (('', headline), ('_all_statistics', everything)):
            path = f'{stem}{suffix}.csv'
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerows(_csv_rows(blocks, models, results, decimals))
            written.append(path)
        for name, frame in long_sheets.items():
            path = f'{stem}_{name}.csv'
            frame.to_csv(path, index=False)
            written.append(path)
        path = f'{stem}_sources.csv'
        provenance.to_csv(path, index=False)
        written.append(path)
        return written

    wb = Workbook()
    wb.remove(wb.active)
    for sheet_name, blocks in (('summary', headline), ('summary_all_statistics', everything)):
        ws = wb.create_sheet(sheet_name)
        row = 1
        for block in blocks:
            row = _write_block(
                ws, row, block['title'], block['rows'], models, results, decimals=decimals
            )
        _fit_columns(ws, results, models, phenotypes)
        if sheet_name == 'summary':
            ws.freeze_panes = 'B4'
    for name, frame in long_sheets.items():
        _write_frame(wb.create_sheet(name), frame)
    _write_frame(wb.create_sheet('sources'), provenance)
    path = f'{stem}.xlsx'
    wb.save(path)
    return [path]


# --------------------------------------------------------------------------- #
# Validation and the runner
# --------------------------------------------------------------------------- #


def _section_title(section: str) -> str:
    return (SCATTER_SPEC if section == SCATTER_SECTION else SECTIONS[section])['title']


def _section_defaults(section: str, scfg: dict) -> dict:
    """`scfg` with the legacy-preserving defaults of every optional key filled in."""
    out = dict(scfg)
    if section == SCATTER_SECTION:
        out.setdefault('create_plot', SCATTER_SPEC['create_plot'])
        out.setdefault('metrics', list(SCATTER_METRICS))
        out.setdefault('models', None)
        out.setdefault('dpi', DEFAULT_DPI)
        out.setdefault('plot_config', {})
        return out
    out.setdefault('create_plot', SECTIONS[section]['create_plot'])
    out.setdefault('create_summary', True)
    out.setdefault('summary_statistic', 'mean')
    out.setdefault('summary_decimals', DEFAULT_DECIMALS)
    out.setdefault('mark_best', True)
    out.setdefault('models', None)
    out.setdefault('dpi', DEFAULT_DPI)
    out.setdefault('plot_config', {})
    return out


def _active(section: str, scfg: dict) -> bool:
    d = _section_defaults(section, scfg)
    if section == SCATTER_SECTION:
        return bool(d['create_plot'])
    return bool(d['create_plot'] or d['create_summary'])


def validate_tool_config(cfg: dict) -> "list[str]":
    """Every problem of a Tool config as one sentence each (empty list = valid).

    Checks: `RUN_KIND == 'tool'`; `RESULT_NAME` is a plain folder name that is
    neither a selected source folder nor a discovered result folder (tool output
    must never mix with a run's own files); `TOOL_GATHER_MODE`; and, for every
    ACTIVE section (a plot or a summary requested): at least two sources, unique
    non-empty names, every folder readable under the gather mode with a valid
    option, and `models` either None or non-empty."""
    errors: "list[str]" = []
    if not isinstance(cfg, dict) or cfg.get('RUN_KIND') != 'tool':
        return [
            "this is not a Tool config (RUN_KIND must be 'tool'); run it with its own "
            "runner (run_sequential.py / run_step1_batch.py / run_step2_assemble.py are for "
            "pipeline configs)."
        ]
    name = cfg.get('RESULT_NAME')
    if not isinstance(name, str) or not name.strip():
        errors.append("RESULT_NAME (the output folder name) is empty.")
        name = ''
    elif os.sep in name or '/' in name or '\\' in name or name in ('.', '..') or SOURCE_SEP in name:
        errors.append(
            f"RESULT_NAME {name!r} must be a plain folder name (no path separators, '..' "
            f"or {SOURCE_SEP!r})."
        )
    mode = cfg.get('TOOL_GATHER_MODE', 'auto')
    if mode not in GATHER_MODES:
        errors.append(f"TOOL_GATHER_MODE {mode!r} is not one of {', '.join(GATHER_MODES)}.")
        mode = 'auto'
    comparison = cfg.get('TOOL_COMPARISON')
    if comparison is None and _td.is_requested(cfg):
        comparison = {}      # ver4-101: a diagnostics-only Tool config has no comparison section
    if not isinstance(comparison, dict):
        errors.append("TOOL_COMPARISON is missing or not a dictionary of sections.")
        return errors
    errors.extend(_td.validate(cfg, lambda folder: source_status(folder, mode)))
    for key in comparison:
        if key not in ALL_SECTIONS:
            errors.append(
                f"TOOL_COMPARISON has an unknown section {key!r} "
                f"(known: {', '.join(ALL_SECTIONS)})."
            )
    discovered = {s.folder for s in discover_result_sources()}
    status_cache: dict = {}
    for section in ALL_SECTIONS:
        if section not in comparison:
            continue
        scfg = comparison[section] or {}
        if not _active(section, scfg):
            continue
        title = _section_title(section)
        if section == SCATTER_SECTION:
            wanted_metrics = _section_defaults(section, scfg)['metrics']
            unknown = (
                [m for m in wanted_metrics if m not in SCATTER_METRICS]
                if isinstance(wanted_metrics, (list, tuple))
                else None
            )
            if not wanted_metrics or unknown is None:
                errors.append(
                    f"Section '{title}': 'metrics' must be a non-empty list "
                    f"(choose from: {', '.join(SCATTER_METRICS)})."
                )
            elif unknown:
                errors.append(
                    f"Section '{title}': unknown metric(s) {', '.join(repr(m) for m in unknown)} "
                    f"(choose from: {', '.join(SCATTER_METRICS)})."
                )
        sources = scfg.get('sources') or []
        if len(sources) < 2:
            errors.append(
                f"Section '{title}': select at least two results to compare "
                f"(found {len(sources)})."
            )
        labels = [str(s.get('label') or '').strip() for s in sources]
        if any(not label for label in labels):
            errors.append(f"Section '{title}': every selected result needs a non-empty name.")
        if any(KEY_SEP in label for label in labels):
            errors.append(f"Section '{title}': result names must not contain {KEY_SEP!r}.")
        dupes = sorted({label for label in labels if label and labels.count(label) > 1})
        if dupes:
            errors.append(
                f"Section '{title}': result names must be unique (duplicated: "
                f"{', '.join(repr(d) for d in dupes)})."
            )
        models = scfg.get('models')
        if models is not None and (not isinstance(models, list) or not models):
            errors.append(
                f"Section '{title}': 'models' must be null (all models) or a non-empty list."
            )
        for src in sources:
            folder = str(src.get('folder', ''))
            if name and (name == folder or name in discovered):
                errors.append(
                    f"RESULT_NAME {name!r} is the folder of a finished run - choose a "
                    f"different output folder so tool outputs never mix with a run's files."
                )
            if folder not in status_cache:
                status_cache[folder] = source_status(folder, mode)
            status = status_cache[folder]
            if status['error']:
                errors.append(f"Section '{title}': {status['error']}.")
                continue
            option = src.get('option')
            if status['options'] and option is None:
                errors.append(
                    f"Section '{title}': {folder!r} holds several ASReml options "
                    f"({', '.join(status['options'])}); select one option per source."
                )
            elif option is not None and option not in status['options']:
                errors.append(f"Section '{title}': {folder!r} has no ASReml option {option!r}.")
    return list(dict.fromkeys(errors))


def _run_scatter_section(
    cfg: dict, scfg: dict, mode: str, out_dir: str, log: Callable[[str], None], result: dict
) -> None:
    """The plot-only "Comparison of metrics between results" section (ver4-60).

    Same contract as one pass of the `run_tool` section loop: read and stack the
    sources (`build_comparison_record`, model filter applied), skip with a NOTE when
    fewer than two usable sources remain, draw inside a try/except so a failure is
    reported without costing the other sections, and record what was written,
    skipped or failed in `result`."""
    section = SCATTER_SECTION
    title = SCATTER_SPEC['title']
    if not scfg['create_plot']:
        result['skipped'][section] = 'comparison plots not requested'
        log(f"[tool] section '{title}': nothing requested - skipped.")
        return
    result['requested'].append(section)
    metrics = [m for m in SCATTER_METRICS if m in set(scfg['metrics'])]
    log(
        f"[tool] section '{title}': {len(scfg['sources'])} source(s); metrics={metrics}; "
        f"models={'all' if scfg['models'] is None else scfg['models']}."
    )
    try:
        record, models, results, _prov = build_comparison_record(scfg, mode, log)
    except ValueError as exc:
        log(f"[tool] ERROR: section '{title}': {exc}")
        result['errors'].append(f"Section '{title}': {exc}")
        return
    if len(results) < 2:
        reason = f"fewer than two usable sources remain ({len(results)})"
        result['skipped'][section] = reason
        log(f"[tool] NOTE: section '{title}' skipped - {reason}.")
        return
    if record.shape[0] == 0 or not models:
        result['skipped'][section] = 'the model filter left no rows'
        log(f"[tool] NOTE: section '{title}' skipped - the model filter left no rows.")
        return
    plot_cfg = dict(DEFAULT_SCATTER_PLOT_CONFIG)
    plot_cfg.update({k: v for k, v in (scfg['plot_config'] or {}).items() if v is not None})
    plot_cfg = _ps.with_extra_formats(plot_cfg, cfg) or {}
    scenario = (
        'between' if record['population'].astype(str).str.contains('->').any() else 'within'
    )
    import accuracy_metrics_plot as _amp  # lazy: the headless module import stays light

    try:
        written = _amp.metric_across_results_plot(
            record,
            models,
            out_dir,
            scenario,
            results,
            metrics=metrics,
            config=plot_cfg,
            dpi=int(scfg['dpi']),
            prefix=SCATTER_PREFIX,
            log=log,
        )
    except Exception as exc:  # noqa: BLE001 - logged, re-reported
        log(
            f"[tool] ERROR: section '{title}': the comparison plots failed - {exc!r}\n"
            f"{traceback.format_exc()}"
        )
        result['errors'].append(f"Section '{title}': comparison plots failed ({exc!r})")
        return
    if not written:
        reason = 'no task is shared by any two results for the selected metrics'
        result['skipped'][section] = reason
        log(f"[tool] NOTE: section '{title}' wrote nothing - {reason}.")
        return
    result['written'].extend(written)


def run_tool(cfg: dict, *, log: Callable[[str], None] = _DEFAULT_LOG) -> dict:
    """Run a Tool config: validate it, then per requested section load the
    sources, filter models, drop sources lacking the section's metrics (NOTE),
    skip the section when fewer than two sources remain (NOTE), and write the
    violins and/or the workbook - each in its own try/except, so one failure
    never costs the other artefact or the other sections. The plot-only
    `SCATTER_SECTION` (ver4-60) runs last, after the three violin/workbook sections.

    Returns `{'written': [paths], 'skipped': {section: reason}, 'errors':
    [messages], 'requested': [sections]}`. `errors` is non-empty after a
    validation failure (nothing else is attempted then) or a plot / workbook
    exception (logged with its traceback); `run_tool.py` turns that into the
    exit code. A non-requested section is not an error."""
    result = {'written': [], 'skipped': {}, 'errors': [], 'requested': []}
    problems = validate_tool_config(cfg)
    if problems:
        for p in problems:
            log(f"[tool] ERROR: {p}")
        result['errors'] = problems
        return result
    mode = cfg.get('TOOL_GATHER_MODE', 'auto')
    out_dir = result_dir_path(cfg['RESULT_NAME'])
    extra = _ps.normalise_extra_formats(cfg.get('PLOT_EXTRA_FORMATS'))
    log(
        f"[tool] performance comparison; engine {cfg.get('ENGINE_VERSION', 'unknown')}; "
        f"output folder {out_dir}; gather mode {mode!r}; extra figure formats {list(extra)}."
    )
    comparison = cfg.get('TOOL_COMPARISON') or {}
    for section in SECTIONS:
        if section not in comparison:
            continue
        scfg = _section_defaults(section, comparison[section] or {})
        title = SECTIONS[section]['title']
        if not (scfg['create_plot'] or scfg['create_summary']):
            result['skipped'][section] = 'neither violin plots nor a summary requested'
            log(f"[tool] section '{title}': nothing requested - skipped.")
            continue
        result['requested'].append(section)
        log(
            f"[tool] section '{title}': {len(scfg['sources'])} source(s); plots={scfg['create_plot']}, "
            f"summary={scfg['create_summary']} ({scfg['summary_statistic']}, "
            f"{scfg['summary_decimals']} dp, mark_best={scfg['mark_best']}); "
            f"models={'all' if scfg['models'] is None else scfg['models']}."
        )
        try:
            record, models, results, provenance = build_comparison_record(
                scfg, mode, log, section=section
            )
        except ValueError as exc:
            log(f"[tool] ERROR: section '{title}': {exc}")
            result['errors'].append(f"Section '{title}': {exc}")
            continue
        if len(results) < 2:
            reason = f"fewer than two usable sources remain ({len(results)})"
            result['skipped'][section] = reason
            log(f"[tool] NOTE: section '{title}' skipped - {reason}.")
            continue
        if record.shape[0] == 0 or not models:
            result['skipped'][section] = 'the model filter left no rows'
            log(f"[tool] NOTE: section '{title}' skipped - the model filter left no rows.")
            continue
        metrics = SECTIONS[section]['metrics']
        if scfg['create_plot']:
            plot_cfg = _ps.with_extra_formats(dict(scfg['plot_config']), cfg) or {}
            try:
                result['written'].extend(
                    comparison_violin_plot(
                        record,
                        models,
                        results,
                        out_dir,
                        metrics,
                        METRIC_PLOT_CONFIG=plot_cfg,
                        PLOT_DPI=int(scfg['dpi']),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - logged, re-reported
                log(
                    f"[tool] ERROR: section '{title}': the violin plots failed - {exc!r}\n"
                    f"{traceback.format_exc()}[tool] the summary workbook (if requested) is still attempted."
                )
                result['errors'].append(f"Section '{title}': violin plots failed ({exc!r})")
        if scfg['create_summary']:
            try:
                result['written'].extend(
                    write_comparison_summary(
                        record,
                        models,
                        results,
                        out_dir,
                        SECTIONS[section]['tag'],
                        metrics,
                        provenance,
                        summary_statistic=scfg['summary_statistic'],
                        decimals=int(scfg['summary_decimals']),
                        mark_best=bool(scfg['mark_best']),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - logged, re-reported
                log(
                    f"[tool] ERROR: section '{title}': the summary workbook failed - {exc!r}\n"
                    f"{traceback.format_exc()}"
                )
                result['errors'].append(f"Section '{title}': summary workbook failed ({exc!r})")
    if SCATTER_SECTION in comparison:
        _run_scatter_section(
            cfg,
            _section_defaults(SCATTER_SECTION, comparison[SCATTER_SECTION] or {}),
            mode,
            out_dir,
            log,
            result,
        )
    if _td.is_requested(cfg):   # ver4-101: per-run diagnostics (optional, default absent)
        _td.run_diagnostics(cfg, out_dir, _result_dir, log, result)
    for path in result['written']:
        log(f"[tool] wrote {path}")
    log(
        f"[tool] done: {len(result['written'])} file(s) written, {len(result['skipped'])} section(s) "
        f"skipped, {len(result['errors'])} error(s)."
    )
    return result
