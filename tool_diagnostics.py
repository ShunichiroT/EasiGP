"""Tool mode - per-run diagnostics (ver4-101).

The comparison sections of Tool mode (`result_comparison.py`) compare FINISHED runs with
each other. The five offline tools of `tools/` look at ONE finished run each:

* ``prediction_diagnostics`` - calibration slope (O02) and within-group accuracy (V06);
* ``member_importance``      - share of the ensemble weight per model class (X03);
* ``member_shapley``         - Shapley value of each member's contribution (X04);
* ``consensus_stability``    - stability of the top markers / pairs across iterations;
* ``candidate_reliability``  - out-of-domain flag and conformal intervals (O03 / O04);
* ``class_hierarchy``        - the hierarchical equal-weight ensemble next to the flat one (H07);
* ``pairwise_diversity``     - which pairs of members are near-duplicates (D13).

This module is the single place that runs them from a Tool config, so the GUI (Option B),
the exported ``tool_config.json`` and ``run_tool.py`` all use the same code. It is an
OPTIONAL top-level key of the Tool config, ``TOOL_DIAGNOSTICS``; it is NOT a member of
``TOOL_COMPARISON`` (that dictionary only holds the sections that need two or more runs),
so a Tool config without the key behaves exactly as before (I11).

Nothing in a finished run is ever written to: the files of the run ``<folder>`` go to
``Result/<RESULT_NAME>/Diagnostics/<folder>/``.

Config (all keys but ``folders`` optional)::

    "TOOL_DIAGNOSTICS": {
        "folders": ["run1", "run2"],      # result sub-folders, at least one
        "tools":   ["prediction_diagnostics", "member_importance", ...],   # default: the four
                                          # that need no extra file (the two newest, class_hierarchy
                                          # and pairwise_diversity, are offered but not ticked)
        "split":   "test",                # prediction table for diagnostics / Shapley
        "top_k":   50,                    # consensus_stability
        "pairs":   false,                 # consensus_stability: also the pair table
        "alpha":   0.1,                   # candidate_reliability miscoverage
        "genotypes": "",                  # candidate_reliability: genotype CSV (required for it)
        "groups":  "",                    # prediction_diagnostics: ID,group CSV (optional)
        "abstain": false,                 # candidate_reliability: also Candidate_abstention.csv
        "ood_inflate": 1.0,               # candidate_reliability: widen flagged lines' intervals (>= 1)
        "interaction_frequency": false    # consensus_stability: also Interaction_selection_frequency.csv
    }                                     # (the last three are written by the GUI only when set)

Import discipline (I2): only the standard library at import time; the tools (which need
pandas / numpy) are imported inside `run_diagnostics`.
"""
import os
import sys
import traceback
from typing import Callable, Dict, List, Optional

#: Offered tools, in display order: name -> one-line description (the GUI help text).
DIAGNOSTIC_TOOLS: Dict[str, str] = {
    'prediction_diagnostics': "Calibration slope and within-group accuracy of the predictions",
    'member_importance': "Ensemble weight per model class",
    'member_shapley': "Shapley value of each ensemble member (Pearson r of the coalition mean)",
    'consensus_stability': "Stability of the selected markers (and pairs) across iterations",
    'candidate_reliability': "Out-of-domain flag and prediction intervals per candidate "
                             "(needs a genotype file)",
    'class_hierarchy': "Equal weights by model class next to flat equal weights (reads the "
                       "test predictions)",
    'pairwise_diversity': "Which pairs of members make the same errors (reads the test "
                          "predictions)",
}

#: The tools that need nothing but the run's own files - ticked by default.
DEFAULT_TOOLS = ('prediction_diagnostics', 'member_importance', 'member_shapley',
                 'consensus_stability')

DEFAULTS = {
    'tools': list(DEFAULT_TOOLS),
    'split': 'test',
    'top_k': 50,
    'pairs': False,
    'alpha': 0.1,
    'genotypes': '',
    'groups': '',
    'abstain': False,          # ver4-103: each new option is off, so an old config runs as before
    'ood_inflate': 1.0,
    'interaction_frequency': False,
}

SUBFOLDER = 'Diagnostics'
SPLITS = ('test', 'valid')
EXIT_OK, EXIT_ERROR, EXIT_NOTHING = 0, 1, 3

_TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools')


def with_defaults(dcfg: Optional[dict]) -> dict:
    """`dcfg` with every optional key filled in (a copy)."""
    out = dict(dcfg or {})
    for key, value in DEFAULTS.items():
        out.setdefault(key, list(value) if isinstance(value, list) else value)
    out.setdefault('folders', [])
    return out


def is_requested(cfg: dict) -> bool:
    """True when the Tool config asks for diagnostics (a non-empty folder list)."""
    dcfg = cfg.get('TOOL_DIAGNOSTICS') if isinstance(cfg, dict) else None
    return bool(isinstance(dcfg, dict) and dcfg.get('folders'))


def validate(cfg: dict, status_of: Callable[[str], dict]) -> List[str]:
    """Every problem of the ``TOOL_DIAGNOSTICS`` block of `cfg` (empty list = valid, and
    also when the block is absent). `status_of(folder)` is `result_comparison.source_status`
    bound to the gather mode: it supplies the "folder does not exist" sentence."""
    if 'TOOL_DIAGNOSTICS' not in cfg or cfg['TOOL_DIAGNOSTICS'] is None:
        return []
    dcfg = cfg['TOOL_DIAGNOSTICS']
    if not isinstance(dcfg, dict):
        return ["TOOL_DIAGNOSTICS must be a dictionary."]
    if not dcfg.get('folders'):
        return []          # nothing requested: like an unticked section, not an error
    errors: List[str] = []
    title = "Per-run diagnostics"
    d = with_defaults(dcfg)
    folders = d['folders']
    if not isinstance(folders, list) or not all(isinstance(f, str) and f for f in folders):
        return [f"{title}: 'folders' must be a list of result folder names."]
    if len(set(folders)) != len(folders):
        errors.append(f"{title}: a result folder is listed twice.")
    name = str(cfg.get('RESULT_NAME') or '')
    for folder in folders:
        if name and name == folder:
            errors.append(
                f"RESULT_NAME {name!r} is the folder of a finished run - choose a different "
                f"output folder so tool outputs never mix with a run's files."
            )
        status = status_of(folder)
        if status.get('error'):
            errors.append(f"{title}: {status['error']}.")
    tools = d['tools']
    if not isinstance(tools, list) or not tools:
        errors.append(f"{title}: 'tools' must be a non-empty list "
                      f"(choose from: {', '.join(DIAGNOSTIC_TOOLS)}).")
        tools = []
    else:
        unknown = [t for t in tools if t not in DIAGNOSTIC_TOOLS]
        if unknown:
            errors.append(f"{title}: unknown tool(s) {', '.join(repr(t) for t in unknown)} "
                          f"(choose from: {', '.join(DIAGNOSTIC_TOOLS)}).")
    if d['split'] not in SPLITS:
        errors.append(f"{title}: 'split' must be one of {', '.join(SPLITS)}.")
    try:
        if int(d['top_k']) < 1:
            raise ValueError
    except (TypeError, ValueError):
        errors.append(f"{title}: 'top_k' must be a whole number of at least 1.")
    try:
        if not 0.0 < float(d['alpha']) < 1.0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append(f"{title}: 'alpha' must be between 0 and 1 (0.1 = 90% intervals).")
    if 'candidate_reliability' in tools:
        genotypes = str(d['genotypes'] or '').strip()
        if not genotypes:
            errors.append(f"{title}: 'candidate_reliability' needs a genotype CSV ('genotypes').")
        elif not os.path.isfile(genotypes):
            errors.append(f"{title}: the genotype file {genotypes!r} was not found.")
    groups = str(d['groups'] or '').strip()
    if groups and 'prediction_diagnostics' in tools and not os.path.isfile(groups):
        errors.append(f"{title}: the groups file {groups!r} was not found.")
    for flag in ('abstain', 'interaction_frequency'):
        if not isinstance(d[flag], bool):
            errors.append(f"{title}: '{flag}' must be true or false.")
    try:
        if isinstance(d['ood_inflate'], bool) or float(d['ood_inflate']) < 1.0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append(f"{title}: 'ood_inflate' must be a number of at least 1 (1 = no inflation).")
    return errors


def _argv(tool: str, folder_path: str, out_dir: str, d: dict) -> List[str]:
    """The command-line arguments one tool is called with."""
    argv = ['--result', folder_path, '--out-dir', out_dir]
    if tool == 'prediction_diagnostics':
        argv += ['--split', d['split']]
        if str(d['groups'] or '').strip():
            argv += ['--groups', str(d['groups']).strip()]
    elif tool == 'member_shapley':
        argv += ['--split', d['split']]
    elif tool == 'consensus_stability':
        argv += ['--top-k', str(int(d['top_k']))]
        if d['pairs']:
            argv += ['--pairs']
    elif tool == 'candidate_reliability':
        argv += ['--genotypes', str(d['genotypes']).strip(), '--alpha', str(float(d['alpha']))]
        if d['abstain'] is True:
            argv += ['--abstain']
        if float(d['ood_inflate']) > 1.0:
            argv += ['--ood-inflate', str(float(d['ood_inflate']))]
    if tool == 'consensus_stability' and d['interaction_frequency'] is True:
        argv += ['--interaction-frequency']
    return argv


def _snapshot(directory: str) -> Dict[str, float]:
    if not os.path.isdir(directory):
        return {}
    return {n: os.path.getmtime(os.path.join(directory, n)) for n in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, n))}


def _import_tool(tool: str):
    """The module `tools/<tool>.py`. The tools import each other by bare name, so `tools/`
    (and the source root) go on `sys.path` first - the same thing running them as scripts does."""
    import importlib
    root = os.path.dirname(_TOOLS_DIR)
    for path in (_TOOLS_DIR, root):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module(tool)


def run_diagnostics(cfg: dict, out_dir: str, result_dir_of: Callable[[str], str],
                    log: Callable[[str], None], result: dict) -> None:
    """Run the requested tools for every listed run and record the outcome in `result`
    (same dictionary `run_tool` returns: 'written', 'skipped', 'errors', 'requested').

    One failing tool never costs the others: each call is wrapped, an exit code 3 ("nothing
    to evaluate", e.g. a run without Marker_effect) is a NOTE, and only an exit code 1 or an
    exception is an error. `result_dir_of(folder)` gives a run's folder path."""
    d = with_defaults(cfg.get('TOOL_DIAGNOSTICS'))
    title = 'Per-run diagnostics'
    section = 'diagnostics'
    tools = [t for t in DIAGNOSTIC_TOOLS if t in d['tools']]
    result['requested'].append(section)
    log(f"[tool] section '{title}': {len(d['folders'])} run(s); tools={tools}; "
        f"split={d['split']}.")
    ran = 0
    n_errors = len(result['errors'])
    for folder in d['folders']:
        folder_path = result_dir_of(folder)
        target = os.path.join(out_dir, SUBFOLDER, folder)
        os.makedirs(target, exist_ok=True)
        for tool in tools:
            label = f"{folder} / {tool}"
            before = _snapshot(target)
            try:
                module = _import_tool(tool)
                code = module.main(_argv(tool, folder_path, target, d))
            except SystemExit as exc:   # an argparse refusal must not stop the other tools
                code = EXIT_ERROR if exc.code not in (0, None) else EXIT_OK
            except Exception as exc:  # noqa: BLE001 - logged, re-reported
                log(f"[tool] ERROR: {label}: {exc!r}\n{traceback.format_exc()}")
                result['errors'].append(f"Diagnostics {label}: {exc!r}")
                continue
            after = _snapshot(target)
            new = sorted(n for n, t in after.items() if n not in before or t > before[n])
            if code == EXIT_OK:
                ran += 1
                result['written'].extend(os.path.join(target, n) for n in new)
            elif code == EXIT_NOTHING:
                log(f"[tool] NOTE: {label}: nothing to evaluate (the files it needs are "
                    f"missing or empty in this run).")
            else:
                log(f"[tool] ERROR: {label}: the tool reported an error (see above).")
                result['errors'].append(f"Diagnostics {label}: the tool reported an error")
    if not ran and len(result['errors']) == n_errors:
        result['skipped'][section] = 'no tool had anything to evaluate in the selected runs'
        log(f"[tool] NOTE: section '{title}' wrote nothing - no tool had anything to evaluate.")
