"""
model_registry.py
==================
Update ID ver4-5, R2 (blueprint §4.2, invariant I14).

Single source of truth for "which models can emit which interpretability
artefact (marker-pair interactions and/or GAT attention weights)". Before
this module existed, four different places in the codebase each answered
this question independently by comparing a model name against the string
literal 'RF' (or, for attention, a short hardcoded tuple) - see the
blueprint's Appendix B8 grep evidence:

    genomic_prediction.py:2950   if base_model_name == 'RF' and ...
    checkpoint_utils.py:295      'interactions': any(base_of(m) == 'RF' ...)
    circos_plot.py:921           model_selected += ['RF']
    circos_plot.py:957           interaction_selected['model'] = 'RF'  (after
                                  dropping the 'model' column and averaging
                                  every selected model's pairs together -
                                  actively wrong whenever a second emitter
                                  is selected, not just incomplete)

Every one of those call sites now reads this module instead, so a new
interaction-emitting model needs to be registered in exactly ONE place for
every downstream consumer to see it correctly.

Deliberately has ZERO imports from the rest of this project (I2) - every
other project module is free to import FROM here (including
genomic_prediction.py, which several of the OTHER modules above are
themselves imported by), with no risk of a circular import.

A model appearing in INTERACTION_ARTEFACT means it CAN emit that artefact -
not that it WILL. Actual emission is still gated by that model's own opt-in
HPARAMETERS field (e.g. RF's `get_interaction`, GAT_prior_knowledge's new
`emit_interaction`), which defaults to False for every model except RF
(whose default stays True, preserving today's exact behaviour).

Model names here are always BASE, schema-key names (see
models.hyperparameter_tuning.base_of / schema_key_of) - a hyperparameter-
tuning-algorithm suffix ('RF__Grid') or a bio-prior instance suffix
('GAT_biological_prior_knowledge_2') is never a key in this dict directly;
callers normalise first (base_of() then schema_key_of()), exactly as
HPARAM_SPECS and DIAGNOSTIC_FLAG_FIELDS already require.

Update ID ver4-5, R2 Stages 8-11: Tier 1 (``ExtraTrees``, ``GBDT``) and
Tier 2 (``XGBoost``, ``EBM``) are now registered and wired end to end -
see ``models/ExtraTrees.py``, ``models/GBDT.py``, ``models/XGBoost.py``,
``models/EBM.py``. ``rrBLUP``/``GBLUP``/``BayesB``/``RKHS``/``SVR``/
``KNN``/``MLP`` (registered as CAPABLE since Stages 1-3, but not yet
WIRED to actually emit) are now also fully wired - see those models' own
files and ``genomic_prediction.py``'s dispatch branches.

Tier 2 models are OPTIONAL dependencies (``xgboost``, ``interpret`` -
neither is a hard requirement, invariant I12) - ``optional_dependency_
for()``/``is_available()`` below exist so every downstream consumer
(the GUI's model picker, headless config validation) can find out
whether a Tier 2 model can actually run BEFORE a run starts, rather than
mid-run after hours of fitting other models (blueprint §4.6 failure-mode
table).
"""

from __future__ import annotations
import importlib

# Base model name -> which interpretability artefact(s) it is CAPABLE of
# writing:
#   'interactions' - Interaction.csv (marker1, marker2, value pairs)
#   'attention'    - Attention.csv (GAT edge attention weights)
#   'both'         - both of the above, from two independent computations
#   None           - neither (see NO_INTERACTION_CAPABILITY_REASON below)
INTERACTION_ARTEFACT = {
    'rrBLUP': 'interactions',
    'GBLUP': 'interactions',
    'BayesB': 'interactions',
    'RKHS': 'interactions',
    'RF': 'interactions',
    'ExtraTrees': 'interactions',
    'GBDT': 'interactions',
    'XGBoost': 'interactions',   # Tier 2 - optional dependency, see is_available()
    'EBM': 'interactions',       # Tier 2 - optional dependency, see is_available()
    'SVR': 'interactions',
    'KNN': 'interactions',
    'MLP': 'interactions',
    'GAT_infinitesimal': None,
    'GAT_infinitesimal_node_level': None,
    'GAT_fully_connected': 'attention',
    'GAT_prior_knowledge': 'both',
    'GAT_biological_prior_knowledge': 'attention',
}

# Base model name -> the OPTIONAL third-party package it needs to actually
# run (not just to emit interactions - the whole model). A model absent
# from this dict has no optional dependency at all (either it needs
# nothing beyond this project's own hard-pinned packages, or - for the R
# models - it needs the R/BGLR toolchain, which this dict deliberately
# does not model, since that check already happens elsewhere via
# configure_r_environment()/rpy2 itself, not an importable Python package).
#
# Update ID ver4-12, R13: 'asreml' (Preprocess/asreml_blup.py's Step 6b
# stage) is deliberately NOT added here either, for a related but distinct
# reason - it is not a MODEL at all (it does not appear in MODEL/MODEL_RUN),
# and it is not merely an importable Python package (it is a licensed R
# package requiring per-machine activation on top of installation) - see
# Preprocess/asreml_license.py's own module docstring for where that fact
# is owned instead.
OPTIONAL_DEPENDENCY = {
    'XGBoost': 'xgboost',
    'EBM': 'interpret',
}

# is_available() result cache, keyed by package name - an import probe is
# cheap but not FREE (it can trigger the imported package's own module-
# level initialisation), and this is checked repeatedly (once per
# AVAILABLE_MODELS render in the GUI, once per config-validation pass) -
# cached so it only actually runs the probe once per process. Never
# invalidated within a process: whether a package is importable does not
# change over a single run's lifetime.
_AVAILABILITY_CACHE: "dict[str, bool]" = {}

# Human-readable reason a model has NO interaction/attention capability at
# all - surfaced in a one-time log line (architecture doc §17's "log
# negative states unconditionally" principle) rather than a user silently
# wondering why a model produced no ring. Note that 'rrBLUP'/'BayesB'/
# 'GBLUP'/'RKHS'/'SVR'/'KNN'/'MLP' are registered as CAPABLE of
# 'interactions' here (per the ver4-5 blueprint's design, §4.2) even though
# none of them actually emit interactions in THIS delivery yet (that wiring
# - H-statistic/surrogate-TreeSHAP extraction, appended HPARAMETERS fields
# - is Requirement 2's later stages, not implemented this pass; see the
# Change Summary). Only the two models that are structurally INCAPABLE of
# ever producing a marker-marker edge, regardless of what extraction method
# might be added later, get an entry here.
NO_INTERACTION_CAPABILITY_REASON = {
    'GAT_infinitesimal': (
        "it uses self-loops only (architecture doc §9.3) - there is no "
        "marker-marker edge for any interaction/attention artefact to "
        "describe."
    ),
    'GAT_infinitesimal_node_level': (
        "it uses a heterogeneous qtl->pheno / qtl->qtl(self) topology "
        "(architecture doc §9.3) - there is no marker-marker edge for any "
        "interaction/attention artefact to describe."
    ),
}

# Models whose interaction values are the model's OWN exact computation
# (TreeSHAP for the tree ensembles; EBM's own native pairwise terms; the
# pairwise SHAP GAT_prior_knowledge already computes to build its own
# graph topology), as opposed to a surrogate/approximate method built to
# reach a model with no native interaction API (RK-8: H-statistic for
# SVR/KNN, NID for MLP, surrogate-TreeSHAP for the four R/BGLR models).
# Informational only - never gates whether extraction runs - but IS
# consulted by main_app.py's GUI help text and by every interaction-
# emitting model's own log line (blueprint §4.7 acceptance criterion 9:
# "Surrogate output is labelled `approximate` ... never quietly presented
# as equivalent"), so the exact/approximate distinction is visible to the
# person reading the output, not just recorded here.
EXACT_INTERACTION_MODELS = frozenset({
    'RF', 'ExtraTrees', 'GBDT', 'XGBoost', 'EBM', 'GAT_prior_knowledge',
})

# Base model name -> its DEFAULT interaction-extraction method, purely
# descriptive (used in log lines and GUI help text - never consulted to
# decide whether extraction runs, since that is each model's own
# get_interaction/emit_interaction HPARAMETERS field). A model absent
# from this dict either has no interaction capability at all (see
# NO_INTERACTION_CAPABILITY_REASON) or emits attention only.
INTERACTION_METHOD = {
    'RF': 'TreeSHAP (exact)',
    'ExtraTrees': 'TreeSHAP (exact)',
    'GBDT': 'TreeSHAP (exact)',
    'XGBoost': 'TreeSHAP (exact)',
    'EBM': 'native pairwise terms (exact)',
    'GAT_prior_knowledge': 'pairwise SHAP, already computed for its own graph topology (exact)',
    'SVR': 'Friedman H-statistic (model-agnostic, approximate ranking)',
    'KNN': 'Friedman H-statistic (model-agnostic, approximate ranking)',
    'MLP': 'Neural Interaction Detection (from first-layer weights, approximate ranking)',
    'rrBLUP': 'surrogate TreeSHAP (APPROXIMATE - see RK-8)',
    'BayesB': 'surrogate TreeSHAP (APPROXIMATE - see RK-8)',
    'GBLUP': 'surrogate TreeSHAP (APPROXIMATE - see RK-8)',
    'RKHS': 'surrogate TreeSHAP (APPROXIMATE - see RK-8)',
}


def _normalise(base_name):
    if base_name is None:
        raise ValueError("model_registry: a base model name is required (got None).")
    return str(base_name)


def emits_interactions(base_name) -> bool:
    """True if `base_name` (a BASE, schema-key model name - see module
    docstring) is capable of writing rows to Interaction.csv. Does not
    check any per-model opt-in flag (e.g. RF's own `get_interaction`) -
    callers combine this with the model's own HPARAMETERS-driven toggle
    where that distinction matters. An unregistered name (not present in
    INTERACTION_ARTEFACT at all) is treated as incapable, not an error -
    callers are never required to register every model they might ever
    pass through here."""
    artefact = INTERACTION_ARTEFACT.get(_normalise(base_name))
    return artefact in ('interactions', 'both')


def emits_attention(base_name) -> bool:
    """True if `base_name` is capable of writing rows to Attention.csv.
    See emits_interactions()'s docstring for the same caveats."""
    artefact = INTERACTION_ARTEFACT.get(_normalise(base_name))
    return artefact in ('attention', 'both')


def has_no_interaction_capability(base_name) -> bool:
    """True only for a model that IS registered (present in
    INTERACTION_ARTEFACT) but is structurally incapable of producing
    either artefact - distinct from a model that is simply unregistered
    (unknown to this module altogether), which returns False here."""
    name = _normalise(base_name)
    return name in INTERACTION_ARTEFACT and INTERACTION_ARTEFACT[name] is None


def no_interaction_capability_reason(base_name) -> str:
    """Human-readable reason has_no_interaction_capability(base_name) is
    True - used by the one-time log line GP() prints for each such
    selected model (see genomic_prediction.py). Returns a generic
    fallback if the specific model has no recorded reason (should not
    happen for any name currently mapped to None in INTERACTION_ARTEFACT,
    but this never raises)."""
    return NO_INTERACTION_CAPABILITY_REASON.get(
        _normalise(base_name),
        "it has no marker-marker edge to describe.",
    )


def is_exact_interaction_model(base_name) -> bool:
    """True if this model's interaction values are its OWN exact
    computation, as opposed to a surrogate/approximate one (RK-8).
    Informational only."""
    return _normalise(base_name) in EXACT_INTERACTION_MODELS


def interaction_method_default(base_name) -> str:
    """Human-readable description of `base_name`'s DEFAULT interaction-
    extraction method - purely descriptive (log lines, GUI help text).
    Returns a generic fallback for a model with no recorded method
    (never raises) rather than assuming every registered model has one."""
    return INTERACTION_METHOD.get(_normalise(base_name), 'no recorded method')


def optional_dependency_for(base_name) -> "str | None":
    """The importable package name `base_name` needs to run AT ALL (not
    just to emit interactions), or None if it needs no optional
    dependency (either it uses only this project's own hard-pinned
    packages, or it is an R model - see OPTIONAL_DEPENDENCY's own
    docstring note for why R models are deliberately absent here)."""
    return OPTIONAL_DEPENDENCY.get(_normalise(base_name))


def is_available(base_name) -> bool:
    """True if `base_name` can actually be dispatched in THIS process -
    i.e. it has no optional dependency at all, OR that dependency
    imports successfully. NEVER RAISES (an import failure of any kind -
    ModuleNotFoundError, or any other exception a package's own module-
    level code might throw - is caught and treated as 'unavailable', not
    propagated), so this is always safe to call from GUI rendering code.
    Cached per package name for the lifetime of this process (see
    _AVAILABILITY_CACHE's own module-level docstring) - a package's
    importability does not change mid-run."""
    package = optional_dependency_for(base_name)
    if package is None:
        return True
    if package not in _AVAILABILITY_CACHE:
        try:
            importlib.import_module(package)
            _AVAILABILITY_CACHE[package] = True
        except Exception:
            _AVAILABILITY_CACHE[package] = False
    return _AVAILABILITY_CACHE[package]


def is_installed(base_name) -> bool:
    """Cheap, GUI-rendering sibling of ``is_available()``: True if `base_name` has no optional
    dependency, or that dependency is INSTALLED (``importlib.util.find_spec`` - located on the
    path, never imported/executed). ``is_available()`` really imports the package; for
    ``xgboost`` and especially ``interpret`` (EBM) that is several seconds of cold-import time -
    far longer on a shared HPC filesystem - and the GUI used to pay it on EVERY launch just to
    decide whether to draw a checkbox. The GUI therefore draws the picker from this function and
    only does the full ``is_available()`` import probe when a Run is started (and ``GP()`` still
    runs its own ``is_available()`` preflight before task 0, unchanged), so a package that is
    installed but cannot actually be imported is still caught before any model is fitted.
    NEVER RAISES. Cached per package name for the life of the process."""
    package = optional_dependency_for(base_name)
    if package is None:
        return True
    if package not in _INSTALLED_CACHE:
        try:
            import importlib.util
            _INSTALLED_CACHE[package] = importlib.util.find_spec(package) is not None
        except Exception:
            _INSTALLED_CACHE[package] = False
    return _INSTALLED_CACHE[package]


_INSTALLED_CACHE: "dict[str, bool]" = {}


def known_models():
    """Every base model name this registry currently knows about, in
    INTERACTION_ARTEFACT's own insertion order - lets a caller iterate the
    full registered set instead of hardcoding it a second time."""
    return list(INTERACTION_ARTEFACT.keys())


# ---------------------------------------------------------------------------
# Update ID ver4-9, R6 (blueprint SS2.3.2): a single, additional authority -
# which of the four parametric/semiparametric genomic-prediction models
# ("conventional", diag010.pdf's own term) a base model name belongs to,
# versus every machine-learning model - used ONLY by weight_plot.py's own
# two-colour-family (blue/green) palette. Additive: no existing name in
# this module is touched.
# ---------------------------------------------------------------------------
CONVENTIONAL_MODELS = frozenset({'rrBLUP', 'GBLUP', 'BayesB', 'RKHS'})


def model_family(base_name) -> str:
    """`'conventional'` for the four parametric/semiparametric genomic
    prediction models in `CONVENTIONAL_MODELS`; `'machine_learning'` for
    everything else, INCLUDING a name this registry has never seen before
    (an unregistered/unrecognised model is still, definitionally, not one
    of the four conventional ones) - so this NEVER raises, and is always
    safe to call from plotting code with whatever model name a run's own
    Weight.csv happens to contain, however that name got there."""
    return 'conventional' if _normalise(base_name) in CONVENTIONAL_MODELS else 'machine_learning'


# ---------------------------------------------------------------------------
# ver4-98 (X04; the class map of the ver4-91 P3 design): the architecture class of a model, so
# that reports can add up weights per class instead of quoting raw weights among near-duplicates
# (ver4-69: BayesB 0.94 vs rrBLUP 0 at equal out-of-fold r). Additive and pure: nothing in the
# engine reads it, so no run changes. `bio_prior` is its own class because that model reads a
# different information source (I6).
# ---------------------------------------------------------------------------
MODEL_CLASS = {
    'rrBLUP': 'additive', 'GBLUP': 'additive',
    'BayesB': 'sparse',
    'RKHS': 'kernel', 'SVR': 'kernel',
    'RF': 'tree', 'ExtraTrees': 'tree', 'GBDT': 'tree', 'XGBoost': 'tree', 'EBM': 'tree',
    'KNN': 'local',
    'MLP': 'neural',
    'GAT_infinitesimal': 'graph', 'GAT_infinitesimal_node_level': 'graph',
    'GAT_fully_connected': 'graph', 'GAT_prior_knowledge': 'graph',
    'GAT_biological_prior_knowledge': 'bio_prior',
}


def model_class(label) -> str:
    """Architecture class of a model label (`MODEL_CLASS`). A tuning-algorithm suffix
    (``RF__Grid``) and a bio-prior instance suffix (``..._2``) are stripped first; a name this
    registry has never seen is ``'other'``. Never raises on a string; None raises like the rest
    of this module."""
    base = _lane_schema_key(_normalise(label).split('__', 1)[0])
    return MODEL_CLASS.get(base, 'other')


# ---------------------------------------------------------------------------
# ver4-102 (D01 / R2; the model-class coverage guidance of the ver4-72 and ver4-77 findings).
# Pure and additive: nothing here changes a result, `GP()` only PRINTS the note. Three groups of
# classes, each of which a model list should cover with at least one member:
#   sparse        - BayesB. A sparse member is what an oligogenic trait needs; ver4-72 measured
#                   +0.007 / +0.047 / +0.019 r (polygenic / oligogenic / epistatic, small
#                   simulation, 4 models + an elastic net vs 4 models) from adding one.
#   smooth        - 'additive' (rrBLUP, GBLUP) or 'kernel' (RKHS, SVR): the polygenic backbone.
#   nonlinear     - 'tree' (RF, ExtraTrees, GBDT, XGBoost, EBM) or 'local' (KNN): interactions.
# Neural, graph and bio_prior members are never required. The measured gains are simulation
# evidence, not real-data confirmation (catalogue V03), and the Super learner down-weights a
# member that does not help, so a missing class is advice, not an error.
# ---------------------------------------------------------------------------
CLASS_COVERAGE_GROUPS = {
    'sparse': ('sparse',),
    'smooth': ('additive', 'kernel'),
    'nonlinear': ('tree', 'local'),
}
#: One example model per group, in the order a user would be advised to add them.
CLASS_COVERAGE_EXAMPLES = {'sparse': 'BayesB', 'smooth': 'rrBLUP or GBLUP', 'nonlinear': 'RF or KNN'}


def class_coverage(models) -> dict:
    """Which `CLASS_COVERAGE_GROUPS` a list of model labels covers.

    `models` is any iterable of labels (tuning and instance suffixes are handled by
    `model_class`; the label ``'ensemble'`` is ignored). Returns ``{'covered': [...], 'missing':
    [...], 'n_models': int}`` with group names in the order of `CLASS_COVERAGE_GROUPS`. Never
    raises on strings; an empty list covers nothing."""
    classes = {model_class(m) for m in models if str(m) != 'ensemble'}
    covered, missing = [], []
    for group, members in CLASS_COVERAGE_GROUPS.items():
        (covered if classes.intersection(members) else missing).append(group)
    return {'covered': covered, 'missing': missing,
            'n_models': len([m for m in models if str(m) != 'ensemble'])}


def class_coverage_note(models):
    """The one-line guidance `GP()` prints, or None when nothing is worth saying (fewer than two
    models, or every group covered)."""
    info = class_coverage(models)
    if info['n_models'] < 2 or not info['missing']:
        return None
    advice = '; '.join(f"{g}: add {CLASS_COVERAGE_EXAMPLES[g]}" for g in info['missing'])
    return (f"model-class coverage - the model list has no member from the "
            f"{', '.join(info['missing'])} group(s) ({advice}). In simulation, adding a missing "
            f"class to a naive ensemble raised r by up to 0.05 (measured for the sparse "
            f"class); the Super learner weights a useless member down. Guidance only - nothing "
            f"is changed. Not confirmed on real data.")


# ---------------------------------------------------------------------------
# ver4-35, R1.f - asynchronous model lane: which models may be fitted in the
# lane worker, and which process-global random streams each model consumes.
#
# Why this exists: the lane (async_lane.py) fits a lane-capable model in a
# second process while the parent fits CPU models. A fit's result depends on
# the process-global random streams (torch / numpy / python `random`) it draws
# from, and the legacy serial run feeds them to the models in MODEL_RUN
# order. The lane can only reproduce that order exactly if no model that runs
# in the PARENT draws from a stream a lane model also draws from - so the
# planner (``async_lane.plan_lane_chains``) needs to know, per model, which
# streams are consumed. Both tables live here (zero project imports, I2) so
# every consumer reads one authority.
# ---------------------------------------------------------------------------

#: Base (schema-key) model names the lane may fit. Start: the bio-prior GAT
#: only. Extend ONLY after blueprint check P5 (``ModelTrialRunner(...)`` equals
#: the main-dispatch call, all seven outputs ``DataFrame.equals``) passes for
#: the model on a real run - MLP and the other GAT variants are candidates.
LANE_CAPABLE = frozenset({'GAT_biological_prior_knowledge'})

#: Base model name -> frozenset of global random streams the model draws from
#: ('torch', 'numpy', 'random').
#:
#: Source: static census of this tree (blueprint check P6 is the RUNTIME
#: census - digests before/after each model on a real serial CPU run - and is
#: still open; see the ver4-35 Change Summary). Findings:
#:   * the MLP and every GAT variant draw from the torch CPU stream (weight
#:     init, dropout, DataLoader shuffling);
#:   * every scikit-learn / gradient-boosting model, the R models' Python-side
#:     surrogates and the bio-prior merge pass an explicit ``random_state``
#:     (``0``/``40``) or a seeded ``default_rng``, so they leave the global
#:     streams alone;
#:   * RKHS / SVR / KNN can explain through ``shap.KernelExplainer`` /
#:     ``shapiq`` whose sampling is not proven seed-free, so they are marked
#:     'numpy' (conservative; irrelevant while no lane model uses numpy).
#: A model ABSENT from this table is "unknown" and is treated by the planner
#: as consuming every stream a lane model consumes (never assumed safe).
GLOBAL_RNG_STREAMS = {
    'rrBLUP': frozenset(),
    'GBLUP': frozenset(),
    'BayesB': frozenset(),
    'RKHS': frozenset({'numpy'}),
    'RF': frozenset(),
    'ExtraTrees': frozenset(),
    'GBDT': frozenset(),
    'XGBoost': frozenset(),
    'EBM': frozenset(),
    'SVR': frozenset({'numpy'}),
    'KNN': frozenset({'numpy'}),
    'MLP': frozenset({'torch'}),
    'GAT_infinitesimal': frozenset({'torch'}),
    'GAT_infinitesimal_node_level': frozenset({'torch'}),
    'GAT_fully_connected': frozenset({'torch'}),
    'GAT_prior_knowledge': frozenset({'torch'}),
    'GAT_biological_prior_knowledge': frozenset({'torch'}),
}


def _lane_schema_key(name) -> str:
    """Strip a bio-prior instance suffix (``..._2``) - the same rule as
    ``models.hyperparameter_tuning.schema_key_of`` (not imported: this module
    has zero project imports, I2). Tuning-algorithm suffixes (``__Grid``) must
    already be removed by the caller (``base_of``)."""
    text = _normalise(name)
    prefix = 'GAT_biological_prior_knowledge_'
    if text.startswith(prefix) and text[len(prefix):].isdigit():
        return 'GAT_biological_prior_knowledge'
    return text


def lane_capable(name) -> bool:
    """True if model `name` (a base name, bio-prior instance suffix allowed)
    may be fitted by the asynchronous lane. Unregistered names are not
    lane-capable, never an error."""
    return _lane_schema_key(name) in LANE_CAPABLE


def global_rng_streams(name):
    """The global random streams model `name` draws from, as a frozenset, or
    ``None`` when the model is not in :data:`GLOBAL_RNG_STREAMS` (unknown -
    the caller must treat it conservatively)."""
    return GLOBAL_RNG_STREAMS.get(_lane_schema_key(name))


# ---------------------------------------------------------------------------
# ver4-36, R1.c / R1.d - joint-search speed levers: which models may be fitted
# with a 1-row `test` stub during the joint SEARCH, and which have an exact
# closed-form kernel search engine.
#
# Why this lives here (I3): `models/joint_tuning.py` must not keep a second list
# of "which models are BGLR-style". Both tables are plain frozensets (zero
# project imports, I2) read through the accessors below.
# ---------------------------------------------------------------------------

#: Models whose R function runs `rbind(train, valid, test)`, masks valid + test
#: and fits over every row, while the joint search only reads
#: ``predicted_valid`` - so a search fit may be given ``test.iloc[:1]`` instead
#: of the whole test frame (R1.c). Confirm fits always get the full test frame.
SEARCH_TEST_STUB_SAFE = frozenset({'rrBLUP', 'GBLUP', 'BayesB', 'RKHS'})

#: Models whose joint-SEARCH fit can be replaced by an exact Gaussian-process
#: posterior mean (closed form, milliseconds) - BRR (rrBLUP) and BGLR-RKHS
#: (R1.d). BayesB has no closed form and GBLUP is deliberately left out (its
#: kernel is built elsewhere); the confirm fit is always the real BGLR fit.
EXACT_KERNEL_SEARCH_MODELS = frozenset({'rrBLUP', 'RKHS'})


def search_test_stub_safe(base_name) -> bool:
    """True if a joint-search fit of model `base_name` (a base name; a
    bio-prior instance suffix is irrelevant because no bio-prior model is in the
    set) may be handed a 1-row `test` frame. Unregistered names are not safe,
    never an error."""
    return _normalise(base_name) in SEARCH_TEST_STUB_SAFE


def exact_kernel_search_capable(base_name) -> bool:
    """True if model `base_name` has an exact closed-form joint-search engine
    (``models.kernel_effects.exact_kernel_column``). Unregistered names are not
    capable, never an error."""
    return _normalise(base_name) in EXACT_KERNEL_SEARCH_MODELS
