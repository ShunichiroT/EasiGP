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
