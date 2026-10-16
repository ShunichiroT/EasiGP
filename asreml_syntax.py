"""
asreml_syntax.py
==================
Update ID ver4-12, R6/R7 (design blueprint SS2.5 "Module layout",
SS2.6, SS2.7).

Single authority translating an ``ASREML`` config dict into ASReml-R
syntax, a human-readable equation, and the estimand (BLUE/BLUP/GBLUP) it
produces - imported by BOTH ``main_app.py`` (Layer 6 - renders the
Equation/Estimand/Syntax panels in Tab 2's ASReml section, and validates
the config at commit time) and ``Preprocess/asreml_blup.py`` (Layer 3a -
the run-time execution engine). Follows the exact same "zero project
imports" discipline ``model_registry.py`` and ``circos_geometry.py``
already established (architecture doc SS2, "Layer 3d - the shared
authorities"): this module imports nothing from the rest of EasiGP, so
both a Streamlit-only consumer and an rpy2-only consumer may import it
with no circularity and no upward dependency (invariant I2).

WHY THIS MODULE EXISTS
--------------------------------------------------------------------------
The GUI needs to (a) render a read-only equation, (b) render an EDITABLE
ASReml-R call the person can hand-tune, and (c) validate the
configuration before it is ever committed to disk. The engine needs to
take that SAME committed text and substitute three placeholders
(``{TRAIT}``, ``{DATA}``, ``{GRM}``) once per task. Building the syntax
string in two different places (GUI preview vs. engine execution) would
silently drift the moment either one changed independently - exactly the
duplication problem ``model_registry.py`` was built to eliminate for
"which models emit interactions" (architecture doc SS10.1). This module is
the single place the mapping from config -> syntax is expressed, imported
by both.

THE THREE PLACEHOLDERS
--------------------------------------------------------------------------
``render_syntax()`` never bakes in a real trait name, a real R data-frame
symbol, or a real inverse-relationship-matrix symbol - it always renders
literal ``{TRAIT}``, ``{DATA}`` (and, when a GRM is configured, ``{GRM}``)
text into the returned string, UNLESS the caller explicitly passes a
concrete ``data_symbol=``/``grm_symbol=`` (used only for a one-off preview
render, never for the text that gets committed to
``Result/<RESULT_NAME>/ASReml/asreml_syntax.R``). ``Preprocess/
asreml_blup.py`` is what actually substitutes these, once per task, since
only it knows this task's own phenotype name and R-side data-frame/Ginv
object names.

ESTIMAND
--------------------------------------------------------------------------
BLUE  - genotype fitted as a FIXED effect (``sparse = ~ Genotype``).
BLUP  - genotype fitted as a RANDOM effect, no genomic relationship matrix
        (``random = ~ Genotype + ...``) - a pedigree-free, purely
        phenotypic BLUP.
GBLUP - genotype fitted as a RANDOM effect WITH a genomic relationship
        matrix (``random = ~ vm(Genotype, source = Ginv) + ...``) - see
        ``Preprocess/asreml_blup.py::build_or_load_grm()`` for where that
        matrix actually comes from (EasiGP's own ``_build_grm()``, not a
        new R dependency - invariant I12).

DESIGN DECISION: BLUE CANNOT BE HELD OUT (Option C - added after a real
production failure, ``log_sequential_local_20260910_140122.txt``)
--------------------------------------------------------------------------
A FIXED genotype term can only ever be estimated for a level (genotype)
that has at least one non-missing observation contributing to the fit.
EasiGP's own Step 6b masks EVERY row of EVERY validation/test
individual's response to NA before fitting (Requirement 8) - which means
a FIXED genotype term's own level for that individual carries zero
information, unconditionally, on every task. This is not an aliasing
accident or a convergence problem (a production run converged cleanly in
9 rounds and still reported this for all 43 held-out individuals) - it
is definitionally true for any fixed effect with no observed data.

Rather than refuse ``genotype_effect='fixed'`` outright whenever a
validation/test split exists, EasiGP fits it as designed for TRAINING
individuals, and separately produces BLUP/GBLUP values (which DO have a
defined value for an unobserved level, via the genotype term's own
variance-component prior) for held-out ones. Concretely, whenever
``genotype_effect='fixed'`` AND a task has any validation/test
individual, ``Preprocess/asreml_blup.py::asreml_adjust_phenotypes()``
fits TWO independent models instead of one:

  Fit 1 ("BLUE, train")   : the user's own committed ``committed_syntax``
      (genotype FIXED, unchanged), on a dataset PHYSICALLY restricted to
      training individuals only - not masked, excluded entirely, so this
      fit's own effect estimates (and every training individual's own
      BLUE) never see a validation/test response at all.
  Fit 2 ("BLUP/GBLUP, holdout") : a SEPARATE committed syntax
      (``ASREML['holdout_committed_syntax']`` - genotype forced RANDOM,
      optionally with a genomic relationship matrix, mirroring
      ``ASREML['use_grm']``), on the FULL dataset with validation/test
      masked exactly as the single-fit path already does. Only its
      valid/test rows are kept; its own training-individual rows are
      discarded (Fit 1's BLUE is authoritative for those).

The result is a genuinely MIXED estimand: 'BLUE' for training rows,
'BLUP'/'GBLUP' for validation/test rows - never a single constant across
the whole result. See ``estimand_description()`` below for the
person-facing one-line summary of this, ``holdout_model_cfg()``/
``suggest_holdout_syntax()`` for how the second model's config/syntax is
derived, and ``Preprocess/asreml_blup.py::_fit_blue_train_plus_holdout()``
for the actual two-fit orchestration. This yields FOUR possible
configurations in total:

  1. genotype_effect='fixed',  use_grm=False -> BLUE (train) + BLUP  (holdout)
  2. genotype_effect='fixed',  use_grm=True  -> BLUE (train) + GBLUP (holdout)
  3. genotype_effect='random', use_grm=False -> BLUP  for everyone, ONE fit
  4. genotype_effect='random', use_grm=True  -> GBLUP for everyone, ONE fit

Patterns 3-4 are the pre-existing single-fit path (unchanged). Patterns
1-2 are new. A THIRD option - one unmasked fit over the whole dataset,
so the genotype's own FIXED effect has data to estimate from even for a
"held-out" individual - was considered and rejected: under
``na.action = na.method(y = "include")`` a masked row contributes
nothing to the likelihood, but an UNMASKED one does, so every held-out
individual's own response would then inform the shared Year/Experiment/
residual variance components used to adjust every OTHER individual too -
a real (if second-order) leakage channel the masked two-fit design
above avoids entirely.

DEVIATION FROM THE DESIGN BLUEPRINT (recorded here, and in the ver4-12
Change Summary SS7): the blueprint's own config-schema table (SS4) types
``ASREML.interaction_groups``/``ASREML.nested_groups`` as a plain
``list[list[str]]``. Requirement 6's own worked example never says
whether an interaction/nesting term is added to the FIXED or the RANDOM
side of the model, and the blueprint itself flags this ambiguity (SS2.6,
"Ambiguity resolved... Flagged for requester confirmation"). Phase 2
resolves the remaining ambiguity by giving each group its own explicit
``'target'`` key (``'fixed'`` or ``'random'``, default ``'random'`` -
AR1 trend terms like Row/Range nesting are far more commonly random
extraneous effects than fixed treatment effects in the reference
scripts), so a group is a ``dict`` (``{'columns': [...], 'target': ...}``),
not a bare ``list[str]``. This is a strictly additive change to the
blueprint's schema (a bare list-of-columns is still accepted and
defaults to ``target='random'`` - see ``_normalise_group()`` below), so a
config that never sets ``'target'`` explicitly still works exactly as the
blueprint describes.

ver4-17 BUGFIX ("Data has N observations but the residual model implies
M" - production failure, task 1/25 of EasiGP_BLUEs_BLUPs_ASReml_0.5,
population='Cluster_4', phenotype='Maturity'): a grid-structured residual
(``ASREML['residual']['dims']`` non-empty - an ``ar1(Range):ar1(Row)``
spatial structure being the motivating case) requires ASReml to see ONE
ROW PER COMBINATION of those dimensions' levels - normally the trial's
own complete physical plot layout, spanning every population/individual
it ever recorded, not just one task's own. Restricting the raw phenotype
records to a single task's own population - exactly what every OTHER
ASReml configuration correctly does, both for efficiency and so a
population's own error-variance estimate is not diluted by unrelated
individuals - silently removes other populations' plots from that
layout whenever population is a genetically-DERIVED subset of one shared
trial (as opposed to a literal per-experiment population), leaving gaps
the grid can no longer fill: ASReml then reports a mismatch between the
actual (subsetted) row count and the row count its own residual
structure implies. See ``residual_requires_full_grid()`` below (the new
single authority both sides of the Python<->R boundary now consult) and
``Preprocess/asreml_blup.py::_fit_single_asreml_model()``'s own updated
row-selection step for the fix. The FIXED-genotype two-fit path
(``_fit_blue_train_plus_holdout()``) is a separate, harder case that
``validate_structure()`` below now refuses outright rather than silently
mishandling - see that function's own comment at the check site for why
grid completeness and Fit 1's leakage-avoidance design cannot both be
satisfied there.

ver4-18 BUGFIX ("Data has N observations but the residual model implies
M" - STILL reproduced after ver4-17, a SEPARATE production failure: task
1/25 of EasiGP_BLUEs_BLUPs_ASReml_0.5, population='Cluster_1',
phenotype='Glaucousness', "Data has 8403 observations but the residual
model implies 11424"): ver4-17 fixed the wrong half of this problem.
Lifting the population restriction is NECESSARY (a genetically-derived
population must not silently drop other populations' plots from the
trial's own Range/Row grid), but it is not SUFFICIENT whenever the raw
phenotype file itself does not contain a row for every combination of
the residual dimensions' own observed levels - a plot that was never
planted, never harvested, or simply never recorded for ANY trait at all
(a border plot, a discarded/failed plot, an irregular field boundary)
is still a missing (Range, Row) combination to ``ar1(Range):ar1(Row)``
no matter how many populations' rows are present, since ver4-17's own
fix already keeps every row the file has. 8403 in the production
failure above is literally the ENTIRE phenotype file's row count;
11424 = 96 x 119, the true number of (Range, Row) level COMBINATIONS
once ``asreml_fit.R``'s own ``factor()`` call collapses each dimension
down to its own observed levels - a gap of 3021 combinations the raw
file never had ANY row for, under any population or trait.

``pad_residual_grid()`` below is the fix: explicitly complete the
Cartesian product of every residual dimension's own OBSERVED levels
(never a synthetic min..max integer run) and add exactly one NaN-filled
"filler" row for every combination the raw file is missing - the
standard field-trial-statistics technique for an incomplete spatial AR1
grid, and exactly what this integration's own unconditional
``na.action = na.method(y = "include", x = "include")`` (below) already
exists to make safe: a filler row's NaN response contributes nothing to
the likelihood (identical to a masked validation/test row), and a
filler row's NaN value in any OTHER model column (genotype included)
does the same for every fixed/random term, rather than ASReml silently
dropping the row (which would undo the padding) or refusing to fit at
all. Called ONCE, up front, by ``genomic_prediction.py`` (the padding
depends only on the phenotype file's own columns, never on any one
task's own individuals) and again, defensively/idempotently, by
``Preprocess/asreml_blup.py::_fit_single_asreml_model()`` - the same
dual-authority pattern ``residual_requires_full_grid()`` itself already
established for this exact class of question.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# BUGFIX ("unexpected symbol" - a real production failure, task 1/25 of
# EasiGP_BLUEs_BLUPs_ASReml_0.5, population='Cluster_3', phenotype='Grain
# Yield'): every place in this codebase that interpolates a column/trait
# name DIRECTLY into rendered ASReml-R text (this module's own
# render_syntax()/_apply_wrapper(), and Preprocess/asreml_blup.py's
# per-task ``{TRAIT}`` substitution) assumed the name was always a bare,
# syntactically valid R identifier. A phenotype trait literally called
# 'Grain Yield' (a space) or 'Grain Size (TKW)' (parentheses) - both
# entirely ordinary column headers, and exactly the kind
# PHENOTYPE_TRAIT_COLUMNS is designed to accept verbatim, never renamed -
# produces `asreml(fixed = Grain Yield ~ 1, ...)`, which R's own
# `parse(text = ...)` (Preprocess/asreml_fit.R) rejects outright:
# "<text>:1:22: unexpected symbol". R accepts ANY string as an identifier
# when backtick-quoted (`` `Grain Yield` ``), so this is the single fix
# point for every such interpolation - never a per-caller workaround.
# ---------------------------------------------------------------------------
_R_BARE_NAME_RE = re.compile(r'^(?!\.\d)[A-Za-z.][A-Za-z0-9._]*$')
# R's own reserved words (?Reserved) - a bare name matching one of these
# is syntactically a keyword, not an identifier, and must be backtick-
# quoted even though it satisfies _R_BARE_NAME_RE above (e.g. a trait
# column someone genuinely named 'TRUE' or 'function').
_R_RESERVED_WORDS = frozenset({
    'if', 'else', 'repeat', 'while', 'function', 'for', 'in', 'next', 'break',
    'TRUE', 'FALSE', 'NULL', 'Inf', 'NaN', 'NA', 'NA_integer_', 'NA_real_',
    'NA_character_', 'NA_complex_',
})


def r_safe_name(name: str) -> str:
    """Render `name` as a syntactically valid R symbol for direct
    interpolation into formula/call-text (never for a plain Python-side
    dict/DataFrame-column lookup, which works with any string regardless
    of R syntax - see this function's own callers for the distinction).

    A name that is ALREADY a bare, valid R identifier (e.g. 'Genotype',
    'Row', 'ar1') - the overwhelming common case - is returned completely
    unchanged, so every existing call site that never needed quoting in
    the first place stays byte-for-byte identical. Anything else (a
    space, a parenthesis, a leading digit, a reserved word such as
    'TRUE') is wrapped in backticks, with any literal backtick or
    backslash inside the name itself escaped first, so the result is
    always a single, valid R symbol no matter what the person's own file
    happened to call the column.
    """
    text = str(name)
    if _R_BARE_NAME_RE.match(text) and text not in _R_RESERVED_WORDS:
        return text
    escaped = text.replace('\\', '\\\\').replace('`', '\\`')
    return f'`{escaped}`'

# ---------------------------------------------------------------------------
# R6 - wrapper coverage, transcribed from ASReml_syntax.pdf SS2 (variance-
# structure CONSTRUCTORS - these build a term's own random-regression basis)
# and SS3 (variance-structure FUNCTIONS - these describe the COVARIANCE
# among the levels of a term already in the model). Each entry records:
#   'arity'      - how many extra positional argument(s) the wrapper takes
#                  beyond the term/column(s) it wraps (0 for most).
#   'args'       - names of those extra arguments, in order (empty tuple
#                  if arity is 0) - the GUI renders one widget per name.
#   'label'      - short human-readable name shown in the GUI's wrapper
#                  selector.
#   'applies_to' - 'random', 'residual', or 'both' - which model
#                  component(s) this wrapper is legal in, so the GUI only
#                  ever offers a wrapper where ASReml itself accepts it
#                  (e.g. 'dsum' is a residual-only complex-group function
#                  per the PDF's own wording).
#   'help'       - one line naming the covariance/basis structure this
#                  wrapper imposes, worded to be traceable back to the PDF
#                  section it came from (R14) - both the GUI and this
#                  registry read the SAME string, so they cannot drift.
# ---------------------------------------------------------------------------
ASREML_CONSTRUCTORS: Dict[str, dict] = {
    'at': {
        'arity': 1, 'args': ('levels',), 'label': 'at() - separate variance per level',
        'applies_to': 'both',
        'help': "Fits a SEPARATE variance (or effect) for each named level of a factor, "
                "rather than assuming every level shares one variance (PDF SS2).",
    },
    'lin': {
        'arity': 0, 'args': (), 'label': 'lin() - linear covariate',
        'applies_to': 'both',
        'help': "Fits the covariate as a plain LINEAR regression term (PDF SS2).",
    },
    'pol': {
        'arity': 1, 'args': ('degree',), 'label': 'pol() - orthogonal polynomial',
        'applies_to': 'both',
        'help': "Fits an orthogonal POLYNOMIAL of the given degree in the covariate, e.g. "
                "for a random-regression trend (PDF SS2).",
    },
    'leg': {
        'arity': 1, 'args': ('degree',), 'label': 'leg() - Legendre polynomial',
        'applies_to': 'both',
        'help': "Fits a LEGENDRE polynomial of the given degree - the standard basis for "
                "random-regression genetic models (PDF SS2).",
    },
    'spl': {
        'arity': 1, 'args': ('df',), 'label': 'spl() - smoothing spline',
        'applies_to': 'both',
        'help': "Fits a smoothing SPLINE with the given degrees of freedom over the "
                "covariate (PDF SS2).",
    },
    'dev': {
        'arity': 0, 'args': (), 'label': 'dev() - deviations from a fitted regression',
        'applies_to': 'both',
        'help': "Fits DEVIATIONS of each level from an already-fitted regression term "
                "(PDF SS2) - used alongside pol()/leg() for a random-regression model.",
    },
    'mv': {
        'arity': 1, 'args': ('formula',), 'label': 'mv() - multivariate structure',
        'applies_to': 'both',
        'help': "Groups several response traits into one MULTIVARIATE model term "
                "(PDF SS2).",
    },
    'grp': {
        'arity': 1, 'args': ('formula',), 'label': 'grp() - named group of covariates',
        'applies_to': 'both',
        'help': "Bundles several covariates into one named GROUP term, fitted with a "
                "single (co)variance parameter (PDF SS2).",
    },
    'mbf': {
        'arity': 1, 'args': ('formula',), 'label': 'mbf() - model based on a factor',
        'applies_to': 'both',
        'help': "Builds a term from an externally supplied per-level design matrix "
                "(PDF SS2) - an advanced option for a custom covariate structure.",
    },
}

ASREML_WRAPPERS: Dict[str, dict] = {
    # Identity & diagonal (PDF SS3).
    'id': {
        'arity': 0, 'args': (), 'label': 'id() - identity (equal variance, no correlation)',
        'applies_to': 'both',
        'help': "Every level shares the SAME variance and is UNCORRELATED with every "
                "other level (PDF SS3) - ASReml's implicit default when no wrapper is "
                "chosen at all.",
    },
    'idv': {
        'arity': 0, 'args': (), 'label': 'idv() - identity, single variance parameter',
        'applies_to': 'both',
        'help': "Identical to id() but forces a SINGLE estimated variance shared across "
                "every level of a multi-level term (PDF SS3).",
    },
    'idh': {
        'arity': 0, 'args': (), 'label': 'idh() - heterogeneous identity',
        'applies_to': 'both',
        'help': "Each level gets its OWN variance, still uncorrelated with every other "
                "level (PDF SS3).",
    },
    'diag': {
        'arity': 0, 'args': (), 'label': 'diag() - diagonal variance matrix',
        'applies_to': 'both',
        'help': "A diagonal (co)variance matrix - equivalent to idh() for most model "
                "terms (PDF SS3).",
    },
    # Correlation structures (PDF SS3).
    'ar1': {
        'arity': 0, 'args': (), 'label': 'ar1() - first-order autoregressive',
        'applies_to': 'both',
        'help': "First-order AUTOREGRESSIVE correlation along an ordered dimension (e.g. "
                "field row/range position) - correlation decays geometrically with "
                "distance (PDF SS3).",
    },
    'ar1h': {
        'arity': 0, 'args': (), 'label': 'ar1h() - heterogeneous AR1',
        'applies_to': 'both',
        'help': "Like ar1(), but allows a DIFFERENT variance at each level while sharing "
                "one autoregressive correlation parameter (PDF SS3).",
    },
    'corv': {
        'arity': 0, 'args': (), 'label': 'corv() - uniform correlation, single variance',
        'applies_to': 'both',
        'help': "A single UNIFORM correlation between every pair of levels, one shared "
                "variance (PDF SS3).",
    },
    'corh': {
        'arity': 0, 'args': (), 'label': 'corh() - uniform correlation, heterogeneous variance',
        'applies_to': 'both',
        'help': "A single UNIFORM correlation between every pair of levels, but each "
                "level gets its own variance (PDF SS3).",
    },
    'corg': {
        'arity': 0, 'args': (), 'label': 'corg() - general correlation',
        'applies_to': 'both',
        'help': "A separate correlation for every PAIR of levels, one shared variance "
                "(PDF SS3).",
    },
    'corgh': {
        'arity': 0, 'args': (), 'label': 'corgh() - general correlation, heterogeneous variance',
        'applies_to': 'both',
        'help': "A separate correlation for every pair of levels AND a separate variance "
                "per level - the most general correlation structure offered (PDF SS3).",
    },
    'exp': {
        'arity': 0, 'args': (), 'label': 'exp() - exponential (continuous distance)',
        'applies_to': 'both',
        'help': "EXPONENTIAL correlation decay as a function of continuous distance "
                "(e.g. real field coordinates rather than an integer row index) "
                "(PDF SS3).",
    },
    'exph': {
        'arity': 0, 'args': (), 'label': 'exph() - exponential, heterogeneous variance',
        'applies_to': 'both',
        'help': "Like exp(), with a separate variance per level (PDF SS3).",
    },
    'gau': {
        'arity': 0, 'args': (), 'label': 'gau() - Gaussian (continuous distance)',
        'applies_to': 'both',
        'help': "GAUSSIAN correlation decay as a function of continuous distance - decays "
                "faster at short range than exp() (PDF SS3).",
    },
    'mat': {
        'arity': 0, 'args': (), 'label': 'mat() - Matern (continuous distance)',
        'applies_to': 'both',
        'help': "MATERN correlation family as a function of continuous distance - "
                "generalises exp()/gau() with a shape parameter (PDF SS3).",
    },
    # General variance structures (PDF SS3).
    'us': {
        'arity': 0, 'args': (), 'label': 'us() - unstructured',
        'applies_to': 'both',
        'help': "A fully UNSTRUCTURED (co)variance matrix - every variance and every "
                "covariance estimated separately, the most flexible (and most "
                "parameter-hungry) structure (PDF SS3).",
    },
    'ante': {
        'arity': 1, 'args': ('order',), 'label': 'ante() - antedependence',
        'applies_to': 'both',
        'help': "An ANTEDEPENDENCE structure of the given order - each level depends "
                "only on the previous 'order' levels (PDF SS3).",
    },
    'chol': {
        'arity': 0, 'args': (), 'label': 'chol() - Cholesky-parameterised unstructured',
        'applies_to': 'both',
        'help': "An unstructured (co)variance matrix, parameterised via its CHOLESKY "
                "factor for numerical stability (PDF SS3).",
    },
    # Factor analytic (PDF SS3).
    'fa': {
        'arity': 1, 'args': ('k',), 'label': 'fa() - factor analytic',
        'applies_to': 'both',
        'help': "A FACTOR-ANALYTIC (co)variance structure with 'k' latent factors - a "
                "parsimonious approximation to us() (PDF SS3).",
    },
    'xfa': {
        'arity': 1, 'args': ('k',), 'label': 'xfa() - extended factor analytic',
        'applies_to': 'both',
        'help': "An EXTENDED factor-analytic structure (adds a specific variance per "
                "level on top of fa()'s k common factors) (PDF SS3).",
    },
    'rr': {
        'arity': 1, 'args': ('k',), 'label': 'rr() - reduced rank',
        'applies_to': 'both',
        'help': "A REDUCED-RANK (co)variance structure of rank 'k', without factor-"
                "analytic's specific-variance term (PDF SS3).",
    },
    # Complex group functions (PDF SS3) - dsum is documented as
    # residual-only.
    'str': {
        'arity': 1, 'args': ('formula',), 'label': 'str() - structured group of terms',
        'applies_to': 'both',
        'help': "Combines several model terms into one STRUCTURED variance group, e.g. "
                "for a direct-sum or Kronecker-product covariance across sub-models "
                "(PDF SS3).",
    },
    'dsum': {
        'arity': 2, 'args': ('formula', 'vmodel'), 'label': 'dsum() - direct sum (residual only)',
        'applies_to': 'residual',
        'help': "Fits an independent residual variance MODEL per level of a conditioning "
                "factor (e.g. a separate residual structure per trial) - a RESIDUAL-only "
                "function (PDF SS3).",
    },
}


def _wrapper_registry() -> Dict[str, dict]:
    """Combined constructor + wrapper registry, for lookups that don't
    care which of the two PDF sections a name came from."""
    combined = dict(ASREML_CONSTRUCTORS)
    combined.update(ASREML_WRAPPERS)
    return combined


# ---------------------------------------------------------------------------
# R6/R7 - the pure, immutable description of one task's mixed model, and
# the render functions that turn it into syntax/equation/estimand text.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """Pure description of one ASReml-R mixed model, independent of which
    trait/data-frame/Ginv symbol it will eventually be rendered against
    (see render_syntax()'s placeholder discussion above).

    Parameters
    ----------
    response : the trait name (rendered as the literal placeholder
        ``{TRAIT}`` by render_syntax(), never a real value here).
    fixed : fixed-effect term strings (already wrapper-decorated, e.g.
        ``('ar1(Row)',)``), EXCLUDING the genotype column - see `sparse`.
    random : random-effect term strings, EXCLUDING the genotype column -
        see `genotype_col`/`genotype_is_random`/`grm`.
    residual : a single already-rendered residual formula string (e.g.
        ``'ar1(Row):ar1(Range)'``), or None for ASReml's own default
        (independent, homogeneous residual variance).
    sparse : fixed-effect term(s) rendered via ASReml's `sparse =`
        argument instead of `fixed =` - used for a many-level FIXED
        genotype term, matching `genotype_fixed_effect_model.R`'s own
        `sparse = ~ Genotype` (efficient for a factor with very many
        levels relative to `fixed =`).
    genotype_col : the column identifying the individual/genotype.
    genotype_is_random : True (BLUP/GBLUP) if `genotype_col` is fitted as
        a random effect; False (BLUE) if fitted as a fixed (sparse) term.
    grm : True if a genomic relationship matrix should wrap the genotype
        random term (`vm(<genotype_col>, source = <grm_symbol>)`) -
        meaningless (ignored) when `genotype_is_random` is False.
    output_level : the column(s) `predict(classify = ...)` groups by -
        also what R2's POPULATION_SOURCE derives `population` from.
    response_unit : ver4-19, Requirements.md R3 - `'genotype'` (default,
        unchanged behaviour: one BLUE/BLUP/GBLUP per `genotype_col`
        level) or `'genotype_by_population'` (a genotype shared across
        more than one population gets its OWN adjusted value per
        population it appears in - see `render_syntax()` below for the
        three per-strategy formulas this selects, and
        `Preprocess/asreml_blup.py::detect_shared_genotypes_across_
        populations()`/`resolve_response_unit()` for how this is
        resolved from the data in the first place).
    population_col : required (and only meaningful) when
        `response_unit == 'genotype_by_population'` - the column
        identifying which population a row belongs to. EasiGP's own
        universal join-key column is always literally `'population'`
        (architecture doc SS4.1) - this field exists mainly so
        `render_syntax()` never hard-codes that string itself, matching
        every other column-name field on this dataclass.
    """
    response: str
    fixed: Tuple[str, ...] = ()
    random: Tuple[str, ...] = ()
    residual: Optional[str] = None
    sparse: Tuple[str, ...] = ()
    genotype_col: str = ''
    genotype_is_random: bool = True
    grm: bool = False
    output_level: Tuple[str, ...] = ()
    response_unit: str = 'genotype'
    population_col: Optional[str] = None


def _normalise_group(group) -> dict:
    """Accepts either the blueprint's own bare ``list[str]`` shape or
    Phase 2's additive ``{'columns': [...], 'target': 'fixed'|'random'}``
    shape (see this module's own docstring, "Deviation from the design
    blueprint") - always returns the latter, defaulting an untargeted
    bare list to ``target='random'``."""
    if isinstance(group, dict):
        return {
            'columns': list(group.get('columns', [])),
            'target': group.get('target', 'random'),
        }
    return {'columns': list(group), 'target': 'random'}


def _apply_wrapper(column: str, wrapper_cfg: Optional[dict]) -> str:
    """Render one column, optionally wrapped by a variance/covariance
    structure function (`ASREML.wrappers[column]`, e.g.
    ``{'name': 'ar1'}`` or ``{'name': 'fa', 'args': {'k': 2}}``) - a bare
    column with no configured wrapper renders unchanged (ASReml's own
    implicit `id()` default).

    The column name itself is passed through `r_safe_name()` (see that
    function's own docstring) before being interpolated into the
    returned text - `wrapper_cfg['name']` (the R FUNCTION name, e.g.
    'ar1') is never quoted, only the DATA column it wraps."""
    safe_column = r_safe_name(column)
    if not wrapper_cfg or not wrapper_cfg.get('name'):
        return safe_column
    name = wrapper_cfg['name']
    registry = _wrapper_registry()
    spec = registry.get(name)
    if spec is None:
        # Unknown wrapper name - render it verbatim rather than silently
        # dropping it (a typo'd/hand-edited wrapper name should surface
        # as a visibly wrong R call, caught by the syntactic parse check
        # at commit time - R7 - rather than disappearing without a trace).
        return f"{name}({safe_column})"
    extra_args = wrapper_cfg.get('args', {}) or {}
    ordered_values = [str(extra_args.get(arg_name, '')) for arg_name in spec['args']]
    ordered_values = [v for v in ordered_values if v != '']
    if ordered_values:
        return f"{name}({safe_column}, {', '.join(ordered_values)})"
    return f"{name}({safe_column})"


def _expand_nested(columns: Sequence[str]) -> List[str]:
    """``A/B/C`` nesting expands to ``A + A:B + A:B:C`` - ASReml-R's own
    documented meaning of the `/` nesting operator, applied progressively
    for however many columns the group has (two columns is the common
    case the blueprint's own worked example uses; three or more nests one
    level deeper each time, the natural generalisation)."""
    terms = []
    for depth in range(1, len(columns) + 1):
        terms.append(':'.join(columns[:depth]))
    return terms


def _build_fixed_random_terms(asreml_cfg: dict) -> "Tuple[List[str], List[str]]":
    """Combine ``asreml_cfg``'s plain ``fixed``/``random`` effect lists
    with every ``interaction_groups``/``nested_groups`` entry's own
    expanded term(s) into the two RAW term lists ``render_syntax()``'s
    ``fixed = ~ ...`` / ``random = ~ ...`` sides are ultimately built
    from. Extracted (ver4-24) into its own function so it is expressed
    exactly once and shared by both consumers that need it:

    - ``build_model_spec()``, which de-duplicates the result before
      putting it in the returned ``ModelSpec`` (see that function's own
      ver4-24 comment) - a defensive backstop, not the primary fix;
    - ``validate_structure()``'s ver4-24 duplicate-term check, which
      deliberately inspects these RAW, NOT-yet-de-duplicated lists, so it
      can report a real config mistake to the person instead of silently
      being handed the already-cleaned-up result ``build_model_spec()``
      would otherwise produce.

    Returns ``(fixed_terms, random_terms)``, in the same order this
    module has always assembled them in: the plain effect list first,
    then each ``interaction_groups`` entry's own single ``:``-joined
    term, then each ``nested_groups`` entry's own ``_expand_nested()``
    chain - NEVER de-duplicated here; that is left entirely to the
    caller, since the two callers above need to see different things.
    """
    wrappers_cfg = asreml_cfg.get('wrappers', {}) or {}
    fixed_terms = [_apply_wrapper(c, wrappers_cfg.get(c)) for c in asreml_cfg.get('fixed', [])]
    random_terms = [_apply_wrapper(c, wrappers_cfg.get(c)) for c in asreml_cfg.get('random', [])]

    for raw_group in asreml_cfg.get('interaction_groups', []) or []:
        group = _normalise_group(raw_group)
        cols = [_apply_wrapper(c, wrappers_cfg.get(c)) for c in group['columns']]
        if len(cols) < 2:
            continue
        term = ':'.join(cols)
        (fixed_terms if group['target'] == 'fixed' else random_terms).append(term)

    for raw_group in asreml_cfg.get('nested_groups', []) or []:
        group = _normalise_group(raw_group)
        cols = [_apply_wrapper(c, wrappers_cfg.get(c)) for c in group['columns']]
        if len(cols) < 2:
            continue
        terms = _expand_nested(cols)
        (fixed_terms if group['target'] == 'fixed' else random_terms).extend(terms)

    return fixed_terms, random_terms


def _dedupe_preserving_order(terms: List[str], side: str, *, log_prefix: str = '[ASReml]') -> List[str]:
    """Remove exact repeats from ``terms``, keeping the FIRST occurrence
    of each and otherwise preserving order (never sorts; never keeps a
    later copy instead of an earlier one - relevant only in the
    essentially-impossible case that two differently-configured entries
    happened to render identical text). Logs a ``ver4-24 NOTE`` only when
    a repeat was actually found and removed, so a correctly-configured
    model produces no extra console output at all - see
    ``build_model_spec()``'s own ver4-24 comment for why this exists and
    why it is a backstop, not the primary fix."""
    seen: "set" = set()
    deduped: List[str] = []
    dropped: List[str] = []
    for term in terms:
        if term in seen:
            dropped.append(term)
            continue
        seen.add(term)
        deduped.append(term)
    if dropped:
        print(
            f"{log_prefix} NOTE: the {side} side of this model had duplicate term(s) "
            f"{dropped!r} - removed the repeat(s) so the rendered ASReml-R formula does not "
            f"literally repeat a term. This should have been caught by "
            f"asreml_syntax.validate_structure()'s own check before this config was ever "
            f"committed or run - seeing this note means that check was bypassed "
            f"(ASREML_STRICT=False), or a ModelSpec was built directly without ever calling "
            f"validate_structure() first."
        )
    return deduped


def build_model_spec(asreml_cfg: dict, trait: str, *,
                      response_unit: str = 'genotype',
                      population_col: "Optional[str]" = None) -> ModelSpec:
    """Build a pure ``ModelSpec`` from one task's ``ASREML`` config block
    and its own phenotype/trait name - the single place that config's
    fixed/random/residual/interaction/nested/wrapper fields are combined
    into the term lists ``render_syntax()``/``render_equation()`` render.

    Parameters
    ----------
    asreml_cfg : the resolved ``ASREML`` config dict (see the ver4-12
        config schema delta) - never mutated.
    trait : this task's own phenotype/trait column name. Only used to
        populate ``ModelSpec.response`` for documentation/equation-
        rendering purposes - ``render_syntax()`` itself always emits the
        literal ``{TRAIT}`` placeholder, never this value directly (see
        module docstring).
    response_unit : ver4-19, Requirements.md R3 - ``'genotype'`` (default)
        or ``'genotype_by_population'``. Already-RESOLVED (never
        ``'auto'`` here - see ``Preprocess/asreml_blup.py::
        resolve_response_unit()`` for where ``'auto'`` gets turned into
        one of these two before this function is ever called).
    population_col : required when ``response_unit ==
        'genotype_by_population'`` (ignored otherwise) - see
        ``ModelSpec.population_col``'s own docstring. When given and
        ``response_unit == 'genotype_by_population'``, this column is
        appended to the returned spec's own ``output_level`` (never to
        ``asreml_cfg`` itself, which is never mutated) if not already
        present, since ``predict(classify = ...)`` must group by
        genotype AND population jointly to report one value per cell.

    Raises
    ------
    ValueError if ``asreml_cfg`` has no ``genotype_col`` - every other
        field defaults to "not configured" (empty/False), but a model
        with no genotype term at all is not a genomic-prediction
        adjustment model, and is refused here rather than silently
        rendering a syntax string with no genetic term whatsoever. Also
        raised if ``response_unit == 'genotype_by_population'`` but no
        ``population_col`` was given, or if ``response_unit`` is not one
        of the two recognised values (never silently falls back to
        ``'genotype'`` - a caller that misspells this deserves a loud
        failure, not a silently wrong model).
    """
    genotype_col = asreml_cfg.get('genotype_col')
    if not genotype_col:
        raise ValueError(
            "asreml_syntax.build_model_spec: ASREML['genotype_col'] is required - a mixed "
            "model with no genotype term cannot produce a genomic-prediction-ready "
            "adjusted phenotype."
        )
    if response_unit not in ('genotype', 'genotype_by_population'):
        raise ValueError(
            f"asreml_syntax.build_model_spec: response_unit={response_unit!r} is not "
            f"recognised - must be 'genotype' or 'genotype_by_population' (already resolved, "
            f"never 'auto' - see Preprocess/asreml_blup.py::resolve_response_unit())."
        )
    if response_unit == 'genotype_by_population' and not population_col:
        raise ValueError(
            "asreml_syntax.build_model_spec: response_unit='genotype_by_population' requires "
            "population_col (Requirements.md R3)."
        )
    genotype_is_random = asreml_cfg.get('genotype_effect', 'random') == 'random'
    use_grm = bool(asreml_cfg.get('use_grm', False))

    fixed_terms, random_terms = _build_fixed_random_terms(asreml_cfg)

    # ver4-24 BUGFIX (production symptom: a rendered
    # ``random = ~ vm(ID, source = Ginv) + Year + Experiment + Year +
    # Year:Experiment`` - a literal duplicate ``Year`` term, sitting
    # alongside a standalone ``Experiment`` main effect that has no
    # well-defined meaning next to its own ``Year:Experiment`` nested
    # term). ROOT CAUSE: ASReml-R's ``/`` nesting operator
    # (``_expand_nested()`` above) already expands ``Year/Experiment``
    # to the FULL chain ``Year + Year:Experiment`` - every column in a
    # nested group is therefore already a term in the rendered model on
    # its own account (the outer column literally, via depth-1 of the
    # chain; the inner column via its own ``:``-joined nested term).
    # Separately ALSO selecting one of those columns as a plain
    # ``fixed``/``random`` main effect (e.g. a GUI multiselect ticked
    # *and* a nested group configured for the same columns) therefore
    # never adds a genuinely new term - it either literally repeats one
    # (the outer column), or asserts an independent main effect for a
    # column whose levels are only meaningful nested inside the other
    # (the inner column).
    #
    # ``validate_structure()``'s own new check (below in this module) is
    # the PRIMARY fix - it refuses a config like this before the GUI
    # ever lets it be committed, and before ``GP()``'s preflight lets a
    # headless run past task 0 (under the default ``ASREML_STRICT=
    # True``). This de-duplication is a DEFENSIVE BACKSTOP for the
    # remaining case that check cannot cover: ``ASREML_STRICT=False``
    # downgrading that preflight to a warning, or any caller (e.g.
    # ``Preprocess/asreml_blup.py``'s ``concurrent_variance_model`` spec)
    # that builds a ``ModelSpec`` directly without first running
    # ``validate_structure()``. Order-preserving (first occurrence wins),
    # so a genuinely duplicated term never silently changes which one of
    # two differently-wrapped copies survives.
    fixed_terms = _dedupe_preserving_order(fixed_terms, 'fixed')
    random_terms = _dedupe_preserving_order(random_terms, 'random')

    sparse_terms: Tuple[str, ...] = ()
    if not genotype_is_random:
        sparse_terms = (genotype_col,)

    residual_cfg = asreml_cfg.get('residual')
    residual_str = None
    if residual_cfg:
        dims = residual_cfg.get('dims', [])
        dim_wrappers = residual_cfg.get('wrappers', {}) or {}
        wrapped_dims = [_apply_wrapper(d, dim_wrappers.get(d)) for d in dims]
        if wrapped_dims:
            residual_str = ':'.join(wrapped_dims)

    output_level = list(asreml_cfg.get('output_level', []))
    if response_unit == 'genotype_by_population' and population_col not in output_level:
        # Requirements.md R3 - predict(classify=...) must group by BOTH
        # the genotype and the population column to report one value per
        # (genotype, population) cell; appended here (to the returned
        # spec only - asreml_cfg itself is never mutated, per this
        # function's own contract) rather than requiring every caller to
        # remember it.
        output_level = output_level + [population_col]

    return ModelSpec(
        response=trait,
        fixed=tuple(fixed_terms),
        random=tuple(random_terms),
        residual=residual_str,
        sparse=sparse_terms,
        genotype_col=genotype_col,
        genotype_is_random=genotype_is_random,
        grm=use_grm,
        output_level=tuple(output_level),
        response_unit=response_unit,
        population_col=population_col if response_unit == 'genotype_by_population' else None,
    )


def render_syntax(spec: ModelSpec, *, data_symbol: str = 'dat',
                   grm_symbol: Optional[str] = None) -> str:
    """Render `spec` into a single ASReml-R ``asreml(...)`` call, as
    plain text.

    Parameters
    ----------
    data_symbol : the R-side data-frame symbol to render into
        ``data = <data_symbol>``. Pass the literal string ``'{DATA}'``
        (the default used by the GUI's committed-syntax panel - R7) to
        keep the placeholder in the rendered text for per-task
        substitution by ``Preprocess/asreml_blup.py``; the default
        ``'dat'`` is for a one-off, non-committed preview render only.
    grm_symbol : the R-side inverse-relationship-matrix symbol, used only
        when ``spec.grm`` is True. Pass ``'{GRM}'`` for the same
        placeholder-preserving reason; defaults to the literal example
        name ``'Ginv'`` when not given.

    Returns
    -------
    The full ``asreml(...)`` call text. ``spec.response`` is NEVER
    substituted into the returned text - the trait always renders as the
    literal ``{TRAIT}`` placeholder (see module docstring) so the SAME
    committed text is reusable, unmodified, across every phenotype this
    run predicts.
    """
    parts = []
    fixed_formula = ' + '.join(spec.fixed) if spec.fixed else '1'
    parts.append(f"fixed = {{TRAIT}} ~ {fixed_formula}")

    # ver4-19, Requirements.md R3 - a genotype shared across more than one
    # population needs its OWN adjusted value per population, not one
    # value pooled across every population it appears in (see
    # Preprocess/asreml_blup.py::detect_shared_genotypes_across_
    # populations()`/`resolve_response_unit()` for how this gets
    # resolved, and the EasiGP_ASReml_BLUE_BLUP_Redesign blueprint SS3.3
    # for the three formulas below). The genetic relationship structure
    # itself (Ginv) stays a property of the genotype ALONE - never
    # duplicated per population - only the fixed/random TERM that
    # consumes it changes shape here.
    safe_genotype_col = r_safe_name(spec.genotype_col)
    safe_population_col = r_safe_name(spec.population_col) if spec.population_col else None
    _by_population = spec.response_unit == 'genotype_by_population'
    if _by_population and not safe_population_col:
        raise ValueError(
            "asreml_syntax.render_syntax: spec.response_unit == 'genotype_by_population' but "
            "spec.population_col is not set - build_model_spec() should never produce this "
            "combination; this indicates a ModelSpec constructed by hand rather than through "
            "build_model_spec()."
        )

    if spec.sparse:
        if _by_population and spec.genotype_col in spec.sparse:
            # y ~ Genotype:population + <design> (cell-means BLUE - one
            # per (genotype, population) combination).
            sparse_terms = [
                f"{safe_genotype_col}:{safe_population_col}" if c == spec.genotype_col
                else r_safe_name(c)
                for c in spec.sparse
            ]
            parts.append(f"sparse = ~ {' + '.join(sparse_terms)}")
        else:
            parts.append(f"sparse = ~ {' + '.join(r_safe_name(c) for c in spec.sparse)}")

    random_terms = list(spec.random)
    if spec.genotype_is_random:
        if spec.grm:
            g_sym = grm_symbol if grm_symbol else 'Ginv'
            if _by_population:
                # random = ~ diag(population):vm(Genotype, source=Ginv) -
                # heterogeneous genetic variance per population, ONE
                # shared Ginv (blueprint SS3.3's recommended default;
                # us(population):vm(...) is a documented, not-yet-
                # implemented future enhancement for a genetic
                # CORRELATION between populations).
                genotype_term = f"diag({safe_population_col}):vm({safe_genotype_col}, source = {g_sym})"
            else:
                genotype_term = f"vm({safe_genotype_col}, source = {g_sym})"
        else:
            if _by_population:
                # random = ~ at(population):Genotype - population-
                # specific i.i.d. genotype variance, no GRM.
                genotype_term = f"at({safe_population_col}):{safe_genotype_col}"
            else:
                genotype_term = safe_genotype_col
        random_terms = [genotype_term] + random_terms
    if random_terms:
        parts.append(f"random = ~ {' + '.join(random_terms)}")

    if spec.residual:
        parts.append(f"residual = ~ {spec.residual}")

    parts.append(f"data = {data_symbol}")
    parts.append('na.action = na.method(y = "include", x = "include")')
    parts.append('trace = TRUE')

    joined = ",\n       ".join(parts)
    return f"asreml({joined})"


def render_equation(spec: ModelSpec) -> str:
    """Render `spec` as a short, human-readable equation block for the
    GUI's read-only "Equation" panel (R7) - expanded in plain English
    rather than R syntax, since the syntax panel right below it already
    shows the exact R call.
    """
    lines = ["y = X\u00b7\u03c4 + Z\u00b7u + e"]
    lines.append(f"  y : {spec.response or '<trait>'}, one record per plot")

    _by_population = spec.response_unit == 'genotype_by_population'
    _population_suffix = (
        f" x {spec.population_col} (Requirements.md R3 - one value per genotype PER POPULATION "
        f"it appears in)" if _by_population else ""
    )

    tau_terms = list(spec.fixed)
    if spec.sparse:
        tau_terms = list(spec.sparse) + tau_terms
    tau_desc = ', '.join(['intercept'] + tau_terms) if tau_terms else 'intercept'
    if not spec.genotype_is_random:
        tau_desc += f" (genotype term {spec.genotype_col!r} is FIXED \u21d2 this model returns BLUEs)"
        tau_desc += _population_suffix
    lines.append(f"  \u03c4 : {tau_desc}")

    u_terms = list(spec.random)
    if spec.genotype_is_random:
        genotype_desc = (
            f"{spec.genotype_col} (RANDOM, with a genomic relationship matrix \u21d2 GBLUPs)"
            if spec.grm else
            f"{spec.genotype_col} (RANDOM \u21d2 BLUPs)"
        )
        genotype_desc += _population_suffix
        u_terms = [genotype_desc] + u_terms
    lines.append(f"  u : {', '.join(u_terms) if u_terms else '(no random effects other than genotype)'}")

    residual_desc = f"R = {spec.residual}" if spec.residual else "independent, homogeneous variance (ASReml default)"
    lines.append(f"  e : residual, {residual_desc}")

    return "\n".join(lines)


def model_term_columns(asreml_cfg: dict) -> "set[str]":
    """Every column name this model's own FORMULA actually references -
    fixed effects, random effects, the genotype column itself, every
    interaction/nested group member, and every residual-structure
    dimension. This is the single authority for "is this column actually
    IN the model" (as opposed to merely being a real column in the
    phenotype file, which `available_columns` alone cannot distinguish) -
    the same "one fact, one place" discipline the architecture doc's
    Layer 3d modules (`model_registry.py`, `circos_geometry.py`) already
    apply elsewhere, extended here to this module per its own "zero
    project imports" / single-authority mandate.

    Used two places that must never independently drift on this: (1)
    `validate()` below, checking a candidate `output_level` BEFORE any
    syntax is ever rendered or committed; (2) `Preprocess/asreml_blup.py::
    _collect_factor_and_order_columns()`, which reads this same set
    (rather than re-deriving it) to decide which columns need R-side
    `factor()` coercion, AND `asreml_adjust_phenotypes()`'s own run-time
    guard immediately before `predict(classify=...)` is ever called.

    NOTE: a column named only in `output_level` is NOT a model term by
    virtue of that alone - `predict(classify=...)` can only classify by
    columns ASReml actually fitted a term against (fixed, random, sparse/
    genotype, or a residual dimension), never by an arbitrary grouping
    column that happens to exist in the data but was never given to the
    model. Requesting one anyway is exactly what produced the R-side
    "<column> is not in the model, unable to predict using it" error this
    check exists to catch before it ever reaches R.
    """
    genotype_col = asreml_cfg.get('genotype_col')
    cols = set(asreml_cfg.get('fixed', []) or []) | set(asreml_cfg.get('random', []) or [])
    if genotype_col:
        cols.add(genotype_col)
    for raw_group in (asreml_cfg.get('interaction_groups', []) or []) + (asreml_cfg.get('nested_groups', []) or []):
        cols.update(_normalise_group(raw_group)['columns'])
    residual_cfg = asreml_cfg.get('residual')
    if residual_cfg:
        cols.update(residual_cfg.get('dims', []) or [])
    # ver4-19, Requirements.md R3: 'population' is EasiGP's own universal,
    # always-present join-key column (architecture doc SS4.1) - Step 6b's
    # new unmasked engine (Preprocess/asreml_blup.py::
    # fit_unmasked_adjustment()) may classify by it (via the genotype:
    # population / at(population):Genotype / diag(population):vm(...)
    # terms `render_syntax()` builds for `response_unit ==
    # 'genotype_by_population'`) regardless of whether the person also
    # separately added it as an ordinary fixed/random effect. Always
    # including it here - rather than threading `response_unit` through
    # this and every caller just to conditionally add it - means
    # `output_level=[genotype_col, 'population']` never trips the "stray
    # output-level column" check in `validate_structure()` below, whether
    # or not response_unit ends up resolving to 'genotype_by_population'
    # for this particular run.
    cols.add('population')
    return cols


def residual_requires_full_grid(asreml_cfg: "Optional[dict]") -> bool:
    """ver4-17 - single authority for "does this task's residual structure
    need ASReml to see one row per COMBINATION of its own dimensions'
    levels" - i.e. a complete factorial design over ``ASREML['residual']
    ['dims']`` (an ``ar1(Range):ar1(Row)``-style spatial structure being
    the motivating case, but the same requirement applies to any
    correlated/ordered residual structure with one or more dimensions,
    not only a two-dimensional AR1 one).

    Reused by BOTH sides of the Python<->R boundary that need to agree on
    this exact same fact (the ``_map_genes_to_markers`` precedent this
    codebase applies to every such cross-cutting question - architecture
    doc SS2, "Layer 3d"):

      - ``genomic_prediction.py``'s Step 6b, to decide whether this
        task's own raw phenotype records must be the FULL, unrestricted
        table (every population/individual the trial ever recorded) or
        may be safely restricted to just this task's own individuals
        (the pre-ver4-17, and still-default-for-non-grid-residuals,
        behaviour);
      - ``Preprocess/asreml_blup.py::_fit_single_asreml_model()``, which
        makes the identical decision from its own copy of ``asreml_cfg``
        as a defensive backstop, independent of whichever table the
        caller actually passed in (this module's own "never trust,
        always verify" discipline, mirrored from RK-2/RK-9's existing
        precedent elsewhere in this ASReml integration).

    A plain ``bool(dims)`` rather than inspecting the wrapper NAMES
    (``ar1``, ``ar2``, ``sar``, ...) deliberately - ANY residual
    structure that names one or more ordered dimensions implies a
    correlation (or at minimum a deliberate ORDERING - see
    ``Preprocess/asreml_blup.py::_collect_factor_and_order_columns()``'s
    own ``order_cols``, which this reuses the identical source field
    for) across those dimensions' levels, and this codebase's own
    Appendix B caveat (asreml_fit.R's top-of-file note: reviewed against
    ASReml-R's documented API only, never executed against a licensed
    install here) means guessing which wrapper names specifically need a
    complete grid, and which do not, is exactly the kind of narrow,
    unverifiable assumption this project's "name it rather than assume"
    standard avoids. Treating every non-empty ``dims`` list as needing
    the full design is the conservative direction: it costs a larger fit
    for a residual structure that turns out not to have actually needed
    one, never a mismatched-size R error for one that did.
    """
    if not asreml_cfg:
        return False
    residual_cfg = asreml_cfg.get('residual')
    if not residual_cfg:
        return False
    return bool(residual_cfg.get('dims', []) or [])


def pad_residual_grid(records: "pd.DataFrame", asreml_cfg: "Optional[dict]") -> "pd.DataFrame":
    """ver4-18 - single authority for COMPLETING a grid-structured
    residual's own factorial design (module docstring, "ver4-18
    BUGFIX" - the fix ``residual_requires_full_grid()`` above only set
    up the DECISION for, not the mechanics of).

    Explicitly builds the Cartesian product of ``ASREML['residual']
    ['dims']``'s own OBSERVED levels (e.g. every distinct ``Range``
    value crossed with every distinct ``Row`` value actually present
    anywhere in ``records`` - never a synthetic min..max integer run: a
    level that never appears anywhere in the file is not a real trial
    position, and inventing one would let a typo'd or sparsely-labelled
    dimension silently balloon the grid) and adds exactly one new row
    for every combination `records` does not already have at least one
    row for. Every added row's residual-dimension columns hold that
    combination's own values; EVERY OTHER COLUMN - the response, the
    genotype column, every other fixed/random/interaction-group column -
    is ``NaN``. This is deliberately safe under this integration's own
    unconditionally-appended ``na.action = na.method(y = "include", x =
    "include")`` (``_build_residual_side()``/``render_syntax()`` below):
    ``y = "include"`` already means a missing response contributes
    nothing to the likelihood (identical to how a masked validation/test
    row already behaves), and ``x = "include"`` means the same for a
    missing factor/covariate value in any fixed or random term - so a
    row that exists ONLY to complete the residual grid, and carries no
    other information at all, is fit exactly as inertly as intended.

    Reused by BOTH sides of the Python<->R boundary, exactly the
    `residual_requires_full_grid()` precedent just above:

      - ``genomic_prediction.py``, ONCE per run (not once per task - the
        padding depends only on the phenotype file's own dimension
        columns, never on any one task's own individuals), immediately
        after ``data_phenotype_raw_for_asreml`` is finalised;
      - ``Preprocess/asreml_blup.py::_fit_single_asreml_model()``, as a
        defensive backstop over whichever table its own caller actually
        passed in - this codebase's usual "never trust, always verify"
        discipline (RK-2/RK-9's own precedent) applied to this exact
        question one more time.

    IDEMPOTENT and side-effect-free on an already-complete grid: finding
    zero missing combinations returns `records` completely unchanged (no
    copy, no reindex), which is what makes calling this twice - once up
    front, once as a defensive backstop - free rather than wasteful.

    A no-op, returning `records` unchanged, whenever
    `residual_requires_full_grid(asreml_cfg)` is `False`, whenever fewer
    than two of the configured dimensions are actually present in
    `records` (a single ordered dimension's own implied size is exactly
    its own observed level count - it cannot itself be short a
    combination, so there is nothing to pad; a configured-but-absent
    dimension column is a `validate()`/`validate_structure()` problem
    this function does not attempt to second-guess), or whenever
    `records` has zero rows (nothing to derive observed levels from).
    """
    if not residual_requires_full_grid(asreml_cfg):
        return records
    dims = [d for d in (asreml_cfg.get('residual') or {}).get('dims', []) or []
            if d in records.columns]
    if len(dims) < 2 or records.shape[0] == 0:
        return records

    import itertools
    import numpy as np
    import pandas as pd

    level_values = [sorted(records[d].dropna().unique().tolist(), key=str) for d in dims]
    if any(len(lv) == 0 for lv in level_values):
        return records

    all_combos = set(itertools.product(*level_values))
    existing_combos = set(map(tuple, records[dims].values.tolist()))
    missing_combos = list(all_combos - existing_combos)
    if not missing_combos:
        return records

    pad_rows = pd.DataFrame({
        dim: [combo[i] for combo in missing_combos] for i, dim in enumerate(dims)
    })
    for col in records.columns:
        if col not in dims:
            pad_rows[col] = np.nan
    pad_rows = pad_rows[list(records.columns)]
    # Match each dimension column's ORIGINAL dtype, so a numeric
    # Range/Row does not become an object-dtype mix of int and float
    # that asreml_fit.R's own `factor()`/`order()` calls could sort or
    # group differently from the surviving original rows.
    for dim in dims:
        try:
            pad_rows[dim] = pad_rows[dim].astype(records[dim].dtype)
        except (TypeError, ValueError):
            pass

    padded = pd.concat([records, pad_rows], axis=0, ignore_index=True)
    n_total = 1
    for lv in level_values:
        n_total *= len(lv)
    print(f"[ASReml] residual grid completion added {len(missing_combos)} filler "
          f"row(s) (NaN response, NaN every other column) to complete the "
          f"{' x '.join(dims)} factorial the residual structure implies "
          f"({n_total} total combination(s) required, {len(existing_combos)} already present "
          f"in the raw phenotype records).")
    return padded


def estimand(asreml_cfg: dict) -> str:
    """'BLUE' | 'BLUP' | 'GBLUP' - see module docstring for the exact
    rule. Reads only `genotype_effect`/`use_grm`, so this can be called
    before a full ModelSpec is built (e.g. for the GUI's estimand banner,
    which must update live as the genotype-effect radio changes, without
    needing every other field to be valid yet).

    NOTE: for `genotype_effect='fixed'`, this names what Fit 1 (training
    individuals) alone produces - it is NOT the estimand of the whole
    result once any validation/test individual exists (see module
    docstring, "Design decision: BLUE cannot be held out"). Anywhere a
    person-facing summary needs to describe the WHOLE result honestly,
    use `estimand_description()` below instead - this function is kept
    as-is because `render_equation()`'s own per-fit wording (`build_
    model_spec()` describes exactly ONE model at a time) still needs it.
    """
    if asreml_cfg.get('genotype_effect', 'random') == 'fixed':
        return 'BLUE'
    return 'GBLUP' if asreml_cfg.get('use_grm', False) else 'BLUP'


def estimand_description(asreml_cfg: dict) -> str:
    """One-line, person-facing summary of what the 'value' column of
    `asreml_pheno`/`ASReml_predicted_phenotypes.csv` ACTUALLY holds,
    split by split - written specifically so nobody reads
    `estimand(asreml_cfg) == 'BLUE'` and assumes every row is a BLUE (see
    module docstring, "Design decision: BLUE cannot be held out" - a
    production run's own confusing failure is exactly what this function
    exists to prevent from recurring in the OUTPUT's own documentation,
    not just in an error message). Used by the GUI's live Estimand
    banner and `genomic_prediction.py`'s own "[GP] ASREML resolved: ..."
    preflight line, so this is visible before a run ever starts, not
    just after reading `Interaction`-style output files.

    Returns
    -------
    `'BLUP'` or `'GBLUP'` unchanged, when genotype is random (patterns 3-4
    of the module docstring's own four-pattern table - a single fit, one
    estimand, `estimand()` above already says the whole truth). When
    genotype is fixed AND `ASREML['fixed_holdout_mode']=='single_fit'`
    (Requirements.md item 2's "original version of BLUEs" pattern - see
    `ASREML_PATTERNS['blue_original']` below) - a single, genuinely
    uniform `'BLUE'`, since every row (train AND validation/test) comes
    from the SAME one fit. Otherwise (the default `'two_fit'` mode) a
    two-clause string naming BOTH estimands and BOTH split groups -
    training rows get a real BLUE, validation/test rows get BLUP or
    GBLUP depending on `use_grm`, from two INDEPENDENT fits (see
    `Preprocess/asreml_blup.py::_fit_blue_train_plus_holdout()`).
    """
    if asreml_cfg.get('genotype_effect', 'random') != 'fixed':
        return estimand(asreml_cfg)
    if asreml_cfg.get('fixed_holdout_mode', 'two_fit') == 'single_fit':
        return ('BLUE (single fit - validation/test individuals are predicted from the same '
                'fit, using only the information already present in the training partition)')
    holdout_estimand = 'GBLUP' if asreml_cfg.get('use_grm', False) else 'BLUP'
    return f"BLUE (training individuals) + {holdout_estimand} (validation/test individuals)"


def holdout_model_cfg(asreml_cfg: dict) -> dict:
    """The ``ASREML`` config for Fit 2 ("BLUP/GBLUP, holdout" - module
    docstring, "Design decision: BLUE cannot be held out") - identical to
    `asreml_cfg` in every respect (fixed/random effects, residual
    structure, output level, interaction/nested groups, wrappers, GRM
    settings) EXCEPT the genotype term, forced to `'random'` here so it
    can be estimated for a masked (unobserved) validation/test
    individual at all - a FIXED term categorically cannot be (see that
    same docstring section for why).

    A no-op (returns a shallow copy of `asreml_cfg` unchanged) when
    `genotype_effect` is already `'random'` - the caller never needs to
    branch on this itself before calling `build_model_spec()`/
    `render_syntax()` against the result.
    """
    if asreml_cfg.get('genotype_effect', 'random') != 'fixed':
        return dict(asreml_cfg)
    shadow = dict(asreml_cfg)
    shadow['genotype_effect'] = 'random'
    return shadow


def suggest_holdout_syntax(asreml_cfg: dict) -> str:
    """Auto-suggested STARTING POINT for
    ``ASREML['holdout_committed_syntax']`` (Fit 2's own committed text -
    see `holdout_model_cfg()` and the module docstring's "Design
    decision" section) - the same shape `render_syntax(build_model_spec(
    ...))` already produces for the GUI's primary syntax preview, over
    `holdout_model_cfg(asreml_cfg)` instead of `asreml_cfg` itself.
    Placeholders only (``{TRAIT}``/``{DATA}``/``{GRM}``), never a
    concrete trait - identical convention to `render_syntax()` itself.

    Purely a GUI convenience to seed an editable text area: per R7's own
    "generate-then-edit-then-use" contract, the person is free to hand-
    edit the result before committing it, exactly as for the primary
    syntax. Never called at run time - only `holdout_committed_syntax`
    (whatever text was actually committed) is ever fitted.
    """
    shadow_cfg = holdout_model_cfg(asreml_cfg)
    spec = build_model_spec(shadow_cfg, trait='{TRAIT}')
    return render_syntax(spec, data_symbol='{DATA}', grm_symbol='{GRM}' if shadow_cfg.get('use_grm') else None)


# ---------------------------------------------------------------------------
# Requirements.md item 2 - restructuring the BLUE/BLUP x GRM combination by
# prediction SCENARIO. Single authority (same discipline as
# `model_registry.py`/`circos_geometry.py` - architecture doc S2, "Layer 3d
# - the shared authorities") for "which of the four-now-five configuration
# patterns is this ASREML config", and "which pattern(s) is a given
# SCENARIO/genotype-sharing answer allowed to run" - read by BOTH
# `main_app.py` (so the GUI only ever OFFERS a legal combination) and
# `genomic_prediction.py::GP()`'s own preflight (so a hand-written/headless
# config that bypassed the GUI gets the exact same protection).
#
# WHY THIS EXISTS (Requirements.md item 2, and the root cause behind item 1)
# --------------------------------------------------------------------------
# A random-effect genotype term (BLUP) - with or without a GRM - has a
# well-defined value for a genotype level with NO observed data, because
# its prior is the term's own estimated variance component: with NO
# genomic relationship matrix, that value is the population mean for
# EVERY such unobserved level (nothing distinguishes one unobserved
# genotype from another); WITH a GRM, it instead borrows information from
# every OTHER genotype in the fit through the relationship matrix, so
# different unobserved individuals get different, informative predictions
# (this is the entire reason to use a GRM at all).
#
# Two things decide whether a validation/test individual's genotype counts
# as "observed" or "unobserved" for this purpose:
#   - SCENARIO='within' splits individuals at RANDOM within one
#     population - by construction, one individual/one genotype level
#     lands in exactly one of train/valid/test (I4's own 1:1 ID<->
#     individual invariant), so a held-out genotype is NEVER also present
#     in the training partition. A GRM is therefore REQUIRED for a
#     non-constant validation/test result.
#   - SCENARIO='between' trains on one population and predicts a
#     DIFFERENT one. Whether a held-out individual's own genotype is ALSO
#     observed in the training population depends entirely on the
#     experiment this data comes from (e.g. the same breeding lines
#     trialled at two sites - genotypes ARE shared - versus two
#     genuinely disjoint panels - genotypes are NOT shared) - EasiGP
#     cannot infer this from the data alone, so it asks.
#
# Every pattern below that does NOT require a GRM (`blup_no_grm`,
# `blue_blup_no_grm`, and the single-fit `blue_original`) is therefore only
# ever legal when genotypes ARE known to be shared between the training and
# holdout partitions - i.e. `scenario='between'` AND the person has
# answered "yes" to the sharing question. Every other case - `within`, or
# `between` with sharing answered "no" or not yet answered at all (the
# SAFER default) - only ever offers the two GRM-based patterns.
# ---------------------------------------------------------------------------
ASREML_PATTERNS: Dict[str, dict] = {
    'blup_no_grm': {
        'label': 'BLUPs, no GRM (genotype random)',
        'genotype_effect': 'random', 'use_grm': False, 'fixed_holdout_mode': None,
    },
    'gblup': {
        'label': 'BLUPs with a GRM - GBLUP (genotype random)',
        'genotype_effect': 'random', 'use_grm': True, 'fixed_holdout_mode': None,
    },
    'blue_blup_no_grm': {
        'label': 'BLUEs (training) + BLUPs (validation/test), no GRM',
        'genotype_effect': 'fixed', 'use_grm': False, 'fixed_holdout_mode': 'two_fit',
    },
    'blue_gblup': {
        'label': 'BLUEs (training) + BLUPs with a GRM - GBLUP (validation/test)',
        'genotype_effect': 'fixed', 'use_grm': True, 'fixed_holdout_mode': 'two_fit',
    },
    'blue_original': {
        'label': ('BLUEs only - original version (one fit; validation/test individuals are '
                  'predicted from the training partition alone, no genomic relationship matrix)'),
        'genotype_effect': 'fixed', 'use_grm': False, 'fixed_holdout_mode': 'single_fit',
    },
}

# Patterns legal whenever the training and holdout partitions are NOT known
# to share genotypes (SCENARIO='within'; or SCENARIO='between' with the
# sharing question answered 'no' or left unanswered) - the two GRM-based
# patterns only, per this section's own docstring above.
_ASREML_GRM_ONLY_PATTERNS: Tuple[str, ...] = ('gblup', 'blue_gblup')


def allowed_pattern_ids(scenario: str, between_shared: "Optional[bool]" = None) -> "list[str]":
    """The `ASREML_PATTERNS` key(s) legal for this prediction `scenario`
    (`'within'`/`'between'`) - Requirements.md item 2. `between_shared` is
    only consulted when `scenario == 'between'`; it is ignored (as if
    `False`) for `'within'`, since a within-population random split can
    never share a genotype level across partitions at all.

    `between_shared=None` (the question hasn't been answered yet, or
    `ASREML['between_shared_genotypes']` was never set) deliberately
    behaves exactly like `False` - the SAFER, GRM-required default -
    never like `True`, so an unanswered question can never silently
    unlock a pattern this module's own docstring says can produce a
    constant validation/test result.
    """
    if scenario == 'between' and between_shared is True:
        return list(ASREML_PATTERNS.keys())
    return list(_ASREML_GRM_ONLY_PATTERNS)


def pattern_id_for_config(asreml_cfg: "Optional[dict]") -> "Optional[str]":
    """Which `ASREML_PATTERNS` key `asreml_cfg` currently represents, or
    `None` for `asreml_cfg=None`/a combination that matches none of them
    (e.g. a hand-edited config with `fixed_holdout_mode` set while
    `genotype_effect == 'random'`) - the single place that reverse
    mapping is expressed, used by the GUI (to pre-select/validate the
    right pattern) and by `validate_scenario_compatibility()` below.
    """
    if not asreml_cfg:
        return None
    genotype_effect = asreml_cfg.get('genotype_effect', 'random')
    use_grm = bool(asreml_cfg.get('use_grm', False))
    fixed_holdout_mode = asreml_cfg.get('fixed_holdout_mode', 'two_fit') if genotype_effect == 'fixed' else None
    for pattern_id, spec in ASREML_PATTERNS.items():
        if (spec['genotype_effect'] == genotype_effect and spec['use_grm'] == use_grm
                and spec['fixed_holdout_mode'] == fixed_holdout_mode):
            return pattern_id
    return None


def validate_scenario_compatibility(asreml_cfg: "Optional[dict]", scenario: str) -> "list[str]":
    """Requirements.md item 2's own enforcement - a `list[str]` of human-
    readable errors (empty list = compatible), checked ALONGSIDE (not
    instead of) `validate_structure()`/`validate()` above, by BOTH the
    GUI's commit-time validation and `genomic_prediction.py::GP()`'s own
    preflight (Layer 2, run before task 0 - see that function's own
    ASREML preflight block). Returns `[]` immediately when ASReml is
    disabled (`asreml_cfg` falsy) - nothing to check.

    Reads `ASREML['between_shared_genotypes']` (`True`/`False`/absent)
    directly from `asreml_cfg` itself, rather than taking it as a
    separate parameter, so every caller (GUI, headless preflight) checks
    against the SAME resolved value the config actually carries at run
    time - never a value the caller reconstructed independently.

    ver4-20 follow-up: for a NEW-engine config (`legacy_masked_holdout`
    is falsy - the only kind the GUI can build any more), this function
    instead checks the (`genotype_effect_strategy`, `mask_valid_test`,
    `scenario`) combination - see `_new_engine_scenario_errors()` below.
    Both branches share this one function/docstring/call-site so a
    caller never has to know which engine a config represents before
    asking whether it is scenario-legal.
    """
    if not asreml_cfg:
        return []
    if not asreml_cfg.get('legacy_masked_holdout', False):
        return _new_engine_scenario_errors(asreml_cfg, scenario)
    pattern_id = pattern_id_for_config(asreml_cfg)
    between_shared = asreml_cfg.get('between_shared_genotypes')
    allowed = allowed_pattern_ids(scenario, between_shared)
    if pattern_id is None:
        return [
            "ASREML's genotype_effect / use_grm / fixed_holdout_mode combination does not match "
            "any known configuration pattern (see asreml_syntax.ASREML_PATTERNS) - this usually "
            "means fixed_holdout_mode was set while genotype_effect == 'random', where it has "
            "no meaning."
        ]
    if pattern_id in allowed:
        return []
    allowed_labels = [ASREML_PATTERNS[p]['label'] for p in allowed]
    if scenario != 'between':
        reason = (
            f"SCENARIO={scenario!r} splits individuals at random within one population, so a "
            f"validation/test individual's genotype is never also observed in the training "
            f"partition - without a genomic relationship matrix that individual's adjusted "
            f"phenotype has no information to shrink from beyond the population mean, and EVERY "
            f"held-out individual collapses to that same constant value"
        )
    elif between_shared:
        reason = "this should not happen - every pattern is allowed once sharing is confirmed"
    else:
        reason = (
            "SCENARIO='between' with the training/validation/test partitions NOT confirmed to "
            "share genotypes (ASREML['between_shared_genotypes'] is False or unset) behaves "
            "exactly like 'within' for this purpose - a genomic relationship matrix is required"
        )
    return [
        f"ASREML pattern {ASREML_PATTERNS[pattern_id]['label']!r} is not allowed for this "
        f"configuration ({reason} - Requirements.md item 2). Allowed pattern(s): "
        f"{allowed_labels}."
    ]


def _new_engine_scenario_errors(asreml_cfg: dict, scenario: str) -> "list[str]":
    """ver4-20 follow-up - the NEW (default) engine's own scenario-
    legality check, mirroring the reasoning `validate_scenario_
    compatibility()`'s legacy branch already applies to the old five-
    pattern vocabulary, restated for the new engine's simpler
    (`genotype_effect_strategy`, `mask_valid_test`) pair. Defence-in-
    depth only: the GUI's own widget visibility rules (main_app.py's
    ASReml tab - "Genotype effect / genomic relationship matrix
    configuration" subsection) already make every combination this
    function refuses unreachable from a fresh interaction; this exists
    for a hand-written/headless config, or a saved GUI-state file
    predating this fix.

    `ASREML['mask_valid_test']=False` (the recommended default for both
    prediction scenarios) is unconditionally legal for every strategy -
    nothing is held out, so there is no "collapses to a constant"
    failure mode to police at all. Masking only becomes legality-
    sensitive because it reopens exactly that failure mode for a
    genotype term with nothing but the population mean to shrink an
    unobserved individual towards.
    """
    if not resolve_mask_valid_test(asreml_cfg):
        return []
    strategy = resolve_genotype_effect_strategy(asreml_cfg)
    if strategy == 'blue_fixed':
        if scenario != 'between':
            return [
                "ASREML['mask_valid_test']=True with genotype_effect_strategy='blue_fixed' "
                "(BLUEs) requires SCENARIO='between' - a FIXED genotype term cannot be "
                "estimated for an individual whose every record is masked, UNLESS that same "
                "genotype also has a real, unmasked observation elsewhere, which is only "
                "possible when predicting BETWEEN populations. Under SCENARIO='within', every "
                "masked validation/test individual's adjusted phenotype would be entirely "
                "inestimable (NaN). Disable masking, or choose a BLUPs strategy instead."
            ]
        return []
    # blup_iid / blup_grm
    if strategy == 'blup_iid' and scenario != 'between':
        return [
            "ASREML['mask_valid_test']=True with genotype_effect_strategy='blup_iid' (BLUPs "
            "without a genomic relationship matrix) under SCENARIO='within' would give every "
            "masked validation/test individual the SAME (population-mean) adjusted phenotype "
            "- an independent random genotype term has nothing but the population mean to "
            "shrink an individual with no observed record towards. Set "
            "ASREML['use_grm']=True (genotype_effect_strategy='blup_grm'), or disable masking."
        ]
    return []


# ---------------------------------------------------------------------------
# ver4-19 (Requirements.md R1-R3, EasiGP_ASReml_BLUE_BLUP_Redesign blueprint)
# --------------------------------------------------------------------------
# Everything ABOVE this point (ASREML_PATTERNS, allowed_pattern_ids(),
# validate_scenario_compatibility(), the whole "Design decision: BLUE cannot
# be held out" masked/two-fit machinery) is the LEGACY engine - unchanged,
# and still reachable (bit-for-bit) via `ASREML['legacy_masked_holdout'] =
# True`. It exists to let a validation/test individual's masked response be
# PREDICTED from a held-out-safe fit, which is exactly why it needs a
# genomic relationship matrix (or a shared-genotype confirmation) to avoid
# Requirements.md R1's "every held-out individual collapses to the same
# constant" failure mode in the first place.
#
# The new DEFAULT engine (`Preprocess/asreml_blup.py::
# fit_unmasked_adjustment()`) sidesteps that whole problem structurally:
# nothing is ever masked, so every individual's own adjusted value comes
# from ITS OWN observed record(s), not a relationship-based prediction of a
# hidden one (BLUEs_vs_BLUPs.md, "the key realisation is that computing a
# test individual's evaluation target from its own observations is not
# leakage"). None of `ASREML_PATTERNS`/`allowed_pattern_ids()`/
# `validate_scenario_compatibility()` apply to it - there is no "held-out"
# partition to legislate over - which is why this new engine has its own,
# much simpler three-way vocabulary instead of the five-pattern one above.
# ---------------------------------------------------------------------------
GENOTYPE_EFFECT_STRATEGIES: Tuple[str, ...] = ('blue_fixed', 'blup_iid', 'blup_grm')
DEFAULT_GENOTYPE_EFFECT_STRATEGY = 'blue_fixed'

# Maps each of the three new strategies onto the SAME
# genotype_effect/use_grm pair the rest of this module (build_model_spec(),
# render_syntax(), estimand()) already understands - deliberately reusing
# those fields rather than inventing a parallel set, so every existing
# rendering/estimand function keeps working unchanged for the new engine
# too; only the MASKING/two-fit decision (legacy-only) does not apply.
_GENOTYPE_EFFECT_STRATEGY_ENGINE_PARAMS: Dict[str, Tuple[str, bool]] = {
    'blue_fixed': ('fixed', False),
    'blup_iid': ('random', False),
    'blup_grm': ('random', True),
}


def genotype_effect_strategy_engine_params(strategy: str) -> Tuple[str, bool]:
    """`(genotype_effect, use_grm)` for one of `GENOTYPE_EFFECT_STRATEGIES`
    - the single place that mapping is expressed (read by
    `resolve_genotype_effect_strategy()` below, and by
    `Preprocess/asreml_blup.py`'s own callers that need to hand a plain
    `asreml_cfg`-shaped `genotype_effect`/`use_grm` pair to
    `build_model_spec()`).

    Raises
    ------
    ValueError for an unrecognised `strategy` - never silently defaults,
    per this module's "name it rather than assume" standard.
    """
    if strategy not in _GENOTYPE_EFFECT_STRATEGY_ENGINE_PARAMS:
        raise ValueError(
            f"asreml_syntax.genotype_effect_strategy_engine_params: {strategy!r} is not one of "
            f"{GENOTYPE_EFFECT_STRATEGIES}."
        )
    return _GENOTYPE_EFFECT_STRATEGY_ENGINE_PARAMS[strategy]


def resolve_genotype_effect_strategy(asreml_cfg: "Optional[dict]") -> str:
    """Which of `GENOTYPE_EFFECT_STRATEGIES` this config represents -
    Requirements.md R2's default-resolution/backward-compatibility
    authority, read by BOTH `main_app.py` (so the GUI's own live preview
    and the headless engine can never disagree about which strategy a
    config represents) and `genomic_prediction.py::GP()`'s own
    preflight/dispatch.

    Resolution order:
      1. `asreml_cfg['genotype_effect_strategy']`, if it is already one of
         `GENOTYPE_EFFECT_STRATEGIES` explicitly (the normal case for any
         config built by the new, ver4-19 GUI flow, or a headless config
         written directly against this delivery).
      2. Otherwise, derived from the EXISTING `genotype_effect`/`use_grm`
         pair, if present (a hand-written config, or one built under
         `legacy_masked_holdout=True`'s own pattern picker) - this is
         what lets an old config keep resolving to a sensible strategy
         name for logging/GUI-caption purposes without needing to be
         rewritten.
      3. `DEFAULT_GENOTYPE_EFFECT_STRATEGY` ('blue_fixed') - Requirements.md
         R2's own literal default - when neither of the above is present
         at all (e.g. `asreml_cfg == {}`).

    Never raises - always returns one of `GENOTYPE_EFFECT_STRATEGIES`.
    """
    if not asreml_cfg:
        return DEFAULT_GENOTYPE_EFFECT_STRATEGY
    explicit = asreml_cfg.get('genotype_effect_strategy')
    if explicit in GENOTYPE_EFFECT_STRATEGIES:
        return explicit
    if 'genotype_effect' in asreml_cfg or 'use_grm' in asreml_cfg:
        genotype_effect = asreml_cfg.get('genotype_effect', 'random')
        use_grm = bool(asreml_cfg.get('use_grm', False))
        if genotype_effect == 'fixed':
            return 'blue_fixed'
        return 'blup_grm' if use_grm else 'blup_iid'
    return DEFAULT_GENOTYPE_EFFECT_STRATEGY


# ---------------------------------------------------------------------------
# MULTI-OPTION SUPPORT (Requirements.md items 1-3, multi-option update).
#
# "Genotype effect" is no longer a two-way radio: a run may now select ANY
# non-empty subset of the three strategies (BLUEs / BLUPs without GRM /
# BLUPs with GRM), and EVERY selected option is run and reported for the
# SAME iterations. The config carries:
#
#   ASREML['genotype_effect_strategies'] : ordered list, subset of
#       GENOTYPE_EFFECT_STRATEGIES, always in that canonical order.
#   ASREML['strategy_settings']          : {strategy: {option-specific
#       overrides}} - only the keys in STRATEGY_SETTING_KEYS.
#   ASREML['committed_syntaxes']         : {strategy: committed text}
#   ASREML['syntax_fingerprints']        : {strategy: fingerprint}
#
# Everything else (genotype_col, fixed/random effects, residual, wrappers,
# workspace, response_unit, use_entire_dataset, accuracy_metrics, ...) is
# SHARED by all options. `strategy_config()` below is the single place
# that turns this multi-option config into the exact single-strategy dict
# the (unchanged) engine already consumes - so a config WITHOUT
# `genotype_effect_strategies` resolves to exactly itself (I11).
# ---------------------------------------------------------------------------
STRATEGY_COLUMN = 'asreml_option'

STRATEGY_LABELS: Dict[str, str] = {
    'blue_fixed': 'BLUEs',
    'blup_iid': 'BLUPs without GRM',
    'blup_grm': 'BLUPs with GRM',
}
# Filesystem/plot-file-safe tags (no spaces) - used for sub-folder names
# and figure file names.
STRATEGY_FILE_TAGS: Dict[str, str] = {
    'blue_fixed': 'BLUEs',
    'blup_iid': 'BLUPs_without_GRM',
    'blup_grm': 'BLUPs_with_GRM',
}

# The ONLY keys a per-option `strategy_settings[strategy]` dict may
# override on the shared config.
STRATEGY_SETTING_KEYS: Tuple[str, ...] = (
    'mask_valid_test', 'per_set_grm', 'grm_source', 'grm_blend',
    'blue_refinement', 'blue_weighting',
)


def strategy_label(strategy: str) -> str:
    """Human-readable label ('BLUEs', 'BLUPs without GRM', 'BLUPs with
    GRM') - falls back to the raw key for an unknown value (never
    raises), so a column value written by a future version still
    renders."""
    return STRATEGY_LABELS.get(strategy, str(strategy))


def resolve_strategy_list(asreml_cfg: "Optional[dict]") -> "List[str]":
    """The ordered list of strategies this config asks for.

      * `[]` when ASReml is off (`asreml_cfg` falsy).
      * `ASREML['genotype_effect_strategies']` (de-duplicated, put in the
        canonical `GENOTYPE_EFFECT_STRATEGIES` order so every consumer -
        GUI, engine, reporting - iterates options identically) when
        present and non-empty.
      * otherwise the single strategy `resolve_genotype_effect_strategy()`
        derives - i.e. every config that predates multi-option support.

    Raises ValueError for an unrecognised entry or an EMPTY explicit list
    (Requirements.md item 1: "If no field is chosen, it causes an error
    before running EasiGP") - never silently defaults."""
    if not asreml_cfg:
        return []
    if 'genotype_effect_strategies' in asreml_cfg:
        raw = asreml_cfg.get('genotype_effect_strategies')
        if raw is None or len(list(raw)) == 0:
            raise ValueError(
                "ASREML['genotype_effect_strategies'] is empty - select at least one of "
                "BLUEs, BLUPs without GRM, BLUPs with GRM."
            )
        raw = list(raw)
        unknown = [r for r in raw if r not in GENOTYPE_EFFECT_STRATEGIES]
        if unknown:
            raise ValueError(
                f"ASREML['genotype_effect_strategies'] contains unrecognised value(s) "
                f"{unknown!r}; allowed: {list(GENOTYPE_EFFECT_STRATEGIES)}."
            )
        return [g for g in GENOTYPE_EFFECT_STRATEGIES if g in set(raw)]
    return [resolve_genotype_effect_strategy(asreml_cfg)]


def is_multi_strategy(asreml_cfg: "Optional[dict]") -> bool:
    """True when this run must execute MORE THAN ONE option (and hence
    every result file carries `STRATEGY_COLUMN`)."""
    try:
        return len(resolve_strategy_list(asreml_cfg)) > 1
    except ValueError:
        return False


def strategy_config(asreml_cfg: "Optional[dict]", strategy: "Optional[str]" = None) -> "Optional[dict]":
    """The single-strategy ASREML dict for `strategy` (default: the first
    resolved one) - exactly what `genomic_prediction.py::_GP_single()`
    (the unchanged engine) consumes.

    A config WITHOUT `genotype_effect_strategies` is returned as-is
    (same object, I11) - nothing to expand. Otherwise a deep copy is
    built with the multi-option-only keys removed, and:
      * `genotype_effect_strategy`/`genotype_effect`/`use_grm` set from
        `strategy`;
      * `strategy_settings[strategy]` (whitelisted keys) applied on top;
      * `committed_syntax`/`syntax_fingerprint` taken from
        `committed_syntaxes`/`syntax_fingerprints`;
      * `accuracy_metrics.use_grm` pinned per option: the user's own
        "with/without GRM" choice applies to BLUEs only (Requirements.md
        item 4); BLUPs always use their OWN fit's variance information
        (without GRM for 'blup_iid', with GRM for 'blup_grm').
    `per_set_grm`/`mask_valid_test` default to False for an option whose
    settings omit them, so an option can never inherit a flag that only
    makes sense for a different option.
    """
    if not asreml_cfg or 'genotype_effect_strategies' not in asreml_cfg:
        return asreml_cfg
    strategies = resolve_strategy_list(asreml_cfg)
    if strategy is None:
        strategy = strategies[0]
    if strategy not in strategies:
        raise ValueError(
            f"asreml_syntax.strategy_config: {strategy!r} is not among this config's "
            f"strategies {strategies!r}."
        )
    import copy as _copy
    cfg = _copy.deepcopy(dict(asreml_cfg))
    settings = _copy.deepcopy((cfg.pop('strategy_settings', None) or {}).get(strategy, {}) or {})
    committed = cfg.pop('committed_syntaxes', None) or {}
    fingerprints = cfg.pop('syntax_fingerprints', None) or {}
    cfg.pop('genotype_effect_strategies', None)

    genotype_effect, use_grm = genotype_effect_strategy_engine_params(strategy)
    cfg['genotype_effect_strategy'] = strategy
    cfg['genotype_effect'] = genotype_effect
    cfg['use_grm'] = use_grm
    cfg['mask_valid_test'] = False
    cfg['per_set_grm'] = False
    for key in STRATEGY_SETTING_KEYS:
        if key in settings:
            cfg[key] = settings[key]
    if strategy != 'blue_fixed':
        cfg['blue_refinement'] = {'enabled': False, 'grm_blend': 0.01, 'max_excluded_fraction': 0.05}
        cfg['blue_weighting'] = 'none'
    if strategy != 'blup_grm':
        cfg['per_set_grm'] = False

    acc = dict(cfg.get('accuracy_metrics') or {})
    if acc:
        if strategy == 'blup_iid':
            acc['use_grm'] = False
        elif strategy == 'blup_grm':
            acc['use_grm'] = True
        cfg['accuracy_metrics'] = acc

    cfg['committed_syntax'] = committed.get(strategy, cfg.get('committed_syntax'))
    cfg['syntax_fingerprint'] = fingerprints.get(strategy, cfg.get('syntax_fingerprint'))
    return cfg


def accuracy_companion_plan(main_strategy: str, use_grm: bool) -> "Optional[dict]":
    """Which EXTRA ASReml fit (if any) the "Calculate adjusted prediction
    accuracy" block needs for an option running `main_strategy` - the
    single place that decision is made, so `genomic_prediction.py` builds
    exactly the companions listed here and nothing else.

    Returns ``{'numerator_mode': ..., 'reference': ...}``, or ``None`` for a
    strategy the block does not support (the legacy masked-holdout engine).

    ``reference`` names the fit that supplies Heritability (the H^2 the
    adjusted accuracy divides by):

      * ``'main'``        - the option's own fit already is a genotype-random
                            fit of the requested kind: NO extra fit.
      * ``'blup_no_grm'`` - a genotype-random, GRM-less companion fit.
      * ``'blup_grm_ref'``- a genotype-random, GRM-based companion fit.

    ``numerator_mode`` is always ``'reuse_pearson'``: every option's
    adjusted accuracy is its own Pearson correlation against its own test
    target (the BLUE / BLUP-without-GRM / BLUP-with-GRM values the option
    itself produced) divided by sqrt(H^2). A BLUP option therefore needs NO
    BLUE fit at all - the BLUE companion an earlier version fitted only to
    re-correlate against BLUEs is gone.

    Companion fits that remain, and why each is required:

      ======================  ===========================================
      option                  extra fit
      ======================  ===========================================
      BLUEs                   one BLUP companion (H^2) - GRM-based when
                              ``use_grm`` else GRM-less
      BLUPs without GRM       none (``use_grm`` False); a GRM-based
                              reference only if ``use_grm`` was set True
      BLUPs with GRM          none (``use_grm`` True); a GRM-less
                              reference only if ``use_grm`` was set False
      ======================  ===========================================

    (The two-step-BLUE refinement fit is separate and independent of this
    plan - it is gated on ``ASREML['blue_refinement']['enabled']`` alone.)
    """
    if main_strategy == 'blue_fixed':
        return {'numerator_mode': 'reuse_pearson',
                'reference': 'blup_grm_ref' if use_grm else 'blup_no_grm'}
    if main_strategy == 'blup_iid':
        return {'numerator_mode': 'reuse_pearson',
                'reference': 'blup_grm_ref' if use_grm else 'main'}
    if main_strategy == 'blup_grm':
        return {'numerator_mode': 'reuse_pearson',
                'reference': 'main' if use_grm else 'blup_no_grm'}
    return None


def expand_strategy_configs(asreml_cfg: "Optional[dict]") -> "List[Tuple[str, dict]]":
    """`[(strategy, single_strategy_cfg), ...]` in canonical order - the
    list `genomic_prediction.GP()`'s wrapper iterates. `[]` when ASReml
    is off."""
    if not asreml_cfg:
        return []
    return [(s, strategy_config(asreml_cfg, s)) for s in resolve_strategy_list(asreml_cfg)]


def validate_all_strategies(asreml_cfg: "Optional[dict]", scenario: str,
                            available_columns: "Optional[list]" = None) -> "List[str]":
    """Every structural + scenario error across ALL selected options
    (each error prefixed with the option's label, so a user knows which
    one to fix). `available_columns=None` skips the column-existence
    checks (`validate_structure()` only). Also returns the
    "nothing selected" error instead of raising, so GUI/gather_config
    callers can show it verbatim."""
    if not asreml_cfg:
        return []
    try:
        expanded = expand_strategy_configs(asreml_cfg)
    except ValueError as exc:
        return [str(exc)]
    errors: List[str] = []
    for strategy, cfg in expanded:
        prefix = f"[{strategy_label(strategy)}] "
        if available_columns is None:
            errs = list(validate_structure(cfg))
        else:
            errs = list(validate(cfg, available_columns))
        errs += list(validate_scenario_compatibility(cfg, scenario))
        errors.extend(prefix + e for e in errs)
    return errors


RESPONSE_UNITS: Tuple[str, ...] = ('auto', 'genotype', 'genotype_by_population')
DEFAULT_RESPONSE_UNIT = 'auto'

# ---------------------------------------------------------------------------
# ver4-20 follow-up (Requirements.md - "simplify the GUI and internal logic
# of the Genotype effect / genomic relationship matrix configuration
# subsection"). The new engine's masking toggle - `ASREML['mask_valid_test']`
# - is a THIRD, independent axis alongside `genotype_effect_strategy` and
# `response_unit`: off (the default) reproduces `fit_unmasked_adjustment()`'s
# original, ver4-19 behaviour exactly (every individual's own record,
# nothing held out); on replaces validation/test individuals' own response
# with NA before the SAME single fit, so their adjusted value becomes a
# genuine held-out BLUP/GBLUP prediction (or, for a FIXED genotype term, an
# inestimable value UNLESS that same genotype also has a real observation
# elsewhere - see `Preprocess/asreml_blup.py::fit_unmasked_adjustment()`'s
# own updated docstring). Deliberately reuses the SAME single-fit machinery
# for both settings - no second "holdout model"/two-fit apparatus is
# reintroduced - which is what keeps this a data-masking toggle rather than
# a second engine.
# ---------------------------------------------------------------------------
DEFAULT_MASK_VALID_TEST = False


def resolve_mask_valid_test(asreml_cfg: "Optional[dict]") -> bool:
    """`bool(ASREML.get('mask_valid_test', DEFAULT_MASK_VALID_TEST))` - the
    single place that default is expressed, read by both `main_app.py` and
    `genomic_prediction.py::GP()` so neither can independently drift on
    what an absent key means.
    """
    if not asreml_cfg:
        return DEFAULT_MASK_VALID_TEST
    return bool(asreml_cfg.get('mask_valid_test', DEFAULT_MASK_VALID_TEST))


# ---------------------------------------------------------------------------
# Requirements.md (latest pass), item 1 - "Genotype effect / genomic
# relationship matrix configuration" gains a new, explicit
# `ASREML['use_entire_dataset']` axis: whether the new engine's single
# per-trait fit (`fit_unmasked_adjustment()`) is restricted to just the
# population(s) actually relevant to ONE task (`False`, the DEFAULT - see
# below), or is deliberately widened to pool EVERY population the
# phenotype file contains into one fit per trait (`True`).
#
# CLARIFIED REQUIREMENT (recorded here, superseding this constant's own
# first-pass reasoning, which read the requirement backwards): every
# OTHER stage of the pipeline already restricts itself to "the target
# population" for a given task - `genomic_prediction.py::_process_one_
# task()`'s own Step 1 keeps only `rows where population ==
# sample.population` (SCENARIO='within'), or the 2-3 populations named
# in that task's own 'X->Y'/'X->Y->Z' string (SCENARIO='between') -
# before ANY other Step (LD pruning, RF filtering, model fitting) ever
# sees the data. "The structure is identical to the current version"
# means Step 6b (this ASReml stage) now follows that SAME, already-
# established discipline by default: EasiGP selects only the target
# population's own raw records for the BLUE/BLUP fit, exactly as it
# already does for every model downstream of it, and ignores the rest.
#
# `ASREML['use_entire_dataset'] = True` is the explicit, opt-in
# WIDENING of that default - pooling every OTHER population's raw
# records into the SAME fit too, while every downstream Step (model
# fitting, predictions, metrics) still only ever considers the target
# population, exactly as before. Since a pooled fit no longer depends
# on which population happens to be the target, it can be - and is -
# hoisted and REUSED across every task that predicts the same trait
# (any population, ratio, or replicate), cutting the number of real
# ASReml-R fits from one per (trait, population) down to one per trait,
# PROVIDED masking is off (`ASREML['mask_valid_test']=False`) - a masked
# fit's own result depends on which individuals THIS task masked, so it
# can never be shared across tasks regardless of `use_entire_dataset`
# (see `genomic_prediction.py::_asreml_get_unmasked_result()`'s own
# cache-key comment for the precise rule).
#
# `DEFAULT_USE_ENTIRE_DATASET = False`: an absent key (a hand-written or
# pre-existing config/GUI-state file that predates this feature) now
# gets the RESTRICTED, target-population-only behaviour - matching what
# every other Step in the pipeline has always done, and what
# Requirements.md's own item 1 calls "the current version". This IS a
# deliberate behaviour change relative to ver4-19/ver4-20, which pooled
# unconditionally with no toggle at all - recorded here explicitly
# (mirroring the module docstring's own "DEVIATION FROM THE DESIGN
# BLUEPRINT" precedent for exactly this kind of judgement call) rather
# than silently changed.
#
# Requirements.md item 4's own three-way walkthrough (entire dataset;
# target-population training set only, under masking; entire target
# population, unmasked) is the authoritative BEHAVIOURAL spec this
# resolver and its caller (`genomic_prediction.py::GP()`'s Step 6b) are
# built against - see that function's own `_asreml_get_unmasked_result()`
# comment for exactly how each of the three cases is produced.
# ---------------------------------------------------------------------------
DEFAULT_USE_ENTIRE_DATASET = False


# ---------------------------------------------------------------------------
# ver4-7, R1 (design blueprint "Per-Set GRM Generation for GBLUP") - an
# opt-in THIRD held-out estimation mode, mutually exclusive with the
# existing NA-masking mechanism (`ASREML['mask_valid_test']`): rather than
# fitting ONE genotype-random+GRM model and masking validation/test
# responses to NA before it, `ASREML['per_set_grm']=True` fits THREE
# INDEPENDENT genotype-random+GRM models - one per split (train/valid/
# test), each scoped to its own genomic relationship matrix (each split's
# own individuals, widened to also include every OTHER population's
# individuals when `resolve_use_entire_dataset()` is True - ver4-57 fix:
# this used to widen the train split only; no individual of this task's
# OTHER train/valid/test sets is ever shared - "no cross-contamination"
# between splits). See `Preprocess/
# asreml_blup.py::fit_per_set_grm_adjustment()` for the actual 3-fit
# dispatch/merge and `genomic_prediction.py`'s own Step 6b dispatch site.
#
# Only meaningful for a BLUP-with-GRM main strategy
# (`genotype_effect='random'`, `use_grm=True`) - a FIXED (BLUE) genotype
# term, or a GRM-less random one, has no genomic relationship matrix for
# this feature to scope per-split at all - see `validate_structure()`'s
# own mutual-exclusion check below.
# ---------------------------------------------------------------------------
DEFAULT_PER_SET_GRM = False


def resolve_per_set_grm(asreml_cfg: "Optional[dict]") -> bool:
    """`bool(ASREML.get('per_set_grm', DEFAULT_PER_SET_GRM))` - mirrors
    `resolve_use_entire_dataset()`'s own single-authority pattern so
    `main_app.py` and `genomic_prediction.py::GP()` can never drift on
    what an absent key means. No override exists for this flag (unlike
    `resolve_use_entire_dataset()`'s grid-residual override) - per-set
    GRM is a purely opt-in feature with no config combination that
    forces it on."""
    if not asreml_cfg:
        return DEFAULT_PER_SET_GRM
    return bool(asreml_cfg.get('per_set_grm', DEFAULT_PER_SET_GRM))


def resolve_use_entire_dataset(asreml_cfg: "Optional[dict]") -> bool:
    """`bool(ASREML.get('use_entire_dataset', DEFAULT_USE_ENTIRE_DATASET))`,
    with ONE unconditional override: whenever this run's own residual
    structure needs the complete physical grid
    (`residual_requires_full_grid()`), restricting Step 6b's raw records
    to just one task's own target population(s) would reopen the exact
    ver4-17/ver4-18 production bug ("Data has N observations but the
    residual model implies M") that lifting the population restriction
    was originally written to fix - a genetically-derived population is
    very often a subset of ONE shared physical trial, so excluding every
    OTHER population's plots silently removes real neighbouring grid
    cells the spatial/temporal residual needs to see, not merely
    genuinely-missing ones `pad_residual_grid()` can safely fill.
    `main_app.py` mirrors this same override to LOCK its own checkbox
    on (disabled, forced True) whenever a spatial/temporal residual is
    configured, so the GUI and the engine can never disagree about which
    combinations are actually reachable - the identical dual-authority
    discipline `residual_requires_full_grid()` itself already
    established (module docstring, ver4-17/ver4-18 BUGFIX notes).

    Read by both `main_app.py` (the live GUI preview/caption) and
    `genomic_prediction.py::GP()` (the actual Step 6b dispatch), so
    neither can independently drift on what an absent key, or a
    grid-residual override, means.
    """
    if residual_requires_full_grid(asreml_cfg):
        return True
    if not asreml_cfg:
        return DEFAULT_USE_ENTIRE_DATASET
    return bool(asreml_cfg.get('use_entire_dataset', DEFAULT_USE_ENTIRE_DATASET))



def validate_structure(asreml_cfg: dict) -> "list[str]":
    """The subset of `validate()`'s checks that depend ONLY on
    `asreml_cfg` itself - never on the phenotype file's own column list -
    so this can run wherever a phenotype-file header read is unavailable
    or unwanted (e.g. `genomic_prediction.py::GP()`'s own fail-fast
    preflight, Layer 2 of R13's three ASReml gating layers, which runs
    before task 0 purely from the resolved config). `validate()` below
    is this same list PLUS the column-existence checks that do need
    `available_columns`; the two must never drift, so `validate()` calls
    this rather than duplicating it.

    Checks performed:
      - `genotype_col` is set.
      - `output_level` is non-empty.
      - `genotype_col` appears in `output_level` (predict(classify=)
        must be able to report a value per genotype - otherwise R8's
        per-individual BLUE/BLUP substitution has nothing to key on).
      - every OTHER `output_level` entry is an actual model term (a
        fixed effect, random effect, interaction/nested group member, or
        residual dimension) - `model_term_columns()` above is the single
        authority for what counts. A column that is merely present in
        the phenotype file but never given to the model cannot be
        classified on; requesting one anyway is exactly what produces
        ASReml-R's own "<column> is not in the model, unable to predict
        using it" error, previously only ever surfaced deep inside the
        R fit itself (Preprocess/asreml_blup.py's run-time guard remains
        as a second, independent layer of defence for a config that
        bypassed this check entirely).
      - no column is used as both a fixed AND a random term.
      - every interaction/nested group references at least 2 columns.
      - ver4-24: no column used inside a `nested_groups` entry is ALSO
        separately selected as a plain fixed/random effect - ASReml-R's
        `A/B` nesting notation already expands to `A + A:B` (see
        `_expand_nested()`), so every column in a nested group is
        already a term in the model on its own account; re-selecting one
        standalone either literally repeats a term (`A`) or asserts an
        ill-defined independent main effect for a column that is only
        meaningful nested inside the other (`B`). Deliberately NOT
        applied to `interaction_groups`, where main effects specified
        alongside the interaction term are the normal, intended, fully-
        crossed-factorial usage (`Year + Experiment + Year:Experiment`).
      - ver4-24: the raw, un-de-duplicated fixed/random term list this
        config would produce (`_build_fixed_random_terms()`) contains no
        term more than once - a general backstop that also catches an
        interaction group and a nested group (or two identical groups)
        independently rendering the identical composite term.
      - a `fa`/`xfa` wrapper's `k` argument is a positive integer (the
        exact "k >= this factor's level count" check the blueprint
        describes needs the actual data, which this pure function does
        not have access to - Preprocess/asreml_blup.py performs that
        stronger, data-aware check immediately before fitting; this is
        the cheap, data-free half of it).
    """
    errors: List[str] = []

    genotype_col = asreml_cfg.get('genotype_col')
    if not genotype_col:
        errors.append("No genotype column selected - required to identify which effect is the "
                       "genetic one.")

    output_level = list(asreml_cfg.get('output_level', []) or [])
    if not output_level:
        errors.append("No output level selected - predict(classify=) needs at least one column "
                       "to group predictions by.")
    if genotype_col and output_level and genotype_col not in output_level:
        errors.append(f"Genotype column {genotype_col!r} must be included in the output level, "
                       f"so predict(classify=) can report one value per genotype.")

    if genotype_col and output_level:
        model_cols = model_term_columns(asreml_cfg)
        stray = [c for c in output_level if c != genotype_col and c not in model_cols]
        if stray:
            errors.append(
                f"Output-level column(s) {stray!r} are not used anywhere in this model (not a "
                f"fixed effect, random effect, interaction/nested group member, or residual-"
                f"structure dimension) - predict(classify=...) can only report levels for "
                f"columns that are actual model terms, and will fail with an R-side "
                f"'is not in the model' error otherwise. Either add {stray!r} as a model "
                f"effect above, or remove it from the output level."
            )

    fixed = set(asreml_cfg.get('fixed', []) or [])
    random = set(asreml_cfg.get('random', []) or [])
    both = fixed & random
    if both:
        errors.append(f"Column(s) {sorted(both)} are selected as BOTH a fixed and a random "
                       f"effect - a term can only be one or the other.")

    for group_kind, groups in (('interaction', asreml_cfg.get('interaction_groups', []) or []),
                                ('nested', asreml_cfg.get('nested_groups', []) or [])):
        for raw_group in groups:
            group = _normalise_group(raw_group)
            cols = group['columns']
            if len(cols) < 2:
                errors.append(f"A {group_kind} group must reference at least 2 columns - "
                               f"got {cols!r}.")

    # ver4-24 BUGFIX (production symptom: a rendered
    # ``random = ~ vm(ID, source = Ginv) + Year + Experiment + Year +
    # Year:Experiment`` - see ``build_model_spec()``'s own ver4-24
    # comment for the full root-cause explanation). ASReml-R's ``/``
    # nesting operator already expands ``Year/Experiment`` to
    # ``Year + Year:Experiment`` - EVERY column referenced inside a
    # nested group is therefore already a term in the rendered model on
    # its own account, so separately ALSO selecting one as a plain
    # fixed/random effect is refused here, rather than silently produced
    # (a literal duplicate for the outer column) or silently dropped (an
    # ill-defined standalone main effect for the inner column). This is
    # deliberately NOT applied to `interaction_groups`: main effects
    # specified alongside an interaction term are the normal, intended,
    # fully-crossed-factorial usage (`Year + Experiment + Year:
    # Experiment`), which has no such restriction.
    nested_group_columns: "set" = set()
    for raw_group in asreml_cfg.get('nested_groups', []) or []:
        nested_group_columns.update(_normalise_group(raw_group)['columns'])
    nested_overlap = sorted(nested_group_columns & (fixed | random))
    if nested_overlap:
        errors.append(
            f"Column(s) {nested_overlap!r} are used inside a nested group AND separately "
            f"selected as a plain fixed/random effect. ASReml-R's 'A/B' nesting notation "
            f"already expands to 'A + A:B' - the outer column's own main effect and the inner "
            f"column's own nested term are both already produced by the nested group itself, "
            f"so selecting either column again produces either a literal duplicate term (the "
            f"outer column) or an ill-defined standalone main effect for a column that is only "
            f"meaningful nested inside the other (the inner column). Remove {nested_overlap!r} "
            f"from the plain fixed/random effect list(s) and let the nested group supply its "
            f"own term(s) instead - or, if the columns are genuinely fully CROSSED rather than "
            f"nested (every level of one appears with every level of the other), use an "
            f"interaction group instead, which has no such restriction."
        )

    # ver4-24 - a general backstop alongside the column-level check just
    # above: inspect the exact RAW term lists `render_syntax()` would
    # build (`_build_fixed_random_terms()`, NOT yet de-duplicated -
    # unlike `build_model_spec()`'s own defensive de-duplication) for any
    # term appearing more than once on the same side of the model. This
    # also catches cases the column-level check above does not, e.g. an
    # `interaction_groups` entry and a `nested_groups` entry (or two
    # identical groups of the same kind) independently rendering the
    # identical composite term (`Year:Experiment` from both).
    _fixed_terms_raw, _random_terms_raw = _build_fixed_random_terms(asreml_cfg)
    for side_name, terms in (('fixed', _fixed_terms_raw), ('random', _random_terms_raw)):
        term_counts: "dict" = {}
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1
        dupes = sorted(t for t, n in term_counts.items() if n > 1)
        if dupes:
            errors.append(
                f"The {side_name} side of this model has duplicate term(s) {dupes!r} - the "
                f"same term would be produced more than once, by some combination of the plain "
                f"{side_name} effect list, an interaction group, and/or a nested group (see the "
                f"'used inside a nested group AND separately selected' check above for the most "
                f"common cause). Remove whichever selection is redundant."
            )

    wrappers_cfg = asreml_cfg.get('wrappers', {}) or {}
    for col, wrapper_cfg in wrappers_cfg.items():
        if not wrapper_cfg or not wrapper_cfg.get('name'):
            continue
        name = wrapper_cfg['name']
        if name in ('fa', 'xfa'):
            k = (wrapper_cfg.get('args', {}) or {}).get('k')
            try:
                if k is None or int(k) < 1:
                    errors.append(f"{name}() on column {col!r} needs a positive integer 'k' "
                                   f"(number of latent factors).")
            except (TypeError, ValueError):
                errors.append(f"{name}() on column {col!r} has a non-integer 'k' argument "
                               f"({k!r}).")

    # Requirements.md item 2: `fixed_holdout_mode` only has meaning when
    # genotype is fitted FIXED - a hand-edited/headless config that sets
    # it alongside `genotype_effect='random'` is silently ignored
    # everywhere else in this module (see `pattern_id_for_config()`), so
    # catching it here rather than leaving it to fail an unrelated-looking
    # scenario-compatibility check later.
    _fixed_holdout_mode = asreml_cfg.get('fixed_holdout_mode')
    if _fixed_holdout_mode is not None and _fixed_holdout_mode not in ('two_fit', 'single_fit'):
        errors.append(
            f"ASREML['fixed_holdout_mode']={_fixed_holdout_mode!r} is not a recognised value - "
            f"must be 'two_fit' or 'single_fit' (or omitted, which defaults to 'two_fit')."
        )
    if _fixed_holdout_mode is not None and asreml_cfg.get('genotype_effect', 'random') != 'fixed':
        errors.append(
            "ASREML['fixed_holdout_mode'] is set but genotype_effect is not 'fixed' - this "
            "setting only has meaning for a FIXED (BLUE) genotype term; remove it, or set "
            "genotype_effect='fixed'."
        )
    if _fixed_holdout_mode == 'single_fit' and asreml_cfg.get('use_grm', False):
        errors.append(
            "ASREML['fixed_holdout_mode']='single_fit' ('the original version of BLUEs' - "
            "Requirements.md item 2) has no genomic-relationship-matrix variant - it is a "
            "single, genotype-FIXED fit with no vm() term at all. Set ASREML['use_grm']=False, "
            "or use fixed_holdout_mode='two_fit' (BLUE train + GBLUP holdout) instead."
        )

    # ver4-17 BUGFIX ("Data has N observations but the residual model
    # implies M" - production failure, task 1/25 of
    # EasiGP_BLUEs_BLUPs_ASReml_0.5, population='Cluster_4', phenotype=
    # 'Maturity'): a grid-structured residual (residual_requires_full_
    # grid() above) needs ASReml to see the trial's COMPLETE physical
    # design - every population/individual the trial ever recorded, not
    # just this task's own. `_fit_blue_train_plus_holdout()`'s own Fit 1
    # ("BLUE, train") deliberately does the OPPOSITE for a different,
    # equally load-bearing reason (this module's own "Design decision:
    # BLUE cannot be held out" section above): it PHYSICALLY EXCLUDES
    # every non-training row, precisely so a validation/test individual's
    # response can never inform the shared effects that adjust everyone
    # else. Masking those rows instead of excluding them (the mechanism
    # that preserves grid completeness everywhere else in this module)
    # would reopen exactly the leakage channel that design decision's own
    # "rejected option" already ruled out - so there is no correct way to
    # satisfy both requirements in Fit 1 at once, and this combination is
    # therefore refused HERE, before any R fit is attempted, rather than
    # either silently producing a leaky BLUE or letting it fail with the
    # same cryptic, unrelated-looking R-side size-mismatch error the
    # single-fit path used to. Patterns 3-4 (genotype_effect='random' -
    # BLUP/GBLUP, single fit) are entirely unaffected - a grid residual
    # composes correctly with those, since masking (never exclusion) is
    # already how that path handles a held-out individual.
    #
    # ver4-23 BUGFIX ("this version does not allow a run" - the check
    # below used to fire regardless of WHICH Step-6b engine the run would
    # actually use, because it read `genotype_effect`/`fixed_holdout_mode`
    # alone and defaulted a missing `fixed_holdout_mode` to `'two_fit'`.
    # That default is correct for a `legacy_masked_holdout=True` config
    # (`fixed_holdout_mode` genuinely defaults to `'two_fit'` there - see
    # `asreml_adjust_phenotypes()`'s own docstring), but the NEW (default,
    # `legacy_masked_holdout=False`) engine never resolves, reads, or acts
    # on `fixed_holdout_mode` at all - `fit_unmasked_adjustment()` fits
    # every row (train, validation and test alike) in the SAME single,
    # UNRESTRICTED call regardless of `genotype_effect`, exactly like
    # "the single-fit path this reduces to" the message below already
    # promises for `genotype_effect='random'`. A grid residual therefore
    # composes correctly with `genotype_effect='fixed'` under the new
    # engine too - there is no `_fit_blue_train_plus_holdout()`-style
    # training-only physical restriction to conflict with it in the first
    # place, whether or not `ASREML['mask_valid_test']` is on (masking
    # only ever overwrites a row's own response with NA - the same
    # `na.action = na.method(y = "include")`-safe mechanism
    # `pad_residual_grid()` itself relies on - never excludes a row).
    # Gating this check on `legacy_masked_holdout` (the one condition
    # `_fit_blue_train_plus_holdout()` is actually reachable under -
    # see `genomic_prediction.py`'s own Step 6b dispatch) is what lets a
    # grid-structured residual run at all under the new default engine,
    # which was being refused here even though nothing about it is
    # actually incompatible.
    if (residual_requires_full_grid(asreml_cfg)
            and asreml_cfg.get('genotype_effect', 'random') == 'fixed'
            and asreml_cfg.get('legacy_masked_holdout', False)
            and (_fixed_holdout_mode or 'two_fit') == 'two_fit'):
        errors.append(
            "ASREML['residual']['dims'] configures a grid-structured residual (e.g. "
            "ar1(...):ar1(...)) that requires ASReml to see the trial's complete physical "
            "design - one row per combination of those dimensions' levels, spanning every "
            "population/individual, not just one task's own. genotype_effect='fixed' with "
            "fixed_holdout_mode='two_fit' (the default whenever any validation/test individual "
            "exists) fits its 'BLUE, train' half on a dataset PHYSICALLY RESTRICTED to training "
            "individuals only (Preprocess/asreml_blup.py::_fit_blue_train_plus_holdout() - a "
            "deliberate leakage-avoidance design, not an oversight), which is fundamentally "
            "incompatible with a grid residual's own completeness requirement and is not "
            "supported under ASREML['legacy_masked_holdout']=True. Use genotype_effect='random' "
            "(BLUP/GBLUP - a grid residual works correctly with the single-fit path this "
            "reduces to), fixed_holdout_mode='single_fit' together with SCENARIO='between' and "
            "ASREML['between_shared_genotypes']=True (no partition is physically restricted or "
            "masked in that mode - Requirements.md item 2), or - the recommended option - drop "
            "ASREML['legacy_masked_holdout'] (or set it False) to use the new default engine, "
            "where a grid residual composes correctly with genotype_effect='fixed' because "
            "nothing is ever physically restricted or excluded."
        )

    # ver4-25, R7 (design blueprint stage 4): the two-step refined-BLUE
    # procedure only ever makes sense for a genotype-FIXED main strategy
    # (see Preprocess/asreml_blup.py::fit_unmasked_adjustment()'s own
    # docstring on `step1_vparameters` for why) and only under the NEW
    # default engine (the legacy masked-holdout engine's own two-fit
    # BLUE/holdout split has a different, incompatible "genotype fixed"
    # meaning entirely - it never calls fit_unmasked_adjustment() at
    # all). Checked here, data-free, rather than only failing loudly deep
    # inside the R fit - matching this function's own stated purpose.
    _blue_refinement_cfg = asreml_cfg.get('blue_refinement') or {}
    if _blue_refinement_cfg.get('enabled', False):
        if asreml_cfg.get('legacy_masked_holdout', False):
            errors.append(
                "ASREML['blue_refinement']['enabled']=True is not supported together with "
                "ASREML['legacy_masked_holdout']=True - the two-step refined-BLUE procedure is "
                "a new-engine-only feature (Preprocess/asreml_blup.py::fit_unmasked_adjustment()). "
                "Drop ASREML['legacy_masked_holdout'] (or set it False) to use it."
            )
        elif asreml_cfg.get('genotype_effect', 'random') != 'fixed':
            errors.append(
                "ASREML['blue_refinement']['enabled']=True requires ASREML['genotype_effect']="
                "'fixed' (a BLUE strategy) - a RANDOM genotype term already estimates its own "
                "variance jointly with every other term in one fit, so there is nothing for "
                "this two-step procedure to refine."
            )
        _grm_blend = _blue_refinement_cfg.get('grm_blend', 0.01)
        if not isinstance(_grm_blend, (int, float)) or not (0.0 <= _grm_blend <= 1.0):
            errors.append(
                f"ASREML['blue_refinement']['grm_blend']={_grm_blend!r} must be a number "
                f"between 0.0 and 1.0."
            )
        _max_excluded = _blue_refinement_cfg.get('max_excluded_fraction', 0.05)
        if not isinstance(_max_excluded, (int, float)) or not (0.0 <= _max_excluded <= 1.0):
            errors.append(
                f"ASREML['blue_refinement']['max_excluded_fraction']={_max_excluded!r} must be "
                f"a number between 0.0 and 1.0."
            )

    # ver4-7, R1 - mutual exclusion between the two held-out estimation
    # mechanisms: masking (NA-in-place, one shared fit) and per-set GRM
    # (three independent, split-scoped fits). Belt-and-braces beyond the
    # GUI's own force-before-render lock (main_app.py's mutual-exclusion
    # pair for these two checkboxes) - a hand-written/headless config
    # that sets both must be refused here too, mirroring every other
    # dual-authority check this module already applies (e.g. `residual_
    # requires_full_grid()`'s own GUI+engine pairing).
    if resolve_per_set_grm(asreml_cfg) and resolve_mask_valid_test(asreml_cfg):
        errors.append(
            "ASREML['per_set_grm']=True and ASREML['mask_valid_test']=True cannot both be set - "
            "these are two DIFFERENT (mutually exclusive) held-out estimation mechanisms: masking "
            "overwrites validation/test responses with NA before ONE shared genotype-random+GRM "
            "fit; per-set GRM instead fits THREE INDEPENDENT genotype-random+GRM models, one per "
            "split, each scoped to its own relationship matrix. Disable one to use the other."
        )
    if resolve_per_set_grm(asreml_cfg) and not (
        asreml_cfg.get('genotype_effect', 'random') == 'random' and asreml_cfg.get('use_grm', False)
    ):
        errors.append(
            "ASREML['per_set_grm']=True requires ASREML['genotype_effect']='random' and "
            "ASREML['use_grm']=True (a BLUP-with-GRM main strategy) - a FIXED (BLUE) genotype "
            "term, or a GRM-less random one, has no genomic relationship matrix for this feature "
            "to scope per split at all."
        )

    return errors


def validate(asreml_cfg: dict, available_columns: "list[str]") -> "list[str]":
    """Semantic validation of `asreml_cfg` against the phenotype file's
    own column set, run BEFORE syntax is ever rendered or committed (R7)
    - both by the GUI's commit button and by `gather_config()` for a
    headless config that bypassed the GUI entirely. Returns a list of
    human-readable error strings (empty list = valid); never raises.

    This is `validate_structure(asreml_cfg)` (every config-only check -
    see that function's own docstring for the full list) PLUS the
    checks below that additionally need `available_columns`:
      - `genotype_col` is present in `available_columns`.
      - every `output_level` entry is a real column.
      - every fixed/random effect column is a real column.
      - every residual dimension is a real column.
      - every interaction/nested group column is a real column.
    """
    errors: List[str] = list(validate_structure(asreml_cfg))
    available = set(available_columns or [])

    genotype_col = asreml_cfg.get('genotype_col')
    if genotype_col and genotype_col not in available:
        errors.append(f"Genotype column {genotype_col!r} was not found among the phenotype "
                       f"file's own columns.")

    output_level = list(asreml_cfg.get('output_level', []) or [])
    for col in output_level:
        if col not in available:
            errors.append(f"Output-level column {col!r} was not found among the phenotype "
                           f"file's own columns.")

    fixed = set(asreml_cfg.get('fixed', []) or [])
    random = set(asreml_cfg.get('random', []) or [])
    for col in fixed | random:
        if col not in available:
            errors.append(f"Effect column {col!r} was not found among the phenotype file's "
                           f"own columns.")

    residual_cfg = asreml_cfg.get('residual')
    if residual_cfg:
        for dim in residual_cfg.get('dims', []):
            if dim not in available:
                errors.append(f"Residual structure column {dim!r} was not found among the "
                               f"phenotype file's own columns.")

    for group_kind, groups in (('interaction', asreml_cfg.get('interaction_groups', []) or []),
                                ('nested', asreml_cfg.get('nested_groups', []) or [])):
        for raw_group in groups:
            group = _normalise_group(raw_group)
            for col in group['columns']:
                if col not in available:
                    errors.append(f"{group_kind.capitalize()} group column {col!r} was not "
                                   f"found among the phenotype file's own columns.")

    return errors


def config_fingerprint(asreml_cfg: "Optional[dict]") -> str:
    """Deterministic hash of the STRUCTURAL parts of an ``ASREML`` config
    (everything except the already-committed text/fingerprint fields
    themselves - PRIMARY (``committed_syntax``/``syntax_fingerprint``)
    AND HOLDOUT (``holdout_committed_syntax``/``holdout_syntax_
    fingerprint`` - module docstring, "Design decision: BLUE cannot be
    held out"), which would make this self-referential either way) -
    used for two distinct purposes:

    1. The GUI's "stale syntax" detector (R7, RK-8): compares a LIVE
       fingerprint (recomputed from the widgets on every rerender)
       against the fingerprint that was current when the syntax was last
       committed, so editing a control after committing shows a
       persistent warning rather than silently running stale syntax.
    2. Folded into ``checkpoint_utils.sample_fingerprint(..., extra=)``
       (R11c) alongside a hash of the COMMITTED syntax text itself, so
       resuming a checkpoint after either the widgets or the committed
       text changed is refused rather than silently mixed.

    Returns a stable value (``'asreml-disabled'``) for ``None`` - i.e. a
    run with ASReml off has one single, constant fingerprint, so turning
    it on/off is itself a fingerprint-changing (checkpoint-invalidating)
    event, exactly as changing any other structural setting would be.
    """
    if not asreml_cfg:
        return 'asreml-disabled'
    payload = {k: v for k, v in asreml_cfg.items() if k not in (
        'committed_syntax', 'syntax_fingerprint',
        'holdout_committed_syntax', 'holdout_syntax_fingerprint',
        'committed_syntaxes', 'syntax_fingerprints',
    )}
    try:
        serialised = json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        serialised = str(payload)
    return hashlib.sha256(serialised.encode('utf-8')).hexdigest()


def reconcile_committed_syntax_with_response_unit(
        asreml_cfg: dict, committed_syntax: str, resolved_response_unit: str, *,
        population_col: str = 'population', log_prefix: str = '[ASReml]') -> str:
    """Requirements.md item 4 fix - production failure: ``Error in
    asreml(fixed = ... ~ 1, random = ~Year + Experiment, ... : Error :
    population is not in the model, unable to predict using it``, raised
    deep inside rpy2 from ``fit_unmasked_adjustment()``, after real
    fitting time had already been spent.

    ROOT CAUSE: ``ASREML['response_unit']`` may be ``'auto'``, resolved
    against the REAL phenotype file only at run time
    (``Preprocess/asreml_blup.py::resolve_response_unit()``). The GUI's
    own live syntax preview cannot do that same resolution at commit
    time (no data read happens there - see ``main_app.py``'s own
    "the live preview always renders the 'genotype' ... shape" caption),
    so whatever gets committed under 'auto' is always rendered for the
    UNSHARED ('genotype') shape. Whenever the real file resolves 'auto'
    to ``'genotype_by_population'`` instead, the committed syntax has no
    ``population`` term in it at all (no ``Genotype:population``/
    ``at(population):Genotype``/``diag(population):vm(...)`` - see
    ``render_syntax()`` above), while the CALLER separately adds
    ``population`` to ``output_level``/``predict(classify=...)`` based on
    the (correctly) resolved response unit - a mismatch between "what the
    fitted model actually contains" and "what predict() is asked to
    report by" that ASReml-R can only discover, and refuse, deep inside
    the fit itself.

    This function is the single place that mismatch is caught, BEFORE any
    fit is attempted (the same "Layer 2, before task 0" placement
    ``genomic_prediction.py::GP()``'s own ``validate_structure()`` call
    already uses for the analogous, config-only version of this same
    class of error).

    Returns
    -------
    The syntax text `fit_unmasked_adjustment()` should actually use:

    - `committed_syntax` UNCHANGED, always, whenever
      `asreml_cfg['response_unit']` was configured EXPLICITLY (not
      `'auto'`) - ver4-22 BUGFIX: there is no run-time-only ambiguity to
      reconcile in that case (the GUI's preview/commit already used the
      same shape), so no other staleness reason (GRM toggled, strategy
      changed, ...) is second-guessed here either - see the ver4-22
      comment below.
    - `committed_syntax` UNCHANGED, whenever it already matches
      `resolved_response_unit`'s own auto-rendered shape (the common
      case: nothing needed fixing).
    - a freshly re-rendered syntax for `resolved_response_unit`, whenever
      `committed_syntax` is BYTE-IDENTICAL to this SAME config's own
      auto-rendered text for the OTHER response-unit shape - i.e. nothing
      was ever hand-edited (R7's "generate-then-edit-then-use" contract
      was never exercised), so silently upgrading/downgrading it to match
      the shape the real data requires is safe and exactly what "Detect
      automatically" was asked to do. Logged, never silent.

    Raises
    ------
    ValueError if `committed_syntax` matches NEITHER shape's own
    auto-rendered text - i.e. it HAS been hand-edited, and does not
    already match the resolved shape either - so this function has no
    safe basis to guess what the person actually wants (this module's
    "name it rather than assume" standard). The alternative is exactly
    the production failure this fix replaces: an opaque, deep-in-rpy2
    ASReml-R error after real fitting time has already been spent.
    """
    if resolved_response_unit not in ('genotype', 'genotype_by_population'):
        raise ValueError(
            f"asreml_syntax.reconcile_committed_syntax_with_response_unit: "
            f"resolved_response_unit={resolved_response_unit!r} must already be resolved to "
            f"'genotype' or 'genotype_by_population' (never 'auto') - see "
            f"Preprocess/asreml_blup.py::resolve_response_unit()."
        )
    if not committed_syntax or '{TRAIT}' not in committed_syntax:
        raise ValueError(
            f"{log_prefix} ASREML['committed_syntax'] is empty or missing the {{TRAIT}} "
            f"placeholder - commit a syntax from the GUI's ASReml section, or set it "
            f"directly in a hand-written config, before running with ASREML enabled."
        )

    # ver4-22 BUGFIX (production failure - a committed syntax rejected as
    # unreconcilable, and misreported as a `population`/response_unit
    # mismatch, when the REAL cause was unrelated staleness - e.g. the
    # "Use GRM" checkbox toggled AFTER the syntax was committed, with no
    # response_unit ambiguity involved at all): this whole function
    # exists SOLELY to resolve the ambiguity created by
    # ``ASREML['response_unit'] == 'auto'`` (see ROOT CAUSE above, and
    # this call site's own comment in ``genomic_prediction.py::GP()``,
    # "whenever ASREML['response_unit'] is 'auto' ..."). The GUI cannot
    # know, at commit time, which of the two shapes 'auto' will resolve
    # to at run time, so its live preview always renders (and the person
    # always commits) the 'genotype' shape for it - this function is what
    # transparently upgrades that to 'genotype_by_population' when the
    # real phenotype file needs it, or refuses to guess if the committed
    # text was ALSO hand-edited.
    #
    # Whenever ``ASREML['response_unit']`` was configured EXPLICITLY
    # (never 'auto' to begin with - e.g. every SCENARIO='within' run,
    # where ``main_app.py::_asreml_build_cfg_from_widgets()`` hardcodes
    # 'genotype' outright, since "a 'within' task covers exactly one
    # population" - Requirements.md R3), the GUI's live preview already
    # rendered, and the person already committed, the SAME shape that
    # governs this run - there is no run-time-only resolution and
    # therefore no response_unit ambiguity to reconcile here at all.
    # Re-deriving an "expected" syntax from `asreml_cfg`'s OTHER fields
    # (GRM on/off, genotype-effect strategy, wrappers, fixed/random
    # terms, ...) and enforcing it in that case would reject a syntax for
    # reasons that have nothing to do with response_unit/`population`,
    # yet get blamed on this function's own (response_unit-specific)
    # diagnostic message below. Staleness of THAT kind is exactly what
    # the GUI's own "the controls above have changed since this syntax
    # was last committed - press 'Commit syntax' again... or the
    # PREVIOUSLY committed text will still be used" warning already
    # covers - and its own documented behaviour is to keep using the
    # previously committed text, not to refuse to run. We do the same
    # here: skip the shape-matching check entirely and hand the
    # committed syntax back unchanged (the two checks just above -
    # non-empty, contains {TRAIT} - still apply regardless).
    if (asreml_cfg or {}).get('response_unit', 'auto') != 'auto':
        return committed_syntax

    other_response_unit = (
        'genotype_by_population' if resolved_response_unit == 'genotype' else 'genotype'
    )

    def _auto_render(unit: str) -> str:
        spec = build_model_spec(
            asreml_cfg, 'placeholder_trait', response_unit=unit, population_col=population_col,
        )
        return render_syntax(
            spec, data_symbol='{DATA}', grm_symbol='{GRM}' if asreml_cfg.get('use_grm') else None,
        )

    _expected = _auto_render(resolved_response_unit)
    if committed_syntax.strip() == _expected.strip():
        return committed_syntax

    _other_auto = _auto_render(other_response_unit)
    if committed_syntax.strip() == _other_auto.strip():
        print(
            f"{log_prefix} NOTE: ASREML['response_unit']='auto' resolved to "
            f"{resolved_response_unit!r} for this run's real phenotype file, but the committed "
            f"syntax was generated (and never hand-edited) for {other_response_unit!r} - the GUI's "
            f"'auto' preview cannot re-check the real file at commit time (see its own caption). "
            f"Transparently re-rendering the SAME, unedited syntax for the resolved shape "
            f"(Requirements.md R3/item 4) - re-commit an explicit response unit in the GUI to "
            f"silence this note."
        )
        return _expected

    raise ValueError(
        f"{log_prefix} ASREML['committed_syntax'] does not match the resolved "
        f"response_unit={resolved_response_unit!r} (this run's real phenotype file needs "
        f"predict(classify=...) to include {population_col!r}, but the committed model has no "
        f"{population_col!r} term at all - or vice versa), AND it does not match either "
        f"response-unit shape's own auto-generated (unedited) syntax, so it appears to have been "
        f"hand-edited - this cannot be safely reconciled automatically. Re-open the ASReml-R "
        f"section, explicitly select the response unit the run log just reported (rather than "
        f"'Detect automatically'), re-generate the syntax, re-apply your hand edit on top of that, "
        f"and commit again - or, for a hand-written config, set ASREML['response_unit'] "
        f"explicitly and render ASREML['committed_syntax'] to match."
    )
