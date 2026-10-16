"""
Preprocess/asreml_blup.py
============================
Update ID ver4-12/ver4-13, R5/R8/R9/R10/R12 (design blueprint SS2.5
"Module layout"); ver4-13 additionally implements "Design decision: BLUE
cannot be held out" (see ``asreml_syntax.py``'s own module docstring
section of that name for the full rationale and the four resulting
configuration patterns).

ver4-15 BUGFIX ("'source' is missing a 'dimnames' attribute"): the GRM
route's inverse relationship matrix used to be attributed
(``asreml_prepare_ginv()`` - row/column names + the ASReml-R 4.x
``INVERSE`` attribute) from THIS side of the Python<->R boundary, then
handed to ``asreml_fit_and_predict()`` as an already-prepared object. That
return trip through rpy2 silently stripped the attribution: the
process-wide ``numpy2ri`` converter (``pipeline_utils.
init_rpy2_conversion()``) converts any R matrix crossing back into Python
into a plain ``numpy.ndarray``, which cannot carry dimnames or arbitrary R
attributes at all, so ``Ginv`` reached the fit with none - producing
``asr_vm(ID, source = Ginv, data = data) : 'source' is missing a
'dimnames' attribute`` the moment the committed syntax's own
``vm(<genotype col>, source = Ginv)`` ran. Fixed by moving the
``asreml_prepare_ginv()`` call inside ``asreml_fit_and_predict()`` itself
(Preprocess/asreml_fit.R) - the raw matrix and the ID vector it needs are
now passed straight through, and the attributed result is consumed
entirely R-side, never crossing back into Python.

ver4-16 BUGFIX ("unused argument (\"ID\")"): the ver4-15 change above added
the ID vector to THIS module's call to ``asreml_fit_and_predict()``
(``ginv_ids_r``, positioned right after ``ginv``) but the R-side function
signature in ``Preprocess/asreml_fit.R`` was never actually given the
matching ``ginv_ids`` parameter to receive it - it still only accepted
``ginv`` and ``genotype_col``. rpy2 therefore matched this module's 12
positional arguments against R's 11 formal parameters one slot short:
``ginv_ids_r`` (a character vector of IDs) landed in the ``genotype_col``
slot, and this module's own (correct) ``genotype_col`` value - typically
the literal string ``"ID"`` - became a 12th, unmatched positional
argument, which R reported as ``unused argument ("ID")``. Fixed R-side
only: ``asreml_fit_and_predict()`` now declares ``ginv_ids = NULL``
between ``ginv`` and ``genotype_col`` (matching this module's existing,
unchanged call order exactly) and calls ``asreml_prepare_ginv(ginv,
ginv_ids)`` internally before the matrix is used anywhere else in the
function - completing the ver4-15 fix this docstring already described.

ver4-17 BUGFIX ("Data has N observations but the residual model implies
M"): a grid-structured residual (``ASREML['residual']['dims']`` non-
empty - ``ar1(Range):ar1(Row)`` being the motivating production case,
task 1/25 of EasiGP_BLUEs_BLUPs_ASReml_0.5, population='Cluster_4',
phenotype='Maturity') needs ASReml to see ONE ROW PER COMBINATION of
those dimensions' levels - normally the trial's complete physical plot
layout, spanning every population/individual it ever recorded, not just
one task's own. ``_fit_single_asreml_model()`` used to filter its own
``working`` DataFrame down to ``all_ids`` (this task's own train+valid+
test individuals) UNCONDITIONALLY - correct, and still the default,
whenever no grid residual is configured, but silently gap-filled (and
therefore size-mismatched) whenever one is, if ``population`` is a
genetically-DERIVED subset of one shared trial rather than a literal
per-experiment population. Fixed by consulting the new single authority,
``asreml_syntax.residual_requires_full_grid()``, on BOTH sides of the
Python<->R boundary that make this same decision independently
(``genomic_prediction.py``'s Step 6b, and this function's own defensive
backstop): whenever it is ``True``, the population/task row filter is
skipped entirely, and every "exterior" individual's row is kept with its
OWN real response untouched (never masked, never removed) - it informs
the shared spatial/residual structure only, exactly the same "design
columns are never touched" principle Requirement 8's own masking step
already applied to ``id_valid``/``id_test`` rows, just extended to
individuals outside this task altogether. See that step's own updated
comment below for the full mechanics, including why a ``grm_source=
'task_pool'`` GRM not covering an exterior individual's genotype is
already handled gracefully by this integration's own pre-existing
``na.action = na.method(x = "include")`` (present in every reference
syntax this module was reviewed against), and why the FIXED-genotype
two-fit path (``_fit_blue_train_plus_holdout()``) is refused outright
for this combination instead of being "fixed" the same way -
``asreml_syntax.validate_structure()``'s own new check has the full
reasoning.

ver4-18 BUGFIX ("Data has N observations but the residual model implies
M" - STILL reproduced after ver4-17, a SEPARATE production failure: task
1/25 of EasiGP_BLUEs_BLUPs_ASReml_0.5, population='Cluster_1',
phenotype='Glaucousness', "Data has 8403 observations but the residual
model implies 11424"): ver4-17 lifted the population restriction, which
is necessary but not sufficient - the raw phenotype file can itself lack
a row for some (Range, Row) combination under ANY population or trait at
all (a border plot, a discarded/failed plot, an irregular field
boundary), and no amount of population-restriction-lifting supplies a
row that was never recorded in the first place. Fixed by
``asreml_syntax.pad_residual_grid()`` - the single authority for
explicitly completing the residual dimensions' own Cartesian product
with NaN-filled filler rows, called once by ``genomic_prediction.py``
(Step 6b) and again here, defensively/idempotently, by
``_fit_single_asreml_model()`` below - see that function's own updated
comment at the call site, and ``pad_residual_grid()``'s own docstring,
for the full mechanics.

ver4-20 follow-up (Requirements.md - "simplify the GUI and internal logic
of the Genotype effect / genomic relationship matrix configuration
subsection"): `fit_unmasked_adjustment()` gains an optional `mask_keys`
parameter. `None` (the default) reproduces its original ver4-19 behaviour
exactly - nothing masked, nothing held out. A non-empty set of
`(genotype_id, population_or_None)` tuples instead overwrites those rows'
own response with NA before the SAME single fit (no second "holdout"
model is reintroduced) - see that function's own docstring for the exact
matching rule, and `genomic_prediction.py`'s own `_asreml_get_unmasked_
result()` for how the per-trait cache is bypassed once masking is on
(the result then depends on this task's own valid/test membership, not
just the trait). This is a strict generalisation: every existing caller
that never passes `mask_keys` is unaffected.

ver4-21 BUGFIX ("60 individual(s)/cell(s) have a NaN adjusted phenotype
value for trait 'LY' - ... unlikely once na.action='include' is in
effect" - production failure on a real multi-trait Cotton trial):
`fit_unmasked_adjustment()`'s unmasked NaN post-condition check used to
treat ANY NaN adjusted value as fatal, on the stated (and, on real data,
simply wrong) assumption that it was rare. `raw_records` is deliberately
the WHOLE trial's raw file across every trait/population this run has,
so it routinely includes genotypes phenotyped for OTHER traits that have
ZERO raw rows with a non-missing response for the trait actually being
fitted - the sparse/fixed genotype term is then genuinely, correctly
inestimable for them, exactly like any other missing-phenotype
individual, not an aliasing/design defect. Worse, it is provably
harmless to let it through: `pipeline_utils.collapse_raw_to_analysed_
level()`'s own placeholder mean (the value genomic_prediction.py uses to
decide `id_train`/`id_valid`/`id_test` membership BEFORE this stage
overwrites it) is a mean of zero values for the exact same genotype, so
the ordinary geno x phenotype `dropna()` has already excluded it from
every task that predicts this trait - a NaN left here for it is simply
never looked up by `apply_to_splits()`. Fixed by distinguishing, in the
post-condition check below, a genotype with zero raw observations for
this trait (now logged and left as NaN, not fatal) from one that DOES
have at least one raw observation yet still comes back NaN (still
fatal - the genuine aliased/inestimable-effect signal this check exists
to catch).

ver4-23 BUGFIX ("this version does not allow a run" - a grid-structured
residual refused unconditionally even under the NEW default engine):
the ver4-17 check in `asreml_syntax.validate_structure()` (see that
function's own updated comment) refused ANY `genotype_effect='fixed'`
config paired with a grid-structured residual whose `fixed_holdout_mode`
was absent or `'two_fit'` - correct for the LEGACY engine, where a
missing `fixed_holdout_mode` genuinely defaults to `'two_fit'` and
therefore genuinely reaches `_fit_blue_train_plus_holdout()` below, but
wrong for the NEW (default) engine, which never resolves or acts on
`fixed_holdout_mode` at all: `fit_unmasked_adjustment()` fits every
individual - train, validation and test alike - in the SAME single,
UNRESTRICTED call regardless of `genotype_effect`, so there is no
training-only physical restriction for a grid residual's completeness
requirement to conflict with in the first place, whether or not
`ASREML['mask_valid_test']` is on (masking only overwrites a row's own
response with NA - it never excludes a row, exactly like a
`pad_residual_grid()` filler row already does). Fixed by gating that
check on `ASREML['legacy_masked_holdout']` - the one condition under
which `_fit_blue_train_plus_holdout()` (this module, below) is actually
reachable at all (see `genomic_prediction.py`'s own Step 6b dispatch,
which calls THIS module's `asreml_adjust_phenotypes()` only when
`legacy_masked_holdout=True`, and `fit_unmasked_adjustment()` otherwise).
A genotype-FIXED, grid-residual config under the new default engine now
runs exactly like any other new-engine config - one unmasked (or
optionally masked) single fit, per `fit_unmasked_adjustment()`'s own
docstring below.

Layer 3a execution engine for the ASReml-R BLUE/BLUP substitution stage
(Step 6b - see genomic_prediction.py::GP()::_process_one_task()'s own
insertion point, between LD pruning and RF marker filtering). Owns:

  - masking validation/test responses to NA before fitting (R8) - the
    LEGACY engine's own, always-on behaviour; the NEW (default) engine's
    `fit_unmasked_adjustment()` masks only when `mask_keys` is supplied
    (ver4-20 follow-up, above) - off by default (Requirements.md's
    recommended default for both prediction scenarios);
  - the TWO-FIT "BLUE (train) + BLUP/GBLUP (holdout)" path
    (`_fit_blue_train_plus_holdout()`), automatically used whenever
    genotype is fitted FIXED and any validation/test individual exists -
    a FIXED genotype term is inestimable for a masked individual, so
    training individuals get a real BLUE from one model and held-out
    individuals get a BLUP/GBLUP from a second, independent one (ver4-13
    - see this module's own "Design decision" note below);
  - the pre-fit fixed-genotype estimability check (RK-9) - now a
    defensive backstop for the case above, since the public entry point
    routes a FIXED-genotype task with any masked individual through the
    two-fit path instead of ever reaching a single fit with that
    combination;
  - the pre-fit aliasing-hazard check for `between`-scenario runs (RK-3);
  - dispatching the actual fit through the R bridge (Preprocess/
    asreml_fit.R, sourced exactly the way genomic_prediction.py already
    sources models/rrBLUP.R et al.);
  - the post-condition assertions that stop a silently-wrong (or
    silently-missing) adjusted phenotype from ever reaching a model
    (RK-2);
  - substituting the converged values back into train/valid/test, by ID,
    never by position (apply_to_splits());
  - building and caching the genomic relationship matrix for the GRM
    route (R12), reusing genomic_prediction.py's own `_build_grm()`
    rather than adding a new R dependency (invariant I12).

DESIGN DECISION: BLUE CANNOT BE HELD OUT (ver4-13, Option C)
--------------------------------------------------------------------------
Discovered from a real production failure
(``log_sequential_local_20260910_140122.txt``): task 1/64 converged
cleanly in 9 ``update.asreml()`` rounds and STILL reported 43 NaN
adjusted-phenotype values, one for every masked (test) individual. Root
cause: a FIXED genotype term (BLUE, ``sparse = ~ Genotype``) can only be
estimated for a level with at least one non-missing observation, and
Requirement 8 masks EVERY row of EVERY validation/test individual's own
response before fitting - so that individual's genotype level carries
zero information into the fit, unconditionally, on every task.

Two options were considered and rejected before this one:
  - Refuse `genotype_effect='fixed'` outright whenever a holdout split
    exists (the ver4-12 behaviour) - correct, but denies BLUE to the
    person entirely, even for the training individuals it CAN legitimately
    estimate.
  - Fit once on the FULL, UNMASKED dataset so every genotype (including
    "held-out" ones) has data to estimate its own BLUE from - rejected:
    under `na.action = na.method(y = "include")` a MASKED row contributes
    nothing to the likelihood, but an unmasked one does, so every
    validation/test individual's own response would inform the shared
    Year/Experiment/residual variance components used to adjust every
    OTHER individual too - a real, if second-order, leakage channel.

The adopted design (`_fit_blue_train_plus_holdout()` below) fits TWO
independent models per task instead, each itself leakage-safe by the
existing masking/exclusion mechanism:

  Fit 1 ("BLUE, train")        : the user's own committed
      `ASREML['committed_syntax']` (genotype FIXED, unchanged), on a
      dataset PHYSICALLY RESTRICTED to training individuals only (rows
      excluded, not masked - there is no "masked partition" in this
      fit's own data at all). A genuine, unshrunk BLUE, informed only by
      training data.
  Fit 2 ("BLUP/GBLUP, holdout"): a SEPARATE committed syntax
      (`ASREML['holdout_committed_syntax']` - genotype forced RANDOM,
      optionally with a genomic relationship matrix mirroring
      `ASREML['use_grm']` - see `asreml_syntax.holdout_model_cfg()`), on
      the FULL dataset with validation/test masked exactly as the
      pre-existing single-fit path already does. Only ITS valid/test
      rows are kept (its own training-individual rows are discarded -
      Fit 1's BLUE is authoritative for those instead).

Triggered by `asreml_adjust_phenotypes()` whenever `genotype_effect ==
'fixed'`, `id_valid + id_test` is non-empty, AND `ASREML['fixed_holdout_
mode']` is `'two_fit'` (the default) rather than `'single_fit'` (see
Requirements.md item 2 / ver4-14 note just below) - there is no correct
single-fit alternative UNLESS the training and holdout partitions are
known to share genotypes, which is exactly what `fixed_holdout_mode ==
'single_fit'` asserts. The result's own `values.estimand` column is
therefore genuinely MIXED ('BLUE' for training rows, 'BLUP'/'GBLUP' for
validation/test rows) whenever the two-fit path runs - see
`asreml_syntax.estimand_description()` for the one-line, person-facing
summary of this used everywhere a banner or log line would otherwise say
a single, misleadingly-uniform estimand. This (ver4-13) yielded FOUR
possible configuration patterns in total (see `asreml_syntax.py`'s own
module docstring for the enumerated table):
  1. fixed  + GRM off -> BLUE (train) + BLUP  (holdout) - two fits
  2. fixed  + GRM on  -> BLUE (train) + GBLUP (holdout) - two fits
  3. random + GRM off -> BLUP  for everyone               - one fit (unchanged)
  4. random + GRM on  -> GBLUP for everyone               - one fit (unchanged)

ver4-14 (Requirements.md item 2) ADDS a fifth pattern, and - just as
importantly - restricts which of the five is ever LEGAL for a given
prediction SCENARIO. `asreml_syntax.ASREML_PATTERNS`/`allowed_pattern_
ids()`/`validate_scenario_compatibility()` are the single authority for
both; see that module's own section docstring for the full reasoning
(in short: patterns 1 and 3 above - the two GRM-less ones - and the new
pattern 5 all require the held-out individual's own genotype to ALSO be
observed, unmasked, somewhere in the training partition, or their result
is the SAME constant value for every held-out individual - Requirements.md
item 1's own root cause):
  5. fixed  + GRM off, fixed_holdout_mode='single_fit' -> BLUE for
     everyone, ONE fit ("the original version of BLUEs" - masks
     validation/test responses exactly as the single-fit path already
     does, but the genotype term stays FIXED throughout; legitimate only
     when SCENARIO='between' and the person has confirmed the training/
     holdout partitions share genotypes - `ASREML['between_shared_
     genotypes']=True`)

CIRCULAR-IMPORT AVOIDANCE (deliberate design note)
--------------------------------------------------------------------------
`genomic_prediction.py` imports `asreml_adjust_phenotypes`/
`apply_to_splits`/`build_or_load_grm` from this module at ITS OWN module
scope (alongside every other Preprocess.* import). Reusing
`genomic_prediction.py::_build_grm()` therefore CANNOT be a module-scope
`from genomic_prediction import _build_grm` here - that would be a genuine
circular import, evaluated while `genomic_prediction.py` is still in the
middle of being imported. `build_or_load_grm()` below imports it lazily,
INSIDE the function body, exactly mirroring the deferred-import pattern
`intra_batch_parallel.py::_run_isolated_task()` already uses for the
identical reason (see that function's own "Defect D4 fix" comment) - by
the time `build_or_load_grm()` is actually CALLED (at runtime, from inside
`GP()`'s own Step 6b), `genomic_prediction` is already a fully-initialised
module sitting in `sys.modules`, so the import succeeds trivially.

DEVIATION FROM THE DESIGN BLUEPRINT: ONE FIT PER TASK, NOT ONE PER SPLIT
--------------------------------------------------------------------------
The ver4-12 design blueprint's SS2.8 (R8) describes and diagrams a SINGLE
fit per task: mask validation+test responses together, fit ONCE, then
`predict()` every split's value from that one fit. Its SS2.10 (R10) file-
schema table, however, separately states the variance-component file's
own `split` column exists "because a different model is fitted to each
[split] (requirement 9)" - which would imply THREE separate fits. These
two passages are not reconcilable as written, and neither the Phase 1
planning brief nor `Requirements.md` was available to resolve which one
is authoritative (both are cited by the blueprint but were not among this
implementation's own attachments). Phase 2 resolves this in favour of
SS2.8's own detailed, diagrammed, acceptance-criteria-backed description -
ONE fit per task, masking `id_valid UNION id_test` together - since it is
by far the more thoroughly specified of the two, and is also what R8
itself (the requirement heading, "the whole update exists for") describes
as the actual mechanism. `AsremlResult.varcomp` therefore holds ONE set of
components per task (no `split` column at all) FOR THE SINGLE-FIT PATH;
`genomic_prediction.py`'s own Step 6b duplicates those same rows across
every split this task actually has data for (train always, valid when
non-empty, test always) when writing `ASReml_variance_components.csv`,
satisfying that file's own documented column shape without fabricating
separate fits. Recorded here, and in the ver4-12 Change Summary SS7, as a
deviation requiring reviewer confirmation against the actual Phase 1
requirement text. ver4-13's own two-fit path (above) is a SEPARATE,
narrower exception to this: it deliberately fits genotype-fixed twice
over disjoint data (a "different model per split" outcome), because
that specific combination has no single-fit solution at all - `varcomp`
in that case carries an explicit `fit` column so a reader can always
tell how many models actually ran and which one produced which rows.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import asreml_accuracy_metrics
import asreml_syntax
from pipeline_utils import r_list_get, ensure_r_native_symbols_global

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_R_FIT_FILE = os.path.join(_MODULE_DIR, 'asreml_fit.R')


def _normalise_ai_sing_mode(asreml_cfg: "Optional[dict]") -> str:
    """ver4-58 - the ONE place `ASREML['ai_sing']` is normalised to
    ``'auto'`` / ``'on'`` / ``'off'`` (``True`` -> ``'on'``, ``False`` ->
    ``'off'``, anything unrecognised -> ``'auto'``). Shared by
    `_push_ai_sing_option_to_r()` and `fit_per_set_grm_adjustment()` so the
    R side and the Python side can never disagree about the mode."""
    raw = (asreml_cfg or {}).get('ai_sing', 'auto')
    if raw is True:
        return 'on'
    if raw is False:
        return 'off'
    mode = str(raw).strip().lower()
    return mode if mode in ('auto', 'on', 'off') else 'auto'


def _borrow_variance_enabled(asreml_cfg: "Optional[dict]") -> bool:
    """ver4-59 - `ASREML['borrow_variance']` (default **True**, i.e. the
    ver4-58 behaviour). Set it to ``False`` to skip the "borrowed variance
    components" refit (every variance parameter fixed through
    `G.param`/`R.param`) and go straight to the ver4-55 `ai.sing = TRUE`
    retry when a per-set-GRM split hits an AI singularity. That refit has not
    been verified against a licensed ASReml-R, so the switch lets it be
    disabled without a code change."""
    raw = (asreml_cfg or {}).get('borrow_variance', True)
    if isinstance(raw, str):
        return raw.strip().lower() not in ('0', 'false', 'no', 'off')
    return bool(raw)


def _is_ai_singularity_error(exc: BaseException) -> bool:
    """ver4-58 - True when `exc` (or anything in its cause chain) is
    ASReml-R's "N singularities in the Average Information matrix" abort."""
    seen = 0
    cur: "Optional[BaseException]" = exc
    while cur is not None and seen < 6:
        if 'singularit' in str(cur).lower():
            return True
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return False


def _push_ai_sing_option_to_r(asreml_cfg: "Optional[dict]", override: "Optional[str]" = None) -> str:
    """ver4-55 BUGFIX - hands `ASREML['ai_sing']` to asreml_fit.R through
    the R option `easigp.asreml.ai_sing` (an option, not a new positional
    argument, so every existing rpy2 call signature stays untouched).

    Accepted values: ``'auto'`` (default; also None/True-less configs) -
    fit normally and retry ONCE with ``ai.sing = TRUE`` only if ASReml
    aborts with "N singularities in the Average Information matrix";
    ``'off'``/False - never retry (the pre-ver4-55 hard error); ``'on'``/
    True - use ``ai.sing = TRUE`` from the first attempt. Anything else
    falls back to ``'auto'``. Never raises: a failure to set the option
    just leaves R's own default (``'auto'``) in force.

    ver4-58: `override` (``'auto'``/``'on'``/``'off'``) replaces the
    configured mode for THIS call only - `fit_per_set_grm_adjustment()`
    passes ``'off'`` for its strict first attempt so a singular split
    surfaces as an error it can handle (borrow the train split's variance
    components) instead of being silently turned into a degenerate
    ``ai.sing`` fit. The option is re-pushed at the start of every fit, so
    an override never leaks into the next one.
    """
    mode = _normalise_ai_sing_mode(asreml_cfg)
    if override in ('auto', 'on', 'off'):
        mode = override
    try:
        import rpy2.robjects as _ro
        _ro.r(f'options(easigp.asreml.ai_sing = "{mode}")')
    except Exception:
        pass
    return mode

# Wrapper names (asreml_syntax.ASREML_CONSTRUCTORS) that mean "treat this
# column as a numeric/continuous covariate", never as a factor - the
# converse of every other configured column, which IS coerced to a factor
# R-side (asreml_fit.R's own "CRITICAL FIX" block) before fitting. See
# _collect_factor_and_order_columns()'s own docstring for the reasoning.
_NUMERIC_WRAPPER_NAMES = frozenset({'lin', 'pol', 'leg', 'spl', 'dev'})

# Candidate column names for ASReml-R's own predict()$pvals output -
# Appendix B check B1's own version caveat extends here too: the exact
# names ('predicted.value'/'std.error' in every ASReml-R 4 build this
# module was written against) are tried first, with a couple of
# plausible alternates as a fallback rather than a hard KeyError, so a
# minor naming difference in some other licensed build degrades to a
# clearly-logged "column not found" rather than an opaque crash.
_VALUE_COL_CANDIDATES = ('predicted.value', 'predicted_value', 'PredictedValue')
_SE_COL_CANDIDATES = ('std.error', 'std_error', 'standard.error', 'StdError')

# ver4-19 BUGFIX ("Error in parse(text = x) : <text>:1:7: unexpected
# symbol / 1: Grain Yield" - production failure, task 1/25 of
# EasiGP_BLUEs_BLUPs_ASReml_0.5, population='Cluster_2', phenotype='Grain
# Yield"): `PHENOTYPE_TRAIT_COLUMNS` accepts a trait column header
# VERBATIM (R2/R8 - see this module's docstring), and a trait literally
# called 'Grain Yield' (a space) or 'Grain Size (TKW)' (parentheses) is
# an entirely ordinary column name. Backtick-quoting it wherever THIS
# module interpolates it into R source text (`asreml_syntax.r_safe_name()`,
# still applied below, defence-in-depth) makes the text WE hand to R
# syntactically valid, but does not protect against ASReml-R's OWN
# internal machinery later re-deriving the response name as a PLAIN
# string (e.g. via `all.vars()`/`terms()`, which strip backticks by
# design) and re-parsing THAT bare string as a formula somewhere inside
# its own summary/predict pipeline - `as.formula("Grain Yield")` fails
# with this exact "<text>:1:7: unexpected symbol" text (confirmed by
# reproducing it locally), and no amount of quoting the ORIGINAL
# interpolation site can prevent a re-parse we never see. The robust fix
# is to never let a response column name containing a space or any other
# R-unsafe character reach ASReml-R at all: `_fit_single_asreml_model()`
# renames `working`'s response column to this fixed, always-bare, always-
# valid alias for the R-side fit only (see that function's own updated
# comment at the rename site) - the TRUE trait name is never needed
# R-side again, since every downstream consumer (this module's own post-
# condition checks, `apply_to_splits()`, and genomic_prediction.py's own
# `values`/`varcomp` tagging) already identifies this task's phenotype
# from Python-side bookkeeping (`trait` / `sample.loc[i, 'phenotype']`),
# never from an R-side column name.
_ASREML_SAFE_RESPONSE_COL = 'EasiGP_ASReml_Response'


# ver4-25, R8 (design blueprint SS2, Stage 6): dense N x N covariance
# matrices beyond this size are memory-mapped from disk rather than kept
# as a plain in-memory numpy array - see `_maybe_mmap_vcov()` below. Named
# so it can be read/overridden from a test without a magic literal.
_VCOV_MMAP_THRESHOLD = 5000


def _maybe_mmap_vcov(vcov: "np.ndarray", *, log_prefix: str = '[ASReml]') -> "np.ndarray":
    """ver4-25, R8 (design blueprint SS2, Stage 6): for a covariance
    matrix over more than `_VCOV_MMAP_THRESHOLD` individuals, write it to
    a PID-prefixed `.npy` file (via `Preprocess.LD_decay_plot.
    unique_path()`, so two fits landing in the same process never
    collide) under this process's own temp directory
    (`tempfile.gettempdir()` - already redirected to `SCRATCH_TMP_DIR`
    when configured, via `pipeline_utils.apply_scratch_tmp_dir()`), then
    re-opens it `mmap_mode='r'` - this is what keeps a many-thousand-
    individual 'full'-mode run's peak memory bounded (the SAME "stream/
    chunk rather than hold a whole large object in RAM" discipline the
    architecture doc's own §18 applies to `Interaction.csv`/
    `Attention.csv`). Small matrices (the common case - most tasks) are
    returned completely unchanged, as a plain in-memory array - never
    written to disk at all, so this is a strict no-op below the
    threshold.

    Never raises: on any I/O failure, logs a NOTE and returns the
    original in-memory array instead (a slower but still-correct
    fallback - a task should never fail outright just because its own
    scratch directory happened to be unwritable)."""
    vcov = np.asarray(vcov, dtype=float)
    if vcov.shape[0] <= _VCOV_MMAP_THRESHOLD:
        return vcov
    try:
        from Preprocess.LD_decay_plot import unique_path
        tmp_dir = tempfile.gettempdir()
        path = unique_path(os.path.join(tmp_dir, f"easigp_asreml_vcov_pid{os.getpid()}.npy"))
        np.save(path, vcov)
        mapped = np.load(path, mmap_mode='r')
        print(f"{log_prefix} 'full'-mode BLUE-reliability weighting: covariance matrix "
              f"({vcov.shape[0]}x{vcov.shape[0]}) exceeds {_VCOV_MMAP_THRESHOLD} individuals - "
              f"memory-mapped from '{path}' rather than held fully in RAM.")
        return mapped
    except Exception as exc:
        print(f"{log_prefix} NOTE: could not memory-map the {vcov.shape[0]}x{vcov.shape[0]} "
              f"covariance matrix ({type(exc).__name__}: {exc}) - keeping it fully in memory "
              f"instead. Never fatal, but watch memory usage for a large 'full'-mode run.")
        return vcov


@dataclass
class AsremlResult:
    """The outcome of one Step-6b fit for one task.

    Attributes
    ----------
    values : DataFrame with columns `id` (str), `split`
        (`'train'`/`'valid'`/`'test'`), `value` (the converged BLUE/BLUP),
        `estimand` (`'BLUE'`/`'BLUP'`/`'GBLUP'` - a SINGLE constant across
        every row for the pre-existing single-fit path, but genuinely
        MIXED - `'BLUE'` for `split=='train'`, `'BLUP'`/`'GBLUP'` for
        `split` in (`'valid'`, `'test'`) - whenever this came from
        `_fit_blue_train_plus_holdout()` (ver4-13, "Design decision: BLUE
        cannot be held out"); read `estimand` PER ROW, never assume a
        single value describes the whole frame). A THIRD estimand-
        producing route, `fit_per_set_grm_adjustment()` (ver4-7, R1 -
        "per-set GRM"), instead produces a UNIFORM `'GBLUP'` across every
        row, but from THREE independent fits (one per split, each with
        its own genomic relationship matrix scoped to just that split's
        own individuals - "no cross-contamination") - see that
        function's own docstring for the full merge discipline, mirrored
        from `_fit_blue_train_plus_holdout()`'s two-fit design. `std_error` (from
        `predict()$pvals`, `NaN` if unavailable) - exactly ONE row per
        individual in `id_train UNION id_valid UNION id_test` (post-
        condition asserted before this is ever returned - see
        `asreml_adjust_phenotypes()`/`_fit_blue_train_plus_holdout()`).
    varcomp : DataFrame with columns `component`, `estimate`, `std_error`,
        `bound` - one row per `summary(fit)$varcomp` term. For the
        single-fit path (this module's own "Deviation from the design
        blueprint" note), from the ONE model fitted this task, no `fit`
        column. For the two-fit BLUE(train)+BLUP/GBLUP(holdout) path, an
        ADDITIONAL leading `fit` column (`'train_blue'` /
        `'holdout_blup'` / `'holdout_gblup'`) disambiguates which of the
        two independent fits produced each row - never blend these two
        models' variance components together as if one fit produced
        them both.
    heritability : plot-basis h² = sigma_g^2 / (sigma_g^2 + sigma_e^2),
        or `None` when the genotype term was NOT fitted as random (a
        FIXED/BLUE genotype term has no genetic variance component to
        compute a heritability from), or when the genetic/residual
        variance components could not be unambiguously identified in
        `varcomp` (logged, never silently guessed). For the two-fit
        path, this is Fit 2's own heritability (the only one of the two
        fits with a genetic variance component at all).
    iterations : additional `update.asreml()` rounds beyond the initial
        fit (R9) - 0 means the first fit already satisfied `fit$converge`.
        For the two-fit path, the LARGER of the two fits' own iteration
        counts (a conservative summary scalar) - see `varcomp`'s own
        per-row `iterations` column, present only for the two-fit path,
        for the exact figure from each fit individually.
    converged : whether `fit$converge` was `True` after `iterations`
        rounds (bounded by `ASREML.max_update_rounds`). For the two-fit
        path, `True` only if BOTH fits converged - see `varcomp`'s own
        per-row `converged` column for which one, if either, did not.
    timing : ver4-25, R9-S0 (design blueprint stage 3) - `None` for a
        fit that produced this result FROM CACHE (nothing new happened);
        otherwise a dict with `role`, `scope_hash`, `n_rows`, `n_levels`,
        `seconds`, `iterations`, `converged` describing the ONE genuine R
        fit that just ran - see `_record_fit_timing()` below for how
        this is built, and `genomic_prediction.py`'s own call sites for
        how it becomes one row of `ASReml_fit_timing.csv`. Deliberately
        NOT copied forward when an `AsremlResult` is served from a
        per-trait cache (`_asreml_unmasked_cache`/`_asreml_companion_
        cache`) - see those closures' own comments - so a cache hit is
        never mistaken for a fresh (and therefore timeable) fit.
    vcov : ver4-25, R8 (design blueprint SS2, Stage 6) - the dense
        first-stage BLUE/BLUP variance-covariance matrix from
        `predict(..., vcov=TRUE)`, aligned to `vcov_ids`' own order (see
        that field), or `None` when this fit was never asked for one
        (every caller except `ASREML['blue_weighting']='full'`'s own
        Step 6b call site - I11). For more than `_VCOV_MMAP_THRESHOLD`
        individuals, this is a read-only, memory-mapped `numpy.ndarray`
        backed by a PID-prefixed `.npy` file under the process's own
        temp directory (`tempfile.gettempdir()` - respects
        `SCRATCH_TMP_DIR` when configured, via `pipeline_utils.
        apply_scratch_tmp_dir()`) rather than a plain in-memory array -
        see `_maybe_mmap_vcov()` below. **Unverified against a licensed
        ASReml-R install** - see the ver4-25 Change Summary's own
        disclosure for `full`-mode weighting.
    vcov_ids : the id order `vcov`'s rows/columns are aligned to - the
        SAME order as `values`'s own `id` column for this fit (ASReml-R's
        `predict(..., vcov=TRUE)` documented contract: the returned
        matrix is aligned to `predict()`'s own row order) - `None`
        whenever `vcov` is `None`.
    heritability_classic : classic (entry-mean basis) heritability
        `sigma_g^2 / (sigma_g^2 + sigma_e^2 / r_bar [+ sigma_ge^2 /
        e_bar])` computed from THIS fit's own variance components and
        the records this fit actually used (see `classic_heritability_
        from_fit()`) - the SAME formula `fit_concurrent_variance_model()`
        reports as `H2_classic`, evaluated on this fit so it can sit
        beside the Cullis heritability computed from this same fit
        (`Metric.csv`'s `Heritability (classic)` column). `None` whenever
        it cannot be computed (a FIXED/BLUE genotype term, an
        unidentifiable variance component, ...) - never fatal, never
        fabricated. Only populated by `fit_unmasked_adjustment()`;
        defaults to `None` for every other producer, so this trailing,
        optional field changes nothing for any existing caller.
    ai_sing_used : ver4-58 - True when the fit only succeeded after the
        `ai.sing = TRUE` retry (ver4-55): at least one variance parameter
        was NOT identifiable from this fit's data and was left at a
        boundary/fixed value, so the fit's variance components are NOT a
        regular REML estimate. Default False.
    borrowed_variance : ver4-58 - True when EVERY variance parameter of
        this fit was held fixed at the values estimated on a larger,
        healthy fit (the per-set-GRM train split) because the split's own
        data could not identify them (`fit_unmasked_adjustment(
        fixed_vparameters=...)`). Default False.
    """
    values: "pd.DataFrame"
    varcomp: "pd.DataFrame"
    heritability: "Optional[float]"
    iterations: int
    converged: bool
    timing: "Optional[dict]" = None
    vcov: "Optional[np.ndarray]" = None
    vcov_ids: "Optional[Sequence[str]]" = None
    heritability_classic: "Optional[float]" = None
    ai_sing_used: bool = False
    borrowed_variance: bool = False


def restrict_vcov_to_population(
    result: "AsremlResult", population_test: "Optional[str]",
) -> "tuple":
    """ver4-25, R8 (design blueprint SS2, Stage 6): resolve
    `result.vcov`/`result.vcov_ids` down to a plain `(vcov_matrix,
    id_list)` pair a `blue_weights.BlueWeightContext` can consume,
    mirroring `genomic_prediction.py`'s own `_asreml_values_lookup()`/
    `_asreml_se_lookup()` convention exactly: when `result.values` carries
    a `population` column (`response_unit == 'genotype_by_population'`),
    restrict to `population_test` first (or, when `population_test` is
    `None`, keep each id's FIRST occurrence only - the same
    `drop_duplicates(subset='id', keep='first')` those two functions
    already use) - since a plain id is not unique across populations
    otherwise. Returns `(None, None)` when `result.vcov` is `None` (no
    vcov was ever requested for this fit).

    Never raises: an id appearing more than once after restriction (should
    not happen - `fit_unmasked_adjustment()`'s own post-condition already
    guards this) is defended against by keeping only the first occurrence,
    matching the two lookup functions' own `keep='first'` convention.
    """
    if result.vcov is None or result.vcov_ids is None:
        return None, None
    vcov_ids = result.vcov_ids
    if vcov_ids and isinstance(vcov_ids[0], tuple):
        # by-population: vcov_ids is a list of (id, population) tuples.
        if population_test is not None:
            positions = [i for i, (_id, _pop) in enumerate(vcov_ids) if str(_pop) == str(population_test)]
        else:
            seen = set()
            positions = []
            for i, (_id, _pop) in enumerate(vcov_ids):
                if str(_id) not in seen:
                    seen.add(str(_id))
                    positions.append(i)
        ids_out = [str(vcov_ids[i][0]) for i in positions]
        vcov_out = np.asarray(result.vcov)[np.ix_(positions, positions)] if positions else np.zeros((0, 0))
        return vcov_out, ids_out
    return np.asarray(result.vcov), [str(v) for v in vcov_ids]


def _normalise_group(group) -> dict:
    """Mirrors asreml_syntax._normalise_group() exactly (kept as an
    independent copy, not an import of that leading-underscore helper,
    since asreml_syntax.py's own "zero project imports" discipline runs
    the OTHER direction here too - this module already imports FROM
    asreml_syntax, so reaching back into its private internals would be
    an upward/sideways private dependency this codebase avoids
    elsewhere)."""
    if isinstance(group, dict):
        return {'columns': list(group.get('columns', [])), 'target': group.get('target', 'random')}
    return {'columns': list(group), 'target': 'random'}


def _collect_factor_and_order_columns(asreml_cfg: dict):
    """Every column this task's model formula references (fixed, random,
    genotype, interaction/nested group members, residual dimensions) -
    split into `factor_cols` (coerced to an R factor before fitting -
    asreml_fit.R's own "CRITICAL FIX" block) and `order_cols` (the
    residual structure's own dimension(s), which `dat` must be SORTED by
    before fitting whenever an ordered residual structure like
    `ar1(Row):ar1(Range)` is configured - the reference scripts' own
    `dat[order(dat$Row, dat$Range), ]` step).

    A column wrapped with a NUMERIC constructor (`lin`/`pol`/`leg`/`spl`/
    `dev` - `_NUMERIC_WRAPPER_NAMES`) is deliberately EXCLUDED from
    `factor_cols`, since these constructors fit a continuous regression
    on the column's own numeric value, not a discrete-level factor.

    The base column set (before the numeric-wrapper exclusion) is
    `asreml_syntax.model_term_columns(asreml_cfg)` - the single authority
    for "which columns does this model's formula actually reference",
    also read by `asreml_syntax.validate_structure()` to check a
    candidate `output_level` before any syntax is ever committed. Reused
    here rather than re-derived, per the `_map_genes_to_markers`
    precedent this codebase applies to every such cross-cutting fact.
    """
    wrappers_cfg = asreml_cfg.get('wrappers', {}) or {}

    all_cols = set(asreml_syntax.model_term_columns(asreml_cfg))

    residual_cfg = asreml_cfg.get('residual')
    order_cols = list(residual_cfg.get('dims', [])) if residual_cfg else []

    factor_cols = sorted(
        c for c in all_cols
        if (wrappers_cfg.get(c) or {}).get('name') not in _NUMERIC_WRAPPER_NAMES
    )
    return factor_cols, order_cols


def _check_fixed_genotype_estimability(id_masked: Sequence[str], genotype_effect: str,
                                        genotype_col: "Optional[str]", log_prefix: str,
                                        *, allow: bool = False) -> None:
    """RK-9 - DEFENSIVE BACKSTOP ONLY as of ver4-13 (see this module's own
    top-of-file docstring, "Design decision: BLUE cannot be held out").

    Originally (ver4-12) this was the PRIMARY defence: raise before ever
    invoking R whenever a FIXED genotype term was paired with a masked
    individual, since that combination is unconditionally inestimable -
    discovered from a real production run
    (``log_sequential_local_20260910_140122.txt``: task 1/64 converged
    cleanly in 9 update.asreml() rounds and still reported 43 NaN
    values). ver4-13 changed the PUBLIC entry point
    (`asreml_adjust_phenotypes()`) to route that exact combination
    through `_fit_blue_train_plus_holdout()` instead of ever reaching
    `_fit_single_asreml_model()` with it - so under normal operation this
    check should never fire at all: `_fit_blue_train_plus_holdout()`'s
    own Fit 1 always has `id_masked=[]` (training individuals only,
    physically excluded rather than masked) and its own Fit 2 always has
    `genotype_effect='random'` (`asreml_syntax.holdout_model_cfg()`
    forces it). Kept here as a second, independent layer of defence -
    the same "never trust, always verify" discipline RK-2's own post-
    condition checks apply - in case the dispatcher above is ever
    bypassed or modified incorrectly; if this DOES fire, that is a bug
    in the dispatch logic, not a normal/expected outcome.

    A no-op whenever `id_masked` is empty (nothing is held out, so
    nothing is unobserved) or `genotype_effect` is not `'fixed'` - a
    RANDOM genotype term (BLUP or GBLUP) has a variance-component prior
    that still yields a well-defined value for an unobserved level (it
    shrinks fully to the population mean with no relationship matrix, or
    borrows from related individuals through one with `use_grm=True`).

    `allow=True` (Requirements.md item 2 - `ASREML['fixed_holdout_mode']
    == 'single_fit'`, "the original version of BLUEs") is the ONE other
    intentional way this stays a no-op despite a masked individual being
    present: `asreml_syntax.validate_scenario_compatibility()` only ever
    permits that mode when `SCENARIO='between'` AND the person has
    confirmed the training/holdout partitions share genotypes - i.e. a
    "held-out" individual's genotype level DOES already have at least
    one non-missing observation elsewhere in the (unmasked) training
    partition of this same fit, so the term is genuinely estimable after
    all, unlike the unconditional case this check otherwise guards
    against.
    """
    if not id_masked or genotype_effect != 'fixed' or allow:
        return
    raise ValueError(
        f"{log_prefix} ASREML['genotype_effect']='fixed' (estimand 'BLUE') cannot produce an "
        f"adjusted phenotype for {len(id_masked)} validation/test individual(s) in this task. "
        f"A FIXED genotype term (fitted via `sparse = ~ {genotype_col}`) can only be estimated "
        f"for a level with at least one non-missing response, and Requirement 8 has just masked "
        f"EVERY row of EVERY validation/test individual's own response to NA - leaving that "
        f"individual's genotype level with zero information. This is NOT an aliasing accident "
        f"(RK-3) or a convergence problem: the fit can converge cleanly and still report this, "
        f"because convergence and estimability are different questions. By design, "
        f"`asreml_adjust_phenotypes()` should have routed this combination through the two-fit "
        f"`_fit_blue_train_plus_holdout()` path instead of ever reaching here with a masked "
        f"individual present - seeing this error means that dispatch logic itself has a bug; "
        f"please report it rather than working around it by disabling ASREML['use_grm'] or "
        f"switching genotype_effect."
    )


def _check_aliasing_hazard(working: "pd.DataFrame", id_train: Sequence[str], id_masked: Sequence[str],
                            effect_columns: Sequence[str], log_prefix: str) -> None:
    """RK-3 (design blueprint SS2.8, "Aliasing hazard"): for a `between`-
    scenario run, masking an ENTIRE population removes every observation
    at that level for any effect column that happens to be constant
    within the masked partition and absent from the training partition -
    that effect becomes inestimable (aliased), and ASReml's own behaviour
    in that case (an `NA`/aliased estimate) would otherwise reach Step
    6b's post-condition checks as a hard-to-diagnose failure far from its
    actual cause. Detected HERE, before any fit is attempted, and raised
    with the offending column and value(s) named explicitly.

    A no-op for `within`-scenario runs in practice (masking a random
    subset of individuals essentially never leaves an effect column
    perfectly separated like this) - this function does not need to know
    which scenario is in effect; it simply finds nothing to report in
    that case.
    """
    if not id_masked:
        return
    train_mask = working['ID'].isin(set(id_train))
    masked_mask = working['ID'].isin(set(id_masked))
    problems = []
    for col in effect_columns:
        if col not in working.columns:
            continue
        train_values = set(working.loc[train_mask, col].dropna().unique().tolist())
        masked_values = set(working.loc[masked_mask, col].dropna().unique().tolist())
        if len(masked_values) == 1 and masked_values and masked_values.isdisjoint(train_values):
            problems.append((col, next(iter(masked_values))))
    if problems:
        detail = '; '.join(f"{col!r} is fixed at {value!r} for every masked individual, and "
                            f"{value!r} never appears in the training partition" for col, value in problems)
        raise ValueError(
            f"{log_prefix} Aliasing hazard detected before fitting: {detail}. This effect is "
            f"INESTIMABLE for the masked partition (every masked individual shares one level "
            f"never seen in training) - review this task's population/effect configuration "
            f"(design blueprint RK-3). Common cause: a population-defining column is ALSO "
            f"fitted as a model effect in a `between`-scenario run."
        )


def _extract_estimand_column(pvals: "pd.DataFrame", candidates: Sequence[str], what: str,
                              log_prefix: str) -> "pd.Series":
    for name in candidates:
        if name in pvals.columns:
            return pvals[name]
    raise KeyError(
        f"{log_prefix} predict()$pvals has none of the expected {what} column name(s) "
        f"{list(candidates)} - available columns: {list(pvals.columns)}. This indicates a "
        f"different ASReml-R API generation than this module was written against (Appendix B "
        f"check B1) - report this rather than guessing which "
        f"column to use."
    )


def _estimate_heritability(varcomp: "pd.DataFrame", genotype_col: "Optional[str]",
                            genotype_effect: str):
    """Plot-basis h^2 = sigma_g^2 / (sigma_g^2 + sigma_e^2) - see this
    module's own AsremlResult docstring for when this is None.

    Component-name matching is a best-effort heuristic (component labels
    are ASReml-R's own free-text term descriptions, e.g.
    `'vm(Genotype, source = Ginv)!Genotype'` or `'Genotype'` for the
    genetic term, and typically `'units!R'`/`'units!units'` for the
    residual) - genotype_col's own name is searched for (case-
    insensitively) among the non-residual rows, and the residual variance
    is taken as whichever remaining row's own component name contains
    'units' (ASReml's standard residual-term naming), falling back to the
    LAST row (ASReml conventionally lists the residual last) if no
    'units'-named row is found.

    Returns
    -------
    `(heritability, genetic_row_index)` - `(None, None)` if either
    component cannot be identified (never raises, never fabricates a
    value - logged by the caller, not silently swallowed).
    `genetic_row_index` is `varcomp`'s own row label for the identified
    genetic-variance term, so the caller can populate a `'heritability'`
    column on THAT row only (design blueprint SS2.10's own file schema:
    "populated only on the genotype-variance row").
    """
    if genotype_effect != 'random' or not genotype_col or varcomp is None or varcomp.shape[0] == 0:
        return None, None
    try:
        components = varcomp['component'].astype(str)
        estimates = varcomp['estimate'].astype(float)
    except (KeyError, ValueError):
        return None, None

    genetic_mask = components.str.contains(genotype_col, case=False, regex=False)
    if not genetic_mask.any():
        return None, None
    genetic_row_index = estimates[genetic_mask].index[0]
    genetic_variance = float(estimates.loc[genetic_row_index])

    residual_mask = components.str.contains('units', case=False, regex=False) & ~genetic_mask
    if residual_mask.any():
        residual_variance = float(estimates[residual_mask].iloc[0])
    else:
        remaining = estimates[~genetic_mask]
        if remaining.empty:
            return None, None
        residual_variance = float(remaining.iloc[-1])

    denominator = genetic_variance + residual_variance
    if denominator <= 0:
        return None, None
    return genetic_variance / denominator, genetic_row_index


def _classify_vparameter(component_raw: "Optional[str]", genotype_col: "Optional[str]") -> str:
    """Canonical, human-stable name for one `summary(fit)$varcomp` row's
    own free-text `component` label - the SINGLE classifier both
    `_estimate_heritability()` (R5, plot-basis h^2) and `fit_concurrent_
    variance_model()`'s canonical varcomp schema (ver4-25, R10) share, so
    the SAME raw ASReml term is never named two different ways in two
    different outputs (the same single-authority discipline `model_
    registry.py`/`circos_geometry.py` already apply elsewhere - I8).

    Returns one of:
      `'genotype'`            - the genotype term itself (raw label
                                 contains `genotype_col`, case-
                                 insensitively, with no OTHER term
                                 combined into it via `:`) - matches both
                                 `vm(Genotype, source = Ginv)!Genotype`
                                 and a plain random `Genotype` term.
      `'residual'`            - ASReml's own residual term (contains
                                 `'units'`, and does not also contain
                                 `genotype_col` - a genotype:units
                                 combination should not occur, but is
                                 treated as the residual is more
                                 informative than a genotype label here).
      `'genotype_by_<factor>'` - an INTERACTION between `genotype_col`
                                 and EXACTLY one other model term (e.g.
                                 `'Genotype:Environment'` ->
                                 `'genotype_by_Environment'`) - `<factor>`
                                 is that other term's own raw name, with
                                 every non-alphanumeric character
                                 replaced by `'_'` for a clean, column-
                                 name-safe suffix.
      `component_raw` itself   - anything else (an ungrouped random/
                                 fixed term, or a genotype interaction
                                 with more than one other term) - left
                                 verbatim, unclassified.

    Never raises - a malformed/empty `component_raw` (or a missing
    `genotype_col`) classifies as itself (or as `'residual'`, if it looks
    like ASReml's own residual naming).
    """
    if not component_raw:
        return component_raw
    label = str(component_raw)
    if not genotype_col:
        return 'residual' if 'units' in label.lower() else label
    lower = label.lower()
    geno_lower = str(genotype_col).lower()
    if 'units' in lower and geno_lower not in lower:
        return 'residual'
    if geno_lower not in lower:
        return label
    parts = [p.strip() for p in label.split(':')]
    other_parts = [p for p in parts if geno_lower not in p.lower()]
    if not other_parts:
        return 'genotype'
    if len(other_parts) == 1:
        factor = re.sub(r'[^0-9a-zA-Z]+', '_', other_parts[0]).strip('_')
        return f'genotype_by_{factor}' if factor else label
    return label


def _fit_scope_hash(trait: str, id_train: Sequence[str], id_valid: Sequence[str],
                     id_test: Sequence[str]) -> str:
    """Short, deterministic identifier for "which exact fit is this" -
    used purely as `ASReml_fit_timing.csv`'s own `scope_hash` column
    (ver4-25, R9-S0), so two rows with the same `(role, scope_hash)` can
    be recognised as the SAME underlying fit re-logged (e.g. by
    different model groups under an ASReml + model-fanout run - see
    `checkpoint_utils.MODEL_GROUP_MERGE_SPEC['asreml_fit_timing']`),
    without needing to compare full ID lists by eye. Never raises."""
    payload = f"{trait}|{sorted(str(v) for v in id_train)}|{sorted(str(v) for v in id_valid)}|" \
              f"{sorted(str(v) for v in id_test)}"
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]


def _record_fit_timing(role: str, scope_hash: str, n_rows: int, n_levels: int,
                        seconds: float, iterations: int, converged: bool) -> dict:
    """Builds the plain dict `AsremlResult.timing` carries - one R fit
    CALL's own cost accounting (ver4-25, R9-S0). A pure, side-effect-free
    constructor; the caller (genomic_prediction.py) is responsible for
    turning this into an actual `ASReml_fit_timing.csv` row (this
    module's own "pure helpers stay pure" discipline)."""
    return {
        'role': role, 'scope_hash': scope_hash, 'n_rows': int(n_rows), 'n_levels': int(n_levels),
        'seconds': float(seconds), 'iterations': int(iterations), 'converged': bool(converged),
    }


def _fit_single_asreml_model(
    raw_records: "pd.DataFrame", trait: str,
    id_train: Sequence[str], id_valid: Sequence[str], id_test: Sequence[str],
    asreml_cfg: dict, committed_syntax: str,
    grm_inv=None, workspace: str = '4gb',
    max_update_rounds: int = 20, log_prefix: str = '[ASReml]',
    *, allow_fixed_with_masked: bool = False, grm_ids: "Optional[Sequence[str]]" = None,
    role: str = 'unspecified', threads: "Optional[float]" = None,
    return_vcov: bool = False,
) -> AsremlResult:
    """Fit ONE mixed model (R5/R8), iterate to convergence (R9), and
    return the converged per-individual value plus variance components -
    see this module's own docstring and `AsremlResult`'s for the full
    contract.

    This is the ONE-FIT implementation. The public entry point,
    `asreml_adjust_phenotypes()` below, calls this DIRECTLY whenever
    `genotype_effect` is `'random'` (patterns 3-4 of the module
    docstring's four-pattern table - unchanged since ver4-12), and calls
    it TWICE internally, with two different (`id_masked`, `asreml_cfg`,
    `committed_syntax`) triples, from `_fit_blue_train_plus_holdout()`
    when `genotype_effect` is `'fixed'` (patterns 1-2, ver4-13 - see this
    module's own top-of-file "Design decision" section). It is never
    aware of which of those two callers it is serving - it just fits the
    one model it is given.

    Parameters
    ----------
    raw_records : this task's own individuals' RAW (potentially plot-
        level, many-rows-per-individual) phenotype records - already
        filtered to `id_train UNION id_valid UNION id_test` by the caller
        (genomic_prediction.py::GP()), carrying every design column
        needed by `asreml_cfg`'s fixed/random/residual/interaction/nested
        terms, plus `trait` and `'ID'`. Never mutated - a working copy is
        masked internally.
    trait : this task's own phenotype/trait column name.
    id_train, id_valid, id_test : this task's own split membership (each
        a plain sequence of ID strings; `id_valid` may be empty).
    asreml_cfg : the resolved `ASREML` config dict.
    committed_syntax : the GUI-committed (or hand-written) ASReml-R call
        text, still containing the `{TRAIT}`/`{DATA}`/(`{GRM}`)
        placeholders - substituted here, once, for this task's own trait.
    grm_inv : optional pre-built genomic relationship matrix, row/column-
        ALIGNED to `grm_ids` (see that parameter) when given, or to
        `id_train + id_valid + id_test` otherwise (the pre-ver4-19
        behaviour, kept as the default so an older caller that never
        passes `grm_ids` is unaffected). `None` when `ASREML['use_grm']`
        is False.
    grm_ids : ver4-19, Requirements.md R1 fix - the ACTUAL id order
        `grm_inv`'s rows/columns are aligned to, when that differs from
        `id_train + id_valid + id_test` (e.g. `genomic_prediction.py`'s
        Step 6b widening the GRM to cover every genotype id present in
        the fitted grid, not just this task's own individuals - see
        that module's own Step 6b call site and
        `resolve_grm_genotype_universe()` below). `None` (the default)
        preserves the pre-ver4-19 assumption that the GRM's own id order
        IS `id_train + id_valid + id_test` - every pre-existing caller
        that never passes this keeps working unchanged.
    log_prefix : prefix for every log line this function prints - never
        changed by any current caller, exposed purely so a future caller
        (e.g. a standalone CLI validator) can distinguish its own output.
    role : ver4-25, R9-S0 - a short label naming WHICH kind of fit this
        is (e.g. `'legacy_masked'`, `'concurrent_iid'`) - used purely to
        populate the returned `AsremlResult.timing['role']`; never
        affects the fit itself. Defaulted so no pre-ver4-25 caller is
        affected (I11).
    allow_fixed_with_masked : Requirements.md item 2 - set `True` ONLY by
        `asreml_adjust_phenotypes()`'s own `fixed_holdout_mode ==
        'single_fit'` branch ("the original version of BLUEs"), where a
        FIXED genotype term IS estimable for a masked individual because
        that individual's genotype level also has a non-missing
        observation elsewhere in this SAME fit's own (unmasked) training
        rows - see `_check_fixed_genotype_estimability()`'s own docstring
        for the full reasoning. Every other caller leaves this `False`.
    return_vcov : ver4-25, R8 (design blueprint SS2, Stage 6) - request
        the dense first-stage variance-covariance matrix of the
        predicted values (`predict(..., vcov=TRUE)`) in the SAME R call
        that already produces `pvals` - `False` (the default - every
        pre-ver4-25 caller) is a byte-identical R call (I11). When
        `True`, the returned `AsremlResult.vcov`/`vcov_ids` are
        populated; see `AsremlResult`'s own docstring for their exact
        shape/alignment and `_maybe_mmap_vcov()` for the large-matrix
        handling. **Unverified against a licensed ASReml-R install** in
        this delivery (see the ver4-25 Change Summary's own disclosure).

    Raises
    ------
    ValueError if the raw records are empty, the committed syntax lacks
        `{TRAIT}`, a FIXED genotype term is paired with a non-empty
        validation/test split (RK-9 - inestimable by construction, see
        `_check_fixed_genotype_estimability()`), an aliasing hazard is
        detected (RK-3), or a post-condition assertion fails (RK-2 -
        every ID covered exactly once, no NaN reaching a non-empty
        split). RuntimeError if the R fit itself raises (surfaced with
        the R-side error text, prefixed `log_prefix`) - the caller's own
        `ASREML['on_nonconvergence']` policy decides whether this aborts
        the run or skips the task.
    """
    
    id_train = [str(v) for v in id_train]
    id_valid = [str(v) for v in id_valid]
    id_test = [str(v) for v in id_test]
    id_masked = id_valid + id_test
    all_ids = id_train + id_masked

    if not committed_syntax or '{TRAIT}' not in committed_syntax:
        raise ValueError(
            f"{log_prefix} committed_syntax is empty or missing the {{TRAIT}} placeholder - "
            f"commit a syntax from the GUI's ASReml section (design blueprint SS2.7) before "
            f"enabling ASREML."
        )

    # ver4-17 BUGFIX ("Data has N observations but the residual model
    # implies M" - see this module's own top-of-file docstring for the
    # full production-failure account): whenever this task's residual
    # structure needs a complete factorial design over its own dimensions
    # (`asreml_syntax.residual_requires_full_grid()` - the single
    # authority both this function AND genomic_prediction.py's Step 6b
    # consult, so neither side can drift from the other on this exact
    # question), `raw_records` must NOT be restricted to this task's own
    # `all_ids` - that restriction is exactly what leaves gaps in an
    # ordered dimension (e.g. field Range/Row) whenever `population` is a
    # genetically-DERIVED subset scattered across a SHARED trial, rather
    # than a literal per-experiment population. Computed here (a
    # defensive backstop, independent of whichever table the caller
    # actually passed in - the same "never trust, always verify"
    # discipline RK-2/RK-9 already apply elsewhere in this module) BEFORE
    # `working` is built, so the one-time row-selection decision below is
    # made once, consistently, from `asreml_cfg` alone.
    _needs_full_grid = asreml_syntax.residual_requires_full_grid(asreml_cfg)

    working = raw_records.copy()
    if _needs_full_grid:
        # ver4-18 BUGFIX - defensive backstop, independent of whichever
        # table the caller actually passed in (this module's own "never
        # trust, always verify" discipline, mirrored from RK-2/RK-9's
        # existing precedent - see asreml_syntax.pad_residual_grid()'s
        # own docstring for the full mechanics). genomic_prediction.py's
        # own Step 6b already calls this ONCE, up front, for the whole
        # run - this call is IDEMPOTENT on an already-padded table (finds
        # zero missing combinations, returns it unchanged), so it costs
        # nothing extra in the normal case and only actually does work
        # for a caller that bypassed that up-front step.
        working = asreml_syntax.pad_residual_grid(working, asreml_cfg)
    working['ID'] = working['ID'].astype(str)
    if _needs_full_grid:
        # Every row raw_records ever had is kept - including individuals
        # entirely OUTSIDE this task (`all_ids`) - so an ordered residual
        # dimension's own levels (e.g. field Range/Row) still cover the
        # trial's full physical layout, not just the plots this task's
        # own population happens to occupy. Their own response is left
        # UNTOUCHED (neither masked nor removed) just below - they are
        # never this task's own train/valid/test individuals, so they
        # are never evaluated, never reach `predict()$pvals` post-
        # filtering (see the `values[values['id'].isin(all_ids)]` step
        # further down, unchanged since ver4-12), and this is therefore
        # not a leakage channel for THIS task's own genomic-prediction
        # accuracy - it is exactly the standard field-trial practice of
        # fitting spatial/design effects from the WHOLE trial before
        # extracting one subset's own adjusted values. A `grm_source=
        # 'task_pool'` GRM (built from `all_ids` only) will not cover
        # these individuals' genotype level when a vm() term is in play;
        # `asreml_fit.R` already handles that gracefully via this
        # integration's own pre-existing `na.action = na.method(x =
        # "include")` (present in every committed syntax this module
        # was reviewed against) - see that file's own coercion-loop
        # comment for the mechanics.
        working = working.reset_index(drop=True)
    else:
        working = working[working['ID'].isin(all_ids)].reset_index(drop=True)
    if working.shape[0] == 0:
        raise ValueError(
            f"{log_prefix} No raw phenotype record found for any of this task's {len(all_ids)} "
            f"individual(s) - cannot fit."
        )

    # ver4-19 BUGFIX ("Error in parse(text = x) : ... unexpected symbol /
    # 1: Grain Yield" - see this module's own top-of-file note next to
    # `_ASREML_SAFE_RESPONSE_COL` for the full root-cause explanation).
    # `working`'s response column is renamed, R-side only, to that fixed
    # bare alias for the rest of THIS function - every reference to the
    # response column below (masking, the `{TRAIT}` substitution, and the
    # `response_col` argument handed to `asreml_fit_and_predict()`) uses
    # the alias, never `trait` itself, so a name with a space/parenthesis/
    # reserved word never reaches R in any form, quoted or not. Applied
    # unconditionally - including to an already-bare trait like
    # 'Maturity' - so this is one code path, always exercised, rather
    # than a conditional branch that only gets tested on the rare unsafe
    # name. `raw_records`/the caller's own `trait` variable are
    # untouched; only this function's local `working` copy is renamed.
    if (_ASREML_SAFE_RESPONSE_COL in working.columns
            and _ASREML_SAFE_RESPONSE_COL != trait):
        raise ValueError(
            f"{log_prefix} The phenotype file already has a column literally named "
            f"{_ASREML_SAFE_RESPONSE_COL!r}, which this ASReml integration reserves "
            f"internally for the response variable during a fit - rename that column in the "
            f"phenotype file and try again."
        )
    working = working.rename(columns={trait: _ASREML_SAFE_RESPONSE_COL})
    response_col = _ASREML_SAFE_RESPONSE_COL

    # Requirement 8: mask valid+test responses to NA in EVERY raw row for
    # a masked individual (a plot-level file may carry several rows per
    # individual) - every OTHER column (every design column) is left
    # completely untouched, so the design stays balanced and any AR1
    # spatial grid stays complete. Only THIS task's own `id_masked`
    # (`id_valid + id_test`) is masked here - an individual outside
    # `all_ids` entirely (only present in `working` at all when
    # `_needs_full_grid` above kept them) is not masked, per that
    # branch's own comment.
    masked_row_mask = working['ID'].isin(set(id_masked))
    working.loc[masked_row_mask, response_col] = np.nan

    genotype_col = asreml_cfg.get('genotype_col')
    _check_fixed_genotype_estimability(
        id_masked, asreml_cfg.get('genotype_effect', 'random'), genotype_col, log_prefix,
        allow=allow_fixed_with_masked,
    )
    effect_columns = set(asreml_cfg.get('fixed', []) or []) | set(asreml_cfg.get('random', []) or [])
    effect_columns.discard(genotype_col)  # genotype itself is EXPECTED to differ - not an aliasing signal
    _check_aliasing_hazard(working, id_train, id_masked, sorted(effect_columns), log_prefix)

    factor_cols, order_cols = _collect_factor_and_order_columns(asreml_cfg)
    output_level = list(asreml_cfg.get('output_level', []) or [])
    if genotype_col not in output_level:
        raise ValueError(
            f"{log_prefix} ASREML['genotype_col']={genotype_col!r} must be included in "
            f"ASREML['output_level']={output_level!r} - predict(classify=) needs it to report "
            f"one value per genotype (asreml_syntax.validate() should have caught this earlier)."
        )

    # Second, independent layer of defence for the SAME check
    # `asreml_syntax.validate_structure()` performs before a syntax is
    # ever committed (R6/R7) - catches a hand-edited/headless config that
    # bypassed the GUI entirely, or a syntax committed before the model's
    # own effects were last edited. Every `output_level` entry other than
    # the genotype column must be an actual model term (a fixed effect,
    # random effect, interaction/nested group member, or residual
    # dimension - `asreml_syntax.model_term_columns()` is the single
    # authority for that set, shared with `_collect_factor_and_order_
    # columns()` just above). A column that merely EXISTS in the
    # phenotype file but was never given to the model cannot be
    # classified on: `predict(classify=...)` fails deep inside R with
    # "<column> is not in the model, unable to predict using it" - raised
    # HERE instead, before R is ever invoked, naming the offending
    # column(s) explicitly rather than leaving that R-side message as the
    # only diagnostic.
    _model_cols = asreml_syntax.model_term_columns(asreml_cfg)
    _stray_output_level = [c for c in output_level if c != genotype_col and c not in _model_cols]
    if _stray_output_level:
        raise ValueError(
            f"{log_prefix} ASREML['output_level']={output_level!r} includes column(s) "
            f"{_stray_output_level!r} that are not a fixed effect, random effect, "
            f"interaction/nested group member, or residual-structure dimension in this model - "
            f"predict(classify=...) cannot report a level for a column that is not actually IN "
            f"the model (asreml_syntax.validate() should have caught this earlier). Either add "
            f"{_stray_output_level!r} as a model effect, or remove it from "
            f"ASREML['output_level']."
        )

    # ver4-19 BUGFIX - `{TRAIT}` is substituted with `response_col` (the
    # fixed, always-bare `_ASREML_SAFE_RESPONSE_COL` alias `working` was
    # already renamed to above), never the original `trait` text - see
    # this module's own top-of-file note next to `_ASREML_SAFE_RESPONSE_
    # COL` for the full root-cause explanation of why quoting `trait`
    # here alone (a previous, insufficient attempt at this same fix) does
    # not actually protect against ASReml-R's own internal re-derivation
    # of the response name. `asreml_syntax.r_safe_name()` is still applied
    # as cheap, harmless defence-in-depth (the alias is already a bare
    # valid identifier, so this is a no-op in practice) rather than
    # removed outright, since every other formula-text interpolation
    # point in this codebase (render_syntax()/_apply_wrapper()) follows
    # the same "always route a column name through r_safe_name()" rule.
    call_text = committed_syntax.replace(
        '{TRAIT}', asreml_syntax.r_safe_name(response_col)
    ).replace('{DATA}', 'dat')
    if grm_inv is not None:
        call_text = call_text.replace('{GRM}', 'Ginv')

    import rpy2.robjects as robjects
    # BUGFIX (undefined symbol: R_NilValue) - asreml_fit.R's own
    # `library(asreml)` (a top-level statement, re-run every time this
    # file is re-sourced - cheap/idempotent once R has actually cached
    # the namespace as loaded) is the ONE dyn.load() in this whole
    # codebase that needs libR.so's own C-API symbols to already be
    # GLOBALLY visible in this embedded-R-inside-Python process - see
    # pipeline_utils.ensure_r_native_symbols_global()'s own docstring for
    # the full root-cause explanation and why the four pre-existing R
    # models (BGLR) never exposed this. Called again here, immediately
    # before the one call site that actually needs it, rather than
    # trusting configure_r_environment()'s own earlier call alone -
    # cheap and idempotent either way (a repeat dlopen() of an
    # already-loaded object is a fast reference-count bump, not a
    # re-load), and this is the highest-value place for it: the single
    # dyn.load() this entire ASReml integration depends on.
    ensure_r_native_symbols_global()
    r_source = robjects.r['source']
    r_source(_R_FIT_FILE)
    _push_ai_sing_option_to_r(asreml_cfg)
    asreml_fit_and_predict = robjects.globalenv['asreml_fit_and_predict']

    # BUGFIX (ver4-15, "'source' is missing a 'dimnames' attribute"):
    # `grm_inv` is passed RAW here - `asreml_prepare_ginv()` (which attaches
    # row/column names and the ASReml-R 4.x INVERSE attribute) is now
    # called INSIDE `asreml_fit_and_predict()` itself (see asreml_fit.R's
    # own top-of-function comment for the full root-cause explanation).
    # Calling `asreml_prepare_ginv()` from THIS side of the Python<->R
    # boundary - as a previous revision did - meant its R-object return
    # value was converted straight back into a plain `numpy.ndarray` by
    # rpy2's process-wide `numpy2ri` converter (installed by
    # `pipeline_utils.init_rpy2_conversion()`), which silently discards
    # dimnames and arbitrary R attributes; `Ginv` then reached
    # `vm(<genotype col>, source = Ginv)` with no dimnames at all. Passing
    # the raw matrix and letting R attribute it, consume it, and never
    # hand it back to Python keeps those attributes intact.
    ginv_ids_r = robjects.NULL
    if grm_inv is not None:
        # ver4-19, Requirements.md R1 fix: `grm_inv`'s own row/column order
        # is `grm_ids` when the caller supplied it (Step 6b's genotype-
        # universe widening - see this function's own docstring) - NEVER
        # `all_ids` in that case, since `grm_inv` may now cover genotypes
        # entirely outside this task's own individuals. Falling back to
        # `all_ids` when `grm_ids` is None reproduces the pre-ver4-19
        # behaviour exactly for any caller that never passes it.
        _ginv_ids = [str(v) for v in grm_ids] if grm_ids is not None else all_ids
        ginv_ids_r = robjects.StrVector(_ginv_ids)

    print(f"{log_prefix} Fitting: {working.shape[0]} raw row(s) "
          + (f"(full trial grid - {len(all_ids)} individual(s) belong to this task, the rest "
             f"feed the shared spatial/residual structure only) " if _needs_full_grid else "")
          + f"for {len(id_train)} training individual(s), {len(id_masked)} masked "
          f"individual(s), factor column(s) {factor_cols}, order column(s) "
          f"{order_cols or '(none)'}.")
    # ver4-25, R9-S0: wall-clock time for THIS ONE R fit call only (never
    # including the Python-side pre/post-processing above/below it) -
    # `AsremlResult.timing` carries this forward so genomic_prediction.py
    # can turn it into one `ASReml_fit_timing.csv` row without this
    # module needing to know anything about that file's own schema/I/O.
    _fit_t0 = time.time()
    try:
        r_result = asreml_fit_and_predict(
            # ver4-19: `response_col` (the safe alias `working` was renamed
            # to above), never `str(trait)` - the R-side `dat[[response_
            # col]] <- as.numeric(...)` coercion must reference the SAME
            # column name `working` was actually renamed to.
            working, call_text, str(response_col), robjects.StrVector(factor_cols),
            robjects.StrVector(order_cols), robjects.StrVector(output_level),
            int(max_update_rounds), str(workspace), str(asreml_cfg.get('pworkspace', workspace)),
            (grm_inv if grm_inv is not None else robjects.NULL),
            ginv_ids_r,
            # Requirements.md item 1 fix: only meaningful (and only passed
            # as a real string) when a GRM is actually configured for this
            # fit - see asreml_fit.R's own updated coercion loop for why
            # the genotype column's factor level ORDER must be forced to
            # match rownames(Ginv) whenever vm() is in play.
            (str(genotype_col) if (grm_inv is not None and genotype_col) else robjects.NULL),
            (float(threads) if threads is not None else robjects.NULL),
            bool(return_vcov),
        )
    except Exception as exc:
        raise RuntimeError(f"{log_prefix} R-side fit failed: {exc!r}") from exc
    _fit_seconds = time.time() - _fit_t0

    pvals = r_list_get(r_result, 'pvals')
    varcomp = r_list_get(r_result, 'varcomp')
    converged = bool(r_list_get(r_result, 'converged')[0])
    iterations = int(r_list_get(r_result, 'iterations')[0])
    timing = _record_fit_timing(
        role, _fit_scope_hash(trait, id_train, id_valid, id_test),
        working.shape[0], len(all_ids), _fit_seconds, iterations, converged,
    )

    id_series = _extract_estimand_column(pvals, (genotype_col,), 'genotype/ID', log_prefix).astype(str)
    value_series = _extract_estimand_column(pvals, _VALUE_COL_CANDIDATES, 'predicted value', log_prefix)
    try:
        se_series = _extract_estimand_column(pvals, _SE_COL_CANDIDATES, 'standard error', log_prefix)
    except KeyError:
        se_series = pd.Series([np.nan] * len(pvals), index=pvals.index)

    # ver4-25, R8: `pred$vcov` (when `return_vcov`) is aligned to
    # `pvals`'s own row order (ASReml-R's documented `predict(...,
    # vcov=TRUE)` contract - see asreml_fit.R's own comment) - i.e. the
    # SAME order `id_series` was just built in, BEFORE any row is ever
    # dropped/reordered below. Captured here, immediately, rather than
    # after the `values['id'].isin(all_ids)` filter a few lines down,
    # since that filter should never actually drop a row in practice
    # (see its own comment) but this ordering guarantee must hold
    # regardless.
    vcov_arr, vcov_ids = None, None
    if return_vcov:
        raw_vcov = r_list_get(r_result, 'vcov')
        vcov_arr = _maybe_mmap_vcov(np.asarray(raw_vcov, dtype=float), log_prefix=log_prefix)
        vcov_ids = id_series.tolist()
        if vcov_arr.shape[0] != len(vcov_ids) or vcov_arr.shape[1] != len(vcov_ids):
            raise ValueError(
                f"{log_prefix} predict(..., vcov=TRUE) returned a {vcov_arr.shape} matrix, but "
                f"predict()$pvals has {len(vcov_ids)} row(s) - these should always match "
                f"(ASReml-R's own documented predict() contract) - refusing to use a covariance "
                f"matrix that cannot be reliably aligned to individual IDs."
            )

    id_to_split = {i: 'train' for i in id_train}
    id_to_split.update({i: 'valid' for i in id_valid})
    id_to_split.update({i: 'test' for i in id_test})
    estimand_label = asreml_syntax.estimand(asreml_cfg)

    values = pd.DataFrame({
        'id': id_series,
        'split': id_series.map(id_to_split),
        'value': value_series.to_numpy(),
        'estimand': estimand_label,
        'std_error': se_series.to_numpy(),
    })
    # A predict() row for an ID outside this task entirely (should not
    # happen - classify is over exactly this task's own data - kept as a
    # defensive filter rather than an assumption) never reaches the
    # post-condition checks below with a meaningless 'split' of NaN.
    values = values[values['id'].isin(all_ids)].reset_index(drop=True)

    # RK-2's post-condition assertions - fatal, never a warning, per the
    # design blueprint's own "the placeholder mean is never allowed to
    # reach a model silently" requirement.
    id_counts = values['id'].value_counts()
    duplicated_ids = id_counts[id_counts > 1].index.tolist()
    if duplicated_ids:
        raise ValueError(
            f"{log_prefix} {len(duplicated_ids)} individual ID(s) appear more than once in "
            f"predict()$pvals for this task: {sorted(duplicated_ids)[:20]} - this usually means "
            f"ASREML['output_level'] includes a column that does not take a single value per "
            f"genotype within this task's own population/replicate."
        )
    missing_ids = sorted(set(all_ids) - set(values['id']))
    if missing_ids:
        raise ValueError(
            f"{log_prefix} {len(missing_ids)} individual ID(s) have NO corresponding row in "
            f"predict()$pvals: {missing_ids[:20]} - Appendix B check B1 (does a MASKED "
            f"individual's response actually appear in predict()$pvals under "
            f"na.action=na.method(y='include')?) needs verifying against the licensed "
            f"installation if this recurs."
        )
    if values['value'].isna().any():
        _n_nan = int(values['value'].isna().sum())
        raise ValueError(
            f"{log_prefix} {_n_nan} individual(s) have a NaN adjusted phenotype value - refusing "
            f"to let this reach any model. The two known, specifically-checked-for causes (a "
            f"FIXED genotype term paired with a masked individual - RK-9; an aliased effect in a "
            f"`between`-scenario run - RK-3) are both ruled out by this point, since either would "
            f"have raised before the fit was ever attempted. This therefore indicates something "
            f"NOT covered by either check - e.g. a non-convergent fit that nonetheless reported "
            f"converged=True, or a genuinely different aliasing pattern - and needs its own "
            f"investigation rather than being assumed to be one of the two known cases."
        )

    heritability, genetic_row_index = _estimate_heritability(
        varcomp, genotype_col, asreml_cfg.get('genotype_effect', 'random')
    )
    # Design blueprint SS2.10 file schema: 'heritability' is populated
    # ONLY on the genotype-variance row (NaN everywhere else), never a
    # separate summary row - added here so genomic_prediction.py's Step
    # 6b can write ASReml_variance_components.csv straight from this
    # DataFrame with no further per-row logic of its own.
    varcomp = varcomp.copy()
    varcomp['heritability'] = np.nan
    if genetic_row_index is not None:
        varcomp.loc[genetic_row_index, 'heritability'] = heritability

    return AsremlResult(values=values, varcomp=varcomp, heritability=heritability,
                         iterations=iterations, converged=converged, timing=timing,
                         vcov=vcov_arr, vcov_ids=vcov_ids)


def asreml_adjust_phenotypes(
    raw_records: "pd.DataFrame", trait: str,
    id_train: Sequence[str], id_valid: Sequence[str], id_test: Sequence[str],
    asreml_cfg: dict, committed_syntax: str,
    grm_inv=None, workspace: str = '4gb',
    max_update_rounds: int = 20, log_prefix: str = '[ASReml]',
    *, grm_ids: "Optional[Sequence[str]]" = None, threads: "Optional[float]" = None,
) -> AsremlResult:
    """PUBLIC entry point for Step 6b (R5/R8/R9) - dispatches to one of
    THREE implementations depending on `asreml_cfg['genotype_effect']`,
    `asreml_cfg['fixed_holdout_mode']`, and whether this task has any
    validation/test individual at all (Requirements.md item 2's five
    configuration patterns - `asreml_syntax.ASREML_PATTERNS`):

      - `genotype_effect == 'random'`, OR no validation/test individual
        exists at all: ONE fit (`_fit_single_asreml_model()`, unchanged
        since ver4-12) - patterns `blup_no_grm`/`gblup`.
      - `genotype_effect == 'fixed'`, at least one validation/test
        individual exists, AND `fixed_holdout_mode == 'single_fit'`
        (Requirements.md item 2 - "the original version of BLUEs",
        pattern `blue_original`): ONE fit, genotype FIXED, over the
        FULL dataset with validation/test masked exactly as the
        single-fit path already does - legitimate ONLY when the same
        genotype is also observed (unmasked) somewhere in the training
        partition, which is precisely what `SCENARIO='between'` +
        `ASREML['between_shared_genotypes']=True` asserts
        (`asreml_syntax.validate_scenario_compatibility()` is the
        single authority that enforces this is never reached
        otherwise - `genomic_prediction.py::GP()`'s own preflight calls
        it before task 0, and the GUI never offers this option unless
        that same condition holds).
      - `genotype_effect == 'fixed'`, at least one validation/test
        individual exists, and `fixed_holdout_mode` is `'two_fit'` (the
        default): TWO independent fits (`_fit_blue_train_plus_holdout()`,
        ver4-13) - patterns `blue_blup_no_grm`/`blue_gblup`. A FIXED
        genotype term cannot be estimated for a masked individual with
        NO other observation at all (see this module's own top-of-file
        "Design decision: BLUE cannot be held out" section) - so
        training individuals get a genuine BLUE from one model, and
        validation/test individuals get a BLUP/GBLUP from a second,
        independent one.

    Every parameter has the exact same meaning in every case - this
    function does no work itself beyond the dispatch decision; see
    `_fit_single_asreml_model()`'s own docstring for the full parameter
    and exception documentation (every branch shares it).
    """
    id_valid_list = [str(v) for v in id_valid]
    id_test_list = [str(v) for v in id_test]
    id_masked = id_valid_list + id_test_list
    genotype_effect = asreml_cfg.get('genotype_effect', 'random')
    fixed_holdout_mode = asreml_cfg.get('fixed_holdout_mode', 'two_fit')

    if genotype_effect != 'fixed' or not id_masked:
        return _fit_single_asreml_model(
            raw_records, trait, id_train, id_valid, id_test, asreml_cfg, committed_syntax,
            grm_inv=grm_inv, workspace=workspace, max_update_rounds=max_update_rounds,
            log_prefix=log_prefix, grm_ids=grm_ids, role='legacy_masked', threads=threads,
        )

    if fixed_holdout_mode == 'single_fit':
        print(f"{log_prefix} ASREML['fixed_holdout_mode']='single_fit' (Requirements.md item 2, "
              f"'the original version of BLUEs') - fitting ONE genotype-FIXED model over the "
              f"full {len(id_train) + len(id_masked)}-individual dataset, masking "
              f"{len(id_masked)} validation/test individual(s); every individual's own value "
              f"comes from this SAME fit.")
        return _fit_single_asreml_model(
            raw_records, trait, id_train, id_valid, id_test, asreml_cfg, committed_syntax,
            grm_inv=grm_inv, workspace=workspace, max_update_rounds=max_update_rounds,
            log_prefix=log_prefix, allow_fixed_with_masked=True, grm_ids=grm_ids,
            role='legacy_masked_single_fit', threads=threads,
        )

    return _fit_blue_train_plus_holdout(
        raw_records, trait, id_train, id_valid_list, id_test_list, asreml_cfg, committed_syntax,
        grm_inv=grm_inv, workspace=workspace, max_update_rounds=max_update_rounds,
        log_prefix=log_prefix, grm_ids=grm_ids, threads=threads,
    )


def _fit_blue_train_plus_holdout(
    raw_records: "pd.DataFrame", trait: str,
    id_train: Sequence[str], id_valid: Sequence[str], id_test: Sequence[str],
    asreml_cfg: dict, committed_syntax: str,
    grm_inv=None, workspace: str = '4gb',
    max_update_rounds: int = 20, log_prefix: str = '[ASReml]',
    *, grm_ids: "Optional[Sequence[str]]" = None, threads: "Optional[float]" = None,
) -> AsremlResult:
    """ver4-13 - "Design decision: BLUE cannot be held out" (this
    module's own top-of-file docstring section has the full rationale;
    only the mechanics are documented here).

    Fits TWO independent models for one task and splices their outputs
    together:

      Fit 1 ("BLUE, train") : `committed_syntax` (genotype FIXED,
          exactly as the person configured it), on `raw_records`
          PHYSICALLY RESTRICTED to `id_train` - training individuals'
          rows only, nothing masked, nothing else present. Every row of
          its own `values` is `split == 'train'`.
      Fit 2 ("BLUP/GBLUP, holdout") : `asreml_cfg['holdout_committed_
          syntax']` (genotype forced RANDOM by `asreml_syntax.
          holdout_model_cfg()`, optionally with `grm_inv` when
          `use_grm=True`), on the FULL `raw_records` (train + valid +
          test), with valid/test masked exactly as the single-fit path
          already does. Only the rows where `split != 'train'` are kept
          from its own `values` - its training-individual rows are
          discarded (Fit 1's BLUE is authoritative for those).

    Both fits reuse `_fit_single_asreml_model()` unchanged - this
    function is pure orchestration plus a final, independent sanity
    check over the COMBINED result (mirroring RK-2's own "never trust,
    always verify" discipline one level up, since a bug in THIS
    function's own row-selection logic - not either underlying fit -
    would not be caught by either fit's own post-condition checks in
    isolation).

    Raises
    ------
    ValueError if `asreml_cfg['holdout_committed_syntax']` is missing or
        lacks the `{TRAIT}` placeholder - exactly the same two-layer
        defence `committed_syntax` itself already has: the GUI's
        `gather_config()` refuses to build a config missing it, and this
        run-time check is the second, independent layer for a hand-
        written/headless config that bypassed the GUI entirely - or if
        either fit itself raises (surfaced unchanged, prefixed with
        which fit it came from), or if the combined result fails the
        assembly-level sanity check above.
    """
    # ver4-17 - DEFENSIVE BACKSTOP for `asreml_syntax.validate_structure()`'s
    # own new check of this exact combination (see that function's own
    # comment for the full reasoning: Fit 1 below PHYSICALLY EXCLUDES
    # every non-training row - the leakage-avoidance design this whole
    # function exists for - which is fundamentally incompatible with a
    # grid-structured residual's own completeness requirement). A purely
    # config-derived check (no task data needed), so - like that
    # function's own check - this never needs a SEPARATE runtime-only
    # variant the way RK-3/RK-9 do; it is repeated here only because
    # `GP()`'s own preflight call to `validate_structure()` can be
    # downgraded to a warning (`ASREML_STRICT=False`), and this function
    # would otherwise fail with the same cryptic, unrelated-looking
    # R-side row-count-mismatch error `_fit_single_asreml_model()` used
    # to raise before ver4-17 - never silently, and never a leaky BLUE.
    if asreml_syntax.residual_requires_full_grid(asreml_cfg):
        raise ValueError(
            f"{log_prefix} ASREML['residual']['dims'] configures a grid-structured residual "
            f"together with genotype_effect='fixed' and a non-empty validation/test split - "
            f"this combination is not supported (see asreml_syntax.validate_structure()'s own "
            f"check for the full reasoning: Fit 1's training-only physical exclusion and a grid "
            f"residual's own completeness requirement cannot both be satisfied). This should "
            f"have been caught by GP()'s own preflight before task 0 - seeing it here means "
            f"ASREML_STRICT=False let a structurally invalid config proceed anyway."
        )

    genotype_col = asreml_cfg.get('genotype_col')
    holdout_syntax = asreml_cfg.get('holdout_committed_syntax')
    if not holdout_syntax or '{TRAIT}' not in holdout_syntax:
        raise ValueError(
            f"{log_prefix} ASREML['genotype_effect']='fixed' with a non-empty validation/test "
            f"split needs ASREML['holdout_committed_syntax'] (a SECOND committed syntax, "
            f"genotype fitted RANDOM, containing {{TRAIT}}) to adjust those individuals' own "
            f"phenotypes - a FIXED genotype term cannot (see this module's own top-of-file "
            f"'Design decision: BLUE cannot be held out' section for why). Commit one from the "
            f"GUI's holdout-model panel, or set it directly in a hand-written config - "
            f"`asreml_syntax.suggest_holdout_syntax(ASREML)` generates a starting point from the "
            f"rest of this config."
        )

    id_train_list = [str(v) for v in id_train]
    id_valid_list = [str(v) for v in id_valid]
    id_test_list = [str(v) for v in id_test]
    holdout_cfg = asreml_syntax.holdout_model_cfg(asreml_cfg)
    holdout_uses_grm = bool(holdout_cfg.get('use_grm', False))
    holdout_fit_label = 'holdout_gblup' if holdout_uses_grm else 'holdout_blup'
    holdout_log_tag = 'GBLUP/holdout' if holdout_uses_grm else 'BLUP/holdout'

    # --- Fit 1: BLUE, training individuals only - PHYSICALLY excluded,
    # never masked, so this fit's own environmental/design-effect
    # estimates (and every training individual's own BLUE) never see a
    # single validation/test response - at least as strict as masking,
    # and considerably simpler to reason about (there is no "masked
    # partition" in this fit's own data at all).
    train_only_records = raw_records[
        raw_records['ID'].astype(str).isin(set(id_train_list))
    ].reset_index(drop=True)
    print(f"{log_prefix} [BLUE/train] Fitting genotype FIXED on {len(id_train_list)} training "
          f"individual(s) only ({train_only_records.shape[0]} raw row(s)) - no validation/test "
          f"data is present in this fit's own dataset at all.")
    fit1 = _fit_single_asreml_model(
        train_only_records, trait, id_train_list, [], [], asreml_cfg, committed_syntax,
        grm_inv=None, workspace=workspace, max_update_rounds=max_update_rounds,
        log_prefix=f"{log_prefix} [BLUE/train]", role='legacy_train_blue', threads=threads,
    )

    # --- Fit 2: BLUP/GBLUP, every held-out individual - genotype forced
    # RANDOM regardless of the user's own genotype_effect='fixed' choice,
    # since `holdout_committed_syntax` was committed to match exactly
    # that. Full dataset, validation/test masked - identical mechanism to
    # the pre-existing single-fit path, so training data legitimately
    # informs held-out predictions (via shared Year/Experiment effects,
    # and via the GRM when configured) without leaking any held-out
    # individual's OWN response into its own adjustment.
    print(f"{log_prefix} [{holdout_log_tag}] Fitting genotype RANDOM"
          f"{' with a genomic relationship matrix' if holdout_uses_grm else ''} on the full "
          f"{len(id_train_list) + len(id_valid_list) + len(id_test_list)}-individual dataset, "
          f"masking {len(id_valid_list) + len(id_test_list)} validation/test individual(s) - "
          f"only THEIR rows are kept from this fit (training rows come from Fit 1's BLUE "
          f"instead).")
    fit2 = _fit_single_asreml_model(
        raw_records, trait, id_train_list, id_valid_list, id_test_list, holdout_cfg, holdout_syntax,
        grm_inv=grm_inv, workspace=workspace, max_update_rounds=max_update_rounds,
        log_prefix=f"{log_prefix} [{holdout_log_tag}]", grm_ids=grm_ids, role='legacy_holdout',
        threads=threads,
    )

    combined_values = pd.concat(
        [fit1.values, fit2.values[fit2.values['split'] != 'train']],
        axis=0, ignore_index=True,
    )

    # Final, independent sanity check over the COMBINED result - RK-2's
    # own "never trust, always verify" discipline applied once more at
    # the assembly boundary between two otherwise-independent fits.
    all_ids = id_train_list + id_valid_list + id_test_list
    _dupe_ids = combined_values['id'].value_counts()
    _dupe_ids = _dupe_ids[_dupe_ids > 1].index.tolist()
    if _dupe_ids:
        raise ValueError(
            f"{log_prefix} {len(_dupe_ids)} individual ID(s) appear in BOTH Fit 1 (BLUE/train) "
            f"and Fit 2 ({holdout_log_tag}) after combining: {sorted(_dupe_ids)[:20]} - this "
            f"indicates id_train/id_valid/id_test overlap upstream, not a problem with either "
            f"fit in isolation."
        )
    _missing_ids = sorted(set(all_ids) - set(combined_values['id']))
    if _missing_ids:
        raise ValueError(
            f"{log_prefix} {len(_missing_ids)} individual ID(s) have no adjusted value in EITHER "
            f"fit after combining: {_missing_ids[:20]}."
        )
    if combined_values['value'].isna().any():
        raise ValueError(
            f"{log_prefix} {int(combined_values['value'].isna().sum())} individual(s) have a NaN "
            f"adjusted phenotype value after combining Fit 1 (BLUE/train) and Fit 2 "
            f"({holdout_log_tag}) - each fit's own post-condition check should have caught this "
            f"independently; report this as a bug rather than working around it."
        )

    # 'fit'/'converged'/'iterations' are set PER ROW here, so
    # genomic_prediction.py's own Step 6b never has to (and must not)
    # blend the two fits' status into one blanket value the way the
    # single-fit path's own scalar summary does.
    varcomp1 = fit1.varcomp.copy()
    varcomp1['fit'] = 'train_blue'
    varcomp1['converged'] = fit1.converged
    varcomp1['iterations'] = fit1.iterations
    varcomp2 = fit2.varcomp.copy()
    varcomp2['fit'] = holdout_fit_label
    varcomp2['converged'] = fit2.converged
    varcomp2['iterations'] = fit2.iterations
    combined_varcomp = pd.concat([varcomp1, varcomp2], axis=0, ignore_index=True)
    combined_varcomp = combined_varcomp[
        ['fit'] + [c for c in combined_varcomp.columns if c != 'fit']
    ]

    print(f"{log_prefix} Combined result for this task: {fit1.values.shape[0]} BLUE row(s) "
          f"(train) + {(fit2.values['split'] != 'train').sum()} "
          f"{'GBLUP' if holdout_uses_grm else 'BLUP'} row(s) (validation/test) = "
          f"{combined_values.shape[0]} total. Estimand is genuinely MIXED - see "
          f"asreml_syntax.estimand_description() for a one-line summary; do not assume every "
          f"row is a BLUE just because ASREML['genotype_effect']='fixed'.")

    # ver4-25, R9-S0: TWO independent R fits happened here (Fit 1 "BLUE,
    # train" and Fit 2 "BLUP/GBLUP, holdout") - `timing` therefore carries
    # a LIST of both fits' own timing dicts (never blended into one, for
    # the same reason `combined_varcomp` above keeps them in separate
    # rows tagged by `fit`) rather than the single-dict shape every other
    # AsremlResult-producing function returns; genomic_prediction.py's
    # own timing-harvesting helper normalises either shape into 0-2 rows.
    combined_timing = [t for t in (fit1.timing, fit2.timing) if t]

    return AsremlResult(
        values=combined_values,
        varcomp=combined_varcomp,
        heritability=fit2.heritability,
        iterations=max(fit1.iterations, fit2.iterations),
        converged=bool(fit1.converged and fit2.converged),
        timing=combined_timing or None,
    )


def _healthy_reference_vparameters(ref: "Optional[AsremlResult]") -> "Optional[Dict[str, float]]":
    """ver4-58 - `{component_raw: estimate}` from a HEALTHY reference fit
    (the per-set-GRM train split), or `None` when there is no usable
    reference: no fit, a fit that itself needed `ai.sing` or borrowed its
    components, an empty variance-component table, a non-finite estimate,
    or any component at a boundary/singular (`bound` 'B'/'S') - borrowing
    a degenerate value would only move the degeneracy to the small split.
    """
    if ref is None or ref.ai_sing_used or ref.borrowed_variance:
        return None
    vc = ref.varcomp
    if vc is None or vc.shape[0] == 0 or not {'component', 'estimate'} <= set(vc.columns):
        return None
    if 'bound' in vc.columns and vc['bound'].astype(str).str.upper().isin(['B', 'S']).any():
        return None
    est = pd.to_numeric(vc['estimate'], errors='coerce')
    if not np.isfinite(est.to_numpy(dtype=float)).all():
        return None
    return dict(zip(vc['component'].astype(str), est.astype(float)))


def _per_set_grm_singular_fallback(fit_split, split_name: str,
                                    train_ref: "Optional[AsremlResult]",
                                    log_prefix: str, strict_exc: BaseException,
                                    allow_borrow: bool = True) -> "AsremlResult":
    """ver4-58 - what `fit_per_set_grm_adjustment()` does when a split's
    STRICT fit aborts with an AI-matrix singularity. `fit_split(**extra)`
    refits that split with extra `fit_unmasked_adjustment()` keywords.

    1. Healthy train reference available (and this is not the train
       split): refit with every variance parameter fixed at the train
       estimates (`fixed_vparameters`).
    2. Otherwise, or if step 1 fails for any reason: the ver4-55
       `ai.sing = TRUE` retry (no override -> R's `'auto'` behaviour),
       with a WARNING that the fit is not a regular REML estimate.
    Raises whatever step 2 raises (the pre-fix behaviour), so
    `ASREML['on_nonconvergence']` keeps working unchanged.
    """
    print(f"{log_prefix} WARNING: this split's own data cannot identify every variance "
          f"parameter (ASReml AI-matrix singularity: {str(strict_exc)[:160]!r}...).")
    ref_vp = (_healthy_reference_vparameters(train_ref)
              if (allow_borrow and split_name != 'train') else None)
    if ref_vp:
        print(f"{log_prefix} Refitting with ALL variance parameters held fixed at the train "
              f"split's {len(ref_vp)} healthy estimate(s) - variance components are "
              f"estimated on the large set, this split is only predicted.")
        try:
            return fit_split(fixed_vparameters=ref_vp, ai_sing_override='off')
        except Exception as borrow_exc:
            print(f"{log_prefix} WARNING: the borrowed-variance refit failed "
                  f"({type(borrow_exc).__name__}: {str(borrow_exc)[:200]}) - falling back to the "
                  f"ai.sing = TRUE retry.")
    else:
        why = ("ASREML['borrow_variance'] is False" if not allow_borrow
               else "this IS the train split" if split_name == 'train'
               else "no healthy train-split reference (missing, itself singular/at a boundary, "
                    "or itself needed ai.sing)")
        print(f"{log_prefix} Cannot borrow variance components: {why} - falling back to the "
              f"ai.sing = TRUE retry.")
    print(f"{log_prefix} WARNING: the ai.sing = TRUE fit leaves the unidentified variance "
          f"parameter at ~0, so this split's GBLUPs may be close to constant and weighted "
          f"ensembles fitted on them will learn little. Prefer a larger split or "
          f"use_entire_dataset=True.")
    return fit_split()


def fit_per_set_grm_adjustment(
    raw_records: "pd.DataFrame", trait: str,
    id_train: Sequence[str], id_valid: Sequence[str], id_test: Sequence[str],
    asreml_cfg: dict, committed_syntax: str,
    *,
    response_unit: str = 'genotype', population_col: str = 'population',
    grm_inv_train=None, grm_ids_train: "Optional[Sequence[str]]" = None,
    grm_inv_valid=None, grm_ids_valid: "Optional[Sequence[str]]" = None,
    grm_inv_test=None, grm_ids_test: "Optional[Sequence[str]]" = None,
    workspace: str = '4gb', max_update_rounds: int = 20,
    log_prefix: str = '[ASReml]', threads: "Optional[float]" = None,
    context_ids: "Optional[Sequence[str]]" = None,
) -> AsremlResult:
    """ver4-7, R1 (design blueprint "Per-Set GRM Generation for GBLUP",
    AMB-1 resolved as a 3rd held-out estimation mode) - fits THREE
    INDEPENDENT genotype-random+GRM models for one task, one per split
    (train/valid/test), each PHYSICALLY RESTRICTED to just that split's
    own individuals (mirroring `_fit_blue_train_plus_holdout()`'s own
    "PHYSICALLY EXCLUDED, never masked" discipline for Fit 1) and each
    scoped to its OWN genomic relationship matrix - the caller
    (`genomic_prediction.py`'s Step 6b) is responsible for resolving each
    split's own `grm_inv_*`/`grm_ids_*` pair via `resolve_grm_genotype_
    universe()` + `build_or_load_grm()`, per the "no cross-contamination"
    scoping rule (when `ASREML['use_entire_dataset']` is True every split
    is widened to the individuals of every OTHER population - see
    `context_ids_valid` / `context_ids_test` below and the ver4-57 note).

    ver4-57 BUGFIX ("use entire dataset" + per-set GRM): previously ONLY
    the train fit was widened to the other populations; the valid and
    test fits (and their GRMs) were built from just their own split's
    individuals, so the information from every other population never
    reached them. `context_ids` is now the list of individuals (the other
    populations' - never any individual of this task's own train/valid/
    test, which the function strips if a caller includes one) that is
    POOLED INTO EVERY split's fit as context, alongside that split's own
    individuals; the caller builds each split's `grm_*` over the same
    own-plus-context universe. `id_train`/`id_valid`/`id_test` are
    therefore always this task's OWN split membership. Reporting: the
    train fit reports every individual it fitted (its own plus the
    context, exactly as the old widened train fit did); the valid and
    test fits report ONLY their own individuals - the context
    individuals are already reported once by the train fit, so keeping
    them would duplicate IDs in the combined result. No individual of one
    of this task's own splits is ever shared with another split's fit
    ("no cross-contamination" of train/valid/test is unchanged).
    `context_ids=None`/empty reproduces the pre-ver4-57 behaviour exactly
    (I11).

    Unlike `_fit_blue_train_plus_holdout()` (genotype FIXED for training
    individuals, RANDOM/masked for held-out ones - a fixed term cannot
    be estimated for a masked individual at all), every one of THIS
    function's three fits uses the SAME (genotype-random, GRM-using)
    `committed_syntax` - only the data each one sees, and the
    relationship matrix it uses, differ. No response is ever masked to
    NA in any of the three fits - each split's own individuals
    contribute their own real, unmasked response to their own split's
    fit, informed by relatedness (via that split's own GRM) rather than
    by validation/test masking.

    Skips (rather than fits) any split whose own `id_*` list is empty -
    `id_valid` in particular is routinely empty for a `SCENARIO='within'`
    task with no validation partition configured.

    Returns
    -------
    `AsremlResult` whose `values['estimand']` is `'GBLUP'` for every row
    (all three fits are genotype-random+GRM) and whose `values['split']`
    correctly reflects which of the three fits produced each row -
    overwritten explicitly here, since `fit_unmasked_adjustment()` itself
    always reports its own `values['split']` as the literal string
    `'all'` (it has no notion of a physical partition at all). `varcomp`
    carries a leading `fit` column (`'per_set_grm_train'` / `'_valid'` /
    `'_test'`), mirroring `_fit_blue_train_plus_holdout()`'s own
    `combined_varcomp` shape, so a reader can always tell which of the
    (up to) three fits produced which rows.

    Raises
    ------
    ValueError if `asreml_cfg` does not configure a genotype-random+GRM
        main strategy, if `committed_syntax` is empty/missing `{TRAIT}`,
        if EVERY split is empty, if any non-empty split has no raw
        phenotype record at all, or if the combined, merged result fails
        the same assembly-level sanity check `_fit_blue_train_plus_
        holdout()` already applies. RuntimeError if any individual fit
        itself raises.
    """
    if asreml_cfg.get('genotype_effect', 'random') != 'random' or not asreml_cfg.get('use_grm', False):
        raise ValueError(
            f"{log_prefix} fit_per_set_grm_adjustment: ASREML['per_set_grm']=True requires "
            f"genotype_effect='random' and use_grm=True (a BLUP-with-GRM main strategy) - see "
            f"asreml_syntax.validate_structure()'s own mutual-exclusion check."
        )
    if not committed_syntax or '{TRAIT}' not in committed_syntax:
        raise ValueError(
            f"{log_prefix} fit_per_set_grm_adjustment: committed_syntax is empty or missing the "
            f"{{TRAIT}} placeholder - commit a syntax from the GUI's ASReml section before "
            f"running with ASREML enabled."
        )

    id_train_list = [str(v) for v in id_train]
    id_valid_list = [str(v) for v in id_valid]
    id_test_list = [str(v) for v in id_test]

    # ver4-57: context individuals (other populations' - see the docstring)
    # pooled into every split's fit. Never includes an individual of any
    # of this task's own splits: an own-split ID is stripped here so a
    # careless caller can never make one split's individual appear in
    # another split's fit.
    _own_all_ids = set(id_train_list) | set(id_valid_list) | set(id_test_list)
    _context_list = sorted({str(v) for v in (context_ids or [])} - _own_all_ids)

    _split_specs = (
        ('train', id_train_list, grm_inv_train, grm_ids_train, _context_list),
        ('valid', id_valid_list, grm_inv_valid, grm_ids_valid, _context_list),
        ('test', id_test_list, grm_inv_test, grm_ids_test, _context_list),
    )

    fit_results: "dict" = {}
    fit_timings: "list" = []
    for split_name, split_ids, grm_inv, grm_ids, context_ids in _split_specs:
        if not split_ids:
            print(f"{log_prefix} [per_set_grm/{split_name}] Empty split - skipped (RK-3, no "
                  f"validation/test partition configured for this task).")
            continue
        _own_set = set(split_ids)
        _fit_set = _own_set | set(context_ids)
        # The split must have at least one record of its OWN (the context
        # alone can never stand in for it).
        _n_own_rows = int(raw_records['ID'].astype(str).isin(_own_set).sum())
        if _n_own_rows == 0:
            raise ValueError(
                f"{log_prefix} fit_per_set_grm_adjustment: no raw phenotype record found for any "
                f"of this task's {len(split_ids)} {split_name} individual(s) - cannot fit."
            )
        split_records = raw_records[
            raw_records['ID'].astype(str).isin(_fit_set)
        ].reset_index(drop=True)
        if context_ids:
            print(f"{log_prefix} [per_set_grm/{split_name}] Fitting genotype RANDOM with a genomic "
                  f"relationship matrix over this split's own {len(split_ids)} individual(s) PLUS "
                  f"{len(context_ids)} individual(s) of every other population pooled in as "
                  f"context ({split_records.shape[0]} raw row(s) in total) - ASREML["
                  f"'use_entire_dataset']=True. Nothing masked; no individual of "
                  f"another train/valid/test split of THIS task is included; only this split's "
                  f"own individuals are reported from this fit.")
        else:
            print(f"{log_prefix} [per_set_grm/{split_name}] Fitting genotype RANDOM with a genomic "
                  f"relationship matrix scoped to this split's own {len(split_ids)} individual(s) "
                  f"only ({split_records.shape[0]} raw row(s)) - PHYSICALLY RESTRICTED, nothing "
                  f"masked; independent of every other split's own fit ('no "
                  f"cross-contamination').")
        # Reuses fit_unmasked_adjustment() (the ordinary new-engine
        # single-fit function - NOT the legacy _fit_single_asreml_
        # model()) so response_unit/factor-column resolution, the
        # {TRAIT}-safe-alias rename, and every other new-engine fitting
        # detail stay identical to every other new-engine fit this run
        # makes - `mask_keys=None` (nothing is ever masked here; the
        # split's own physical restriction, above, is what isolates it)
        # and `id_train/id_valid/id_test` set to (split_ids, [], []) so
        # NOTHING within this one call is treated as held-out.
        #
        # ver4-58 BUGFIX (production failure, cotton / `LEN`, task 1/25,
        # `per_set_grm/valid`: "1 singularities in the Average
        # Information matrix"): a small, physically-restricted split
        # (valid/test) often cannot identify sigma_g^2 and sigma_e^2 from
        # its own records. Order of attempts, controlled by
        # `ASREML['ai_sing']`:
        #   1. STRICT fit (no ai.sing) - the ordinary, healthy outcome.
        #   2. on an AI singularity: refit with EVERY variance parameter
        #      held FIXED at the train split's own (healthy) estimates -
        #      estimate the variance components on the large set, predict
        #      the small one. The GBLUPs stay informative.
        #   3. only when no healthy train reference exists (or step 2
        #      itself fails): the ver4-55 `ai.sing = TRUE` retry, with a
        #      WARNING - this leaves the unidentified parameter at ~0, so
        #      every GBLUP of the split collapses towards a constant.
        # `'off'` keeps the hard error, `'on'` uses ai.sing from the start.
        _split_log_prefix = f"{log_prefix} [per_set_grm/{split_name}]"

        def _fit_this_split(**extra):
            return fit_unmasked_adjustment(
                split_records, trait, asreml_cfg.get('genotype_col'), asreml_cfg,
                committed_syntax,
                response_unit=response_unit, population_col=population_col,
                grm_inv=grm_inv, grm_ids=grm_ids, workspace=workspace,
                max_update_rounds=max_update_rounds,
                log_prefix=_split_log_prefix,
                mask_keys=None, role=f'per_set_grm_{split_name}', threads=threads,
                **extra,
            )

        if _normalise_ai_sing_mode(asreml_cfg) == 'auto':
            try:
                split_result = _fit_this_split(ai_sing_override='off')
            except RuntimeError as _strict_exc:
                if not _is_ai_singularity_error(_strict_exc):
                    raise
                split_result = _per_set_grm_singular_fallback(
                    _fit_this_split, split_name, fit_results.get('train'),
                    _split_log_prefix, _strict_exc,
                    allow_borrow=_borrow_variance_enabled(asreml_cfg),
                )
        else:
            split_result = _fit_this_split()
        # fit_unmasked_adjustment()'s own returned `values['split']` is
        # always the literal string `'all'` (its own docstring - "there
        # is no PHYSICAL train/valid/test partition in this fit at all")
        # - corrected here to the real split name, since THIS function's
        # own contract (unlike fit_unmasked_adjustment()'s) is that
        # `split` names which of the three independent fits produced
        # each row.
        split_result.values = split_result.values.copy()
        split_result.values['split'] = split_name
        if context_ids and split_name != 'train':
            # ver4-57: report ONLY this split's own individuals. The
            # context individuals were fitted to inform the variance
            # components / relationship structure, but each of them is
            # reported (once) by the train fit, so keeping them here
            # would duplicate IDs in the combined result below.
            _n_before = split_result.values.shape[0]
            split_result.values = split_result.values[
                split_result.values['id'].astype(str).isin(_own_set)
            ].reset_index(drop=True)
            print(f"{log_prefix} [per_set_grm/{split_name}] Kept {split_result.values.shape[0]} of "
                  f"{_n_before} adjusted value(s) - the {split_name} split's own individuals; the "
                  f"remaining context individual(s) are reported by the train fit.")
        fit_results[split_name] = split_result
        if split_result.timing:
            fit_timings.append(split_result.timing)

    if not fit_results:
        raise ValueError(
            f"{log_prefix} fit_per_set_grm_adjustment: every split (train/valid/test) is empty "
            f"for this task - nothing to fit."
        )

    combined_values = pd.concat(
        [r.values for r in fit_results.values()], axis=0, ignore_index=True,
    )

    # Final, independent sanity check over the COMBINED result - the SAME
    # "never trust, always verify" discipline `_fit_blue_train_plus_
    # holdout()`'s own assembly-level check already applies, extended
    # from a 2-way to an N-way (up to 3) merge.
    all_ids = id_train_list + id_valid_list + id_test_list
    _dupe_ids = combined_values['id'].value_counts()
    _dupe_ids = _dupe_ids[_dupe_ids > 1].index.tolist()
    if _dupe_ids:
        raise ValueError(
            f"{log_prefix} {len(_dupe_ids)} individual ID(s) appear in more than one of the "
            f"per-set-GRM fits after combining: {sorted(_dupe_ids)[:20]} - this indicates "
            f"id_train/id_valid/id_test overlap upstream, not a problem with any one fit in "
            f"isolation."
        )
    _missing_ids = sorted(set(all_ids) - set(combined_values['id']))
    if _missing_ids:
        raise ValueError(
            f"{log_prefix} {len(_missing_ids)} individual ID(s) have no adjusted value in ANY of "
            f"the per-set-GRM fits after combining: {_missing_ids[:20]}."
        )
    if combined_values['value'].isna().any():
        raise ValueError(
            f"{log_prefix} {int(combined_values['value'].isna().sum())} individual(s) have a NaN "
            f"adjusted phenotype value after combining the per-set-GRM fits - each fit's own "
            f"post-condition check should have caught this independently; report this as a bug "
            f"rather than working around it."
        )

    varcomp_frames = []
    for split_name, r in fit_results.items():
        vc = r.varcomp.copy()
        # ver4-58: a split whose variance components were borrowed from the
        # train split is labelled so (no new column - the append schema of
        # ASReml_variance_components.csv is unchanged); an ai.sing fit is
        # labelled likewise so it is never mistaken for a regular estimate.
        vc['fit'] = (f'per_set_grm_{split_name}_borrowedVC' if r.borrowed_variance
                     else f'per_set_grm_{split_name}_aiSing' if r.ai_sing_used
                     else f'per_set_grm_{split_name}')
        vc['converged'] = r.converged
        vc['iterations'] = r.iterations
        varcomp_frames.append(vc)
    combined_varcomp = pd.concat(varcomp_frames, axis=0, ignore_index=True)
    combined_varcomp = combined_varcomp[
        ['fit'] + [c for c in combined_varcomp.columns if c != 'fit']
    ]

    # Heritability: reported from whichever split's own fit resolved one
    # first (train, then valid, then test) - purely a convenience summary
    # scalar (mirrors `_fit_blue_train_plus_holdout()`'s own choice of
    # Fit 2's heritability as ITS summary scalar); every split's own
    # per-row heritability is still available, per-row, in `varcomp`.
    heritability = None
    for split_name in ('train', 'valid', 'test'):
        _r = fit_results.get(split_name)
        if _r is not None and _r.heritability is not None:
            heritability = _r.heritability
            break

    print(f"{log_prefix} Combined per-set-GRM result for this task: " +
          ", ".join(f"{fit_results[s].values.shape[0]} GBLUP row(s) ({s})" for s in fit_results) +
          f" = {combined_values.shape[0]} total. Every row's estimand is 'GBLUP' - three "
          f"INDEPENDENT fits, one per split, each with its own relationship matrix.")

    return AsremlResult(
        values=combined_values,
        varcomp=combined_varcomp,
        heritability=heritability,
        iterations=max(r.iterations for r in fit_results.values()),
        converged=all(r.converged for r in fit_results.values()),
        timing=fit_timings or None,
        ai_sing_used=any(r.ai_sing_used for r in fit_results.values()),
        borrowed_variance=any(r.borrowed_variance for r in fit_results.values()),
    )


def apply_to_splits(train: "pd.DataFrame", valid: "pd.DataFrame", test: "pd.DataFrame",
                     id_train: Sequence[str], id_valid: Sequence[str], id_test: Sequence[str],
                     result: AsremlResult, *,
                     population_train: "Optional[str]" = None,
                     population_valid: "Optional[str]" = None,
                     population_test: "Optional[str]" = None):
    """Overwrite `train`/`valid`/`test`'s LAST column (the phenotype - I4)
    IN PLACE with the converged BLUE/BLUP, matched by ID - never by
    position. Positional alignment between each frame and its own
    `id_*` list (I4) is what makes a plain positional assignment (after
    reordering `result.values` to match `id_*`'s own order) correct and
    is asserted explicitly below rather than assumed.

    ver4-19, Requirements.md R3: when `result.values` carries a
    `population` column (i.e. `result` came from a `response_unit ==
    'genotype_by_population'` fit - `fit_unmasked_adjustment()` below),
    the lookup joins on `(id, population)` instead of `id` alone, and the
    caller MUST supply `population_train`/`population_valid`/
    `population_test` - the SINGLE population label every row of that
    split belongs to. This is always exactly one label per split, for
    both `SCENARIO='within'` (all three splits share the current task's
    one population) and `SCENARIO='between'` (each split IS its own
    named population, by construction - architecture doc SS6). Ignored
    (and optional) when `result.values` has no `population` column
    (`response_unit == 'genotype'` - the pre-ver4-19, still-default
    shape for every existing caller that never passes these).

    Returns
    -------
    `(train, valid, test)` - new DataFrames (copies; the originals are
    never mutated), each with an identical shape/column-order to its
    input, only the last column's VALUES changed.

    Raises
    ------
    ValueError if a split's row count doesn't match its own ID list's
    length (a broken positional-alignment invariant upstream - I4), if
    any requested ID has no adjusted value in `result` (should not happen
    given the caller's own post-condition checks, kept here anyway as a
    second, independent layer of defence - RK-2), or (R3) if
    `result.values` is population-keyed but a non-empty split's own
    population label was not supplied.
    """
    _by_population = 'population' in result.values.columns
    if _by_population:
        _vals = result.values.copy()
        _vals['__key__'] = list(zip(_vals['id'].astype(str), _vals['population'].astype(str)))
        lookup = _vals.set_index('__key__')['value']
    else:
        lookup = result.values.set_index('id')['value']

    def _substitute(df: "pd.DataFrame", ids: Sequence[str], split_label: str,
                     population_label: "Optional[str]") -> "pd.DataFrame":
        if df.shape[0] == 0:
            return df
        ids = [str(i) for i in ids]
        if len(ids) != df.shape[0]:
            raise ValueError(
                f"apply_to_splits: the {split_label} split has {df.shape[0]} row(s) but "
                f"{len(ids)} ID(s) were given - positional alignment (invariant I4) has been "
                f"broken upstream of Step 6b."
            )
        if _by_population:
            if population_label is None:
                raise ValueError(
                    f"apply_to_splits: result.values is keyed by (id, population) "
                    f"(response_unit='genotype_by_population' - Requirements.md R3) but no "
                    f"population_{split_label} was supplied for the non-empty {split_label} "
                    f"split."
                )
            keys = [(i, str(population_label)) for i in ids]
            missing = sorted(set(keys) - set(lookup.index))
            if missing:
                raise ValueError(
                    f"apply_to_splits: {len(missing)} (id, population) pair(s) in the "
                    f"{split_label} split have no corresponding ASReml-adjusted value: "
                    f"{missing[:20]}."
                )
            ordered_values = lookup.loc[keys].to_numpy()
        else:
            missing = sorted(set(ids) - set(lookup.index))
            if missing:
                raise ValueError(
                    f"apply_to_splits: {len(missing)} ID(s) in the {split_label} split have no "
                    f"corresponding ASReml-adjusted value: {missing[:20]}."
                )
            ordered_values = lookup.loc[ids].to_numpy()
        df = df.copy()
        df.iloc[:, -1] = ordered_values
        return df

    train = _substitute(train, id_train, 'train', population_train)
    valid = _substitute(valid, id_valid, 'valid', population_valid)
    test = _substitute(test, id_test, 'test', population_test)
    return train, valid, test


def build_or_load_grm(genotype_source, ids: Sequence[str], cache: dict, *, grm_blend: float = 0.01):
    """Genomic relationship matrix (inverse), for exactly `ids` in exactly
    that row/column order, reusing `genomic_prediction.py::_build_grm()`
    (invariant I12 - see this module's own top-of-file docstring for why
    that import is deferred to inside this function).

    Parameters
    ----------
    genotype_source : `(train_df, valid_df, test_df)` - marker DataFrames
        whose LAST column is the phenotype (`_build_grm()`'s own expected
        shape - see that function's docstring) and whose ROW ORDER,
        concatenated as `rbind(train_df, valid_df, test_df)`, is the
        SAME order as `ids` (an invariant the CALLER - genomic_prediction.
        py's Step 6b - guarantees by construction: `full_marker_pool`/the
        current task pool's own row order is always `id_train + id_valid
        + id_test`, per invariant I4's positional-alignment discipline;
        this function trusts that rather than re-deriving it, since
        re-deriving it here would need an ID column these marker
        DataFrames never carry in the first place - see this module's own
        docstring on why).
    ids : the individual IDs this GRM must cover, in the exact order the
        caller needs the returned matrix aligned to.
    cache : a plain dict, owned by ONE `GP()` call (process-local - see
        `build_or_load_grm`'s own module-level cross-reference in
        genomic_prediction.py) - keyed by `frozenset(ids)`, so the SAME
        individual set (any order) reuses an already-built matrix rather
        than rebuilding it (blueprint SS2.12, "Caching": G depends only on
        the marker file and the individual set - never the split, trait,
        or replicate). Stores a `DataFrame` (index/columns = the ID order
        the matrix was ORIGINALLY built in), not a bare `ndarray`, so a
        later request for the SAME set in a DIFFERENT order still gets a
        correctly-realigned result via `.loc[ids, ids]`, rather than
        silently returning a matrix aligned to the wrong order.
    grm_blend : `(1 - grm_blend) * G + grm_blend * I` bending towards
        identity before inversion (`ASREML.grm_blend`, default 0.01) -
        guards against a singular/ill-conditioned relationship matrix
        (too few markers relative to individuals, near-duplicate
        individuals) being impossible to invert. Not part of the design
        blueprint's own 3-argument public signature for this function -
        an additive, keyword-only, defaulted deviation (see the ver4-12
        Change Summary SS7) needed so `ASREML.grm_blend` has anywhere to
        reach this function from at all.

    Returns
    -------
    `np.ndarray`, shape `(len(ids), len(ids))`, the INVERSE relationship
    matrix, row/column-aligned to `ids` in the given order.

    Raises
    ------
    RuntimeError if `_build_grm()` itself fails (returns `None` - see its
    own docstring for when), or if its output shape doesn't match `ids`.
    """
    ids = [str(i) for i in ids]
    key = frozenset(ids)
    cached_df = cache.get(key)
    if cached_df is not None:
        return cached_df.loc[ids, ids].to_numpy()

    from genomic_prediction import _build_grm  # deferred - see module docstring
    from pipeline_utils import get_active_compute_resources

    try:
        device = get_active_compute_resources().get('device', 'cpu')
    except Exception:
        device = 'cpu'

    train_df, valid_df, test_df = genotype_source
    n_expected = sum(df.shape[0] for df in (train_df, valid_df, test_df) if df.shape[0] != 0)
    if n_expected != len(ids):
        raise RuntimeError(
            f"[ASReml] build_or_load_grm: the supplied marker pool has {n_expected} row(s) but "
            f"{len(ids)} ID(s) were requested - the caller's own row-order-matches-ids invariant "
            f"has been broken."
        )

    g = _build_grm(train_df, valid_df, test_df, device)
    if g is None:
        raise RuntimeError(
            "[ASReml] Genomic relationship matrix build failed (see the '[GP] NOTE:' line just "
            "above, from _build_grm() itself, for why) - cannot proceed with "
            "ASREML['use_grm']=True for this task."
        )
    if g.shape != (len(ids), len(ids)):
        raise RuntimeError(
            f"[ASReml] build_or_load_grm: _build_grm() returned a {g.shape} matrix, expected "
            f"({len(ids)}, {len(ids)})."
        )

    # Symmetrise (guards against floating-point rounding asymmetry from
    # the underlying X @ X.T product) and bend towards identity (guards
    # against a singular/ill-conditioned matrix) BEFORE inverting -
    # blueprint SS2.12's own documented numerics.
    g = (g + g.T) / 2.0
    identity = np.eye(g.shape[0])
    g = (1.0 - grm_blend) * g + grm_blend * identity
    try:
        g_inv = np.linalg.inv(g)
    except np.linalg.LinAlgError:
        print(f"[ASReml] WARNING: the bent genomic relationship matrix (grm_blend={grm_blend}) "
              f"is still singular - falling back to the Moore-Penrose pseudo-inverse. Consider "
              f"increasing ASREML['grm_blend'] if this recurs.")
        g_inv = np.linalg.pinv(g)

    cache[key] = pd.DataFrame(g_inv, index=ids, columns=ids)
    return g_inv


# ===========================================================================
# ver4-19 - Requirements.md R1, R2, R3
# EasiGP_ASReml_BLUE_BLUP_Redesign_ver4-19.md
# ===========================================================================
#
# Everything ABOVE this point is the LEGACY engine (ver4-12...ver4-18):
# masked validation/test responses, the "BLUE cannot be held out" two-fit
# design, and the five `asreml_syntax.ASREML_PATTERNS`. It is unchanged and
# remains fully reachable, bit-for-bit, via `ASREML['legacy_masked_holdout']
# = True` - `genomic_prediction.py::GP()`'s own Step 6b dispatch is what
# actually chooses between the two engines; nothing in this module decides
# that for itself.
#
# Everything BELOW this point is the NEW default engine. Three things
# distinguish it from the legacy one, and they are exactly Requirements.md's
# three items:
#   R1 - `resolve_grm_genotype_universe()` / `log_grm_universe_diagnostics()`
#        widen the genomic relationship matrix to cover every genotype id
#        actually present in the data being fitted, not just one task's own
#        population subset (root-cause fix - also applied to the LEGACY
#        engine above via the `grm_ids` parameter threaded through
#        `_fit_single_asreml_model()` et al., since R1 says "fix this"
#        unconditionally, not "fix this only for the new engine").
#   R2 - `GENOTYPE_EFFECT_DESCRIPTIONS` / `describe_genotype_effect_strategy()`
#        (the three GUI captions) and `fit_unmasked_adjustment()` (fits over
#        EVERY individual's own raw records - train, validation and test
#        alike, nothing masked) plus `fit_concurrent_variance_model()` (the
#        optional, training-only side model for variance components /
#        heritability when the main strategy is `'blue_fixed'`).
#   R3 - `detect_shared_genotypes_across_populations()` /
#        `resolve_response_unit()` decide whether a genotype needs its own
#        adjusted value PER POPULATION (`asreml_syntax.build_model_spec()`'s
#        own `response_unit='genotype_by_population'` branch renders the
#        actual formula change - see that module).
# ---------------------------------------------------------------------------


def resolve_grm_genotype_universe(raw_records: "pd.DataFrame", genotype_col: str,
                                   available_marker_ids, log_prefix: str = '[ASReml]'):
    """Requirements.md R1 - the genomic relationship matrix must cover
    every genotype id present in the grid ACTUALLY being fitted, never
    merely one task's own population subset (see this module's own
    "ver4-19" section header just above, and the
    EasiGP_ASReml_BLUE_BLUP_Redesign blueprint SS1 for the full root-
    cause account: a grid-structured residual pulls in every
    population's rows for spatial reasons, but the pre-ver4-19 GRM
    stayed scoped to the current task's own individuals, silently
    leaving most of the fitted grid's own genotype levels unresolvable
    against it).

    This is a PURE function - it never reads a file, never builds the
    relationship matrix itself, and never touches `raw_records`. It only
    decides WHICH ids `genomic_prediction.py`'s own caller should build
    the GRM for (via the existing, unchanged `build_or_load_grm()`).

    Parameters
    ----------
    raw_records : the exact DataFrame about to be (or already) fitted -
        whatever the caller resolved as its own working records (already
        the FULL trial grid whenever a grid-structured residual is
        configured, per `asreml_syntax.residual_requires_full_grid()`;
        otherwise one task's own individuals only). Deriving the
        universe from whatever was ACTUALLY fitted - rather than from a
        separately tracked "whole file" concept - is what keeps this
        correct in both cases without special-casing either one.
    genotype_col : the column identifying the individual/genotype in
        `raw_records`.
    available_marker_ids : any container supporting `in` (a `set` is
        cheapest) - every id that actually HAS marker/genotype data
        available (e.g. `set(data_genotype_original['ID'])` for a CSV
        genotype file, or the PLINK `.fam` file's own IID set). A
        genotype with phenotype records but no marker record must still
        be excluded from the GRM - a different, legitimate reason to
        drop it, reported separately (never silently) by
        `log_grm_universe_diagnostics()` below, rather than folded into
        one uninformative blanket warning the way the pre-ver4-19
        R-side "genotype id ... is NOT among the ... id(s) the genomic
        relationship matrix was built for" message did.

    Returns
    -------
    `(universe_ids, n_total_distinct, excluded_ids)`:
      - `universe_ids` : sorted `list[str]` of distinct genotype ids in
        `raw_records` that DO have marker data - safe to hand straight
        to `build_or_load_grm()`.
      - `n_total_distinct` : how many distinct genotype ids
        `raw_records` has at all, regardless of marker availability.
      - `excluded_ids` : sorted `list[str]` of genotype ids dropped for
        lacking marker data.

    Raises
    ------
    ValueError if `genotype_col` is not a column of `raw_records` at
    all - a config/wiring bug upstream, not a data-quality question this
    function should silently paper over.
    """
    if genotype_col not in raw_records.columns:
        raise ValueError(
            f"{log_prefix} resolve_grm_genotype_universe: genotype column {genotype_col!r} is "
            f"not present in the supplied records - cannot resolve a genotype universe for the "
            f"genomic relationship matrix."
        )
    all_ids = sorted(set(raw_records[genotype_col].dropna().astype(str).unique().tolist()))
    n_total = len(all_ids)
    available = (
        available_marker_ids if isinstance(available_marker_ids, (set, frozenset))
        else set(str(v) for v in available_marker_ids)
    )
    universe_ids = [i for i in all_ids if i in available]
    excluded_ids = [i for i in all_ids if i not in available]
    return universe_ids, n_total, excluded_ids


def log_grm_universe_diagnostics(universe_ids: Sequence[str], n_total_distinct: int,
                                  excluded_ids: Sequence[str], log_prefix: str = '[ASReml]') -> None:
    """Requirements.md R1 SS1.4 - an unconditional, always-printed
    accounting of the genotype universe a GRM was actually built for,
    replacing the pre-ver4-19 single `WARNING` line (which reported the
    SYMPTOM - rows with an unmatched genotype level - without ever
    saying whether that was expected). Mirrors architecture doc SS18's
    own "log negative states unconditionally" discipline: printed every
    time a GRM is built, whether or not any genotype was actually
    excluded, so a clean run's own log is just as informative as one
    with a real data-quality issue.
    """
    n_included = len(universe_ids)
    n_excluded = len(excluded_ids)
    _sample = (
        f" (e.g. {excluded_ids[:10]}{', ...' if n_excluded > 10 else ''})" if n_excluded else ""
    )
    print(f"{log_prefix} Genotype universe for this fit: {n_total_distinct} distinct genotype "
          f"id(s) found across the fitted grid; {n_included} have marker data and are included "
          f"in the genomic relationship matrix; {n_excluded} lack marker data and are "
          f"excluded{_sample}.")


def detect_shared_genotypes_across_populations(pheno_df: "pd.DataFrame", genotype_col: str,
                                                population_col: str = 'population'):
    """Requirements.md R3 - pure detector for "does any genotype in this
    phenotype file appear in more than one population". Callable from
    BOTH the headless engine (`resolve_response_unit()` below) and
    `main_app.py`'s own pre-run validation (so the GUI can show the
    auto-detected recommendation before a run even starts, the same way
    `resource_profiles.py` shows a resource *estimate* before
    submission) - no I/O beyond the already-loaded phenotype frame
    handed to it.

    Cheap: one `groupby` over the phenotype file, no marker data needed.

    Returns
    -------
    `(shared_ids, n_total_genotypes)` - `shared_ids` is the sorted
    `list[str]` of genotype ids that appear under more than one distinct
    population value; `n_total_genotypes` is how many distinct genotype
    ids the file has in total (for the diagnostic line's own
    denominator).
    """
    working = pheno_df[[genotype_col, population_col]].dropna().copy()
    working[genotype_col] = working[genotype_col].astype(str)
    working[population_col] = working[population_col].astype(str)
    counts = working.groupby(genotype_col)[population_col].nunique()
    shared_ids = sorted(counts[counts > 1].index.tolist())
    return shared_ids, int(counts.shape[0])


def resolve_response_unit(asreml_cfg: dict, pheno_df: "pd.DataFrame", genotype_col: str,
                           population_col: str = 'population', log_prefix: str = '[ASReml]') -> str:
    """Requirements.md R3 - resolve `ASREML['response_unit']`
    (`asreml_syntax.RESPONSE_UNITS`) into a concrete
    `'genotype'`/`'genotype_by_population'` value, auto-detecting from
    the data when the config says `'auto'` (the default -
    `asreml_syntax.DEFAULT_RESPONSE_UNIT`). Pure given its inputs (no
    file I/O of its own) and idempotent - calling it again with an
    already-concrete `response_unit` simply confirms and re-logs that
    value, so it is safe for both the GUI (which may resolve this once,
    early, for its own live syntax preview) and the headless engine
    (which resolves it again, from the same phenotype data, immediately
    before fitting) to call it independently without ever disagreeing.

    Returns
    -------
    `'genotype'` or `'genotype_by_population'` - never `'auto'`.
    """
    configured = (asreml_cfg or {}).get('response_unit', 'auto')
    if configured not in ('auto', 'genotype', 'genotype_by_population'):
        raise ValueError(
            f"{log_prefix} ASREML['response_unit']={configured!r} is not recognised - must be "
            f"'auto', 'genotype', or 'genotype_by_population'."
        )
    if configured != 'auto':
        print(f"{log_prefix} response_unit resolved: {configured!r} (explicitly configured).")
        return configured

    shared_ids, n_total = detect_shared_genotypes_across_populations(
        pheno_df, genotype_col, population_col
    )
    if shared_ids:
        resolved = 'genotype_by_population'
        print(f"{log_prefix} response_unit resolved: 'genotype_by_population' (auto) - "
              f"{len(shared_ids)} of {n_total} distinct {genotype_col!r} id(s) in the raw "
              f"phenotype file appear in more than one population; treating "
              f"({genotype_col}, {population_col}) as the adjustment unit so each population's "
              f"performance for a shared genotype is estimated separately (Requirements.md R3).")
    else:
        resolved = 'genotype'
        print(f"{log_prefix} response_unit resolved: 'genotype' (auto) - no genotype id in the "
              f"raw phenotype file ({n_total} distinct id(s) checked) appears in more than one "
              f"population; one adjusted value per genotype (unchanged earlier behaviour).")
    return resolved


# ---------------------------------------------------------------------------
# Requirements.md R2 - the three GUI captions. Kept here (rather than in
# asreml_syntax.py) per the EasiGP_ASReml_BLUE_BLUP_Redesign blueprint SS2.5:
# "the three caption strings should live as data in Preprocess/asreml_blup.py
# itself... and be read by both main_app.py (to render the caption) and any
# headless config-validation path (to print the same text into the run log
# when ASREML['genotype_effect_strategy'] is resolved)".
# ---------------------------------------------------------------------------
GENOTYPE_EFFECT_DESCRIPTIONS = {
    'blue_fixed': {
        'label': 'BLUEs (genotype fixed) \u2014 recommended, default',
        'recommendation': 'recommended',
        'caption': (
            "Keeps genomic-relationship information out of the training target entirely, so "
            "downstream models must learn a real marker\u2192phenotype mapping instead of "
            "partially reproducing a relationship-based estimate they were implicitly handed. "
            "This is the only one of the three that treats every model family \u2014 the "
            "additive genomic models and the tree/kernel/neural models alike \u2014 the same "
            "way, and it matches standard two-stage genomic-selection practice. "
            "(BLUEs_vs_BLUPs.md Q1, Q2, Q3)"
        ),
    },
    'blup_iid': {
        'label': 'BLUPs without GRM (genotype random, independent) \u2014 conditionally recommended',
        'recommendation': 'conditional',
        'caption': (
            "Corrects for unequal replication among genotypes without borrowing information "
            "across relatives, so it avoids the GRM's circularity. In a balanced trial this is "
            "an affine transform of the BLUE and gives identical correlations and rankings \u2014 "
            "prefer it over BLUEs mainly when replication is genuinely unequal, and note that "
            "any MSE computed against it is on a shrunk scale, not the BLUE's scale. "
            "(BLUEs_vs_BLUPs.md Q4)"
        ),
    },
    'blup_grm': {
        'label': 'BLUPs with GRM (genotype random, genomic relationship matrix) \u2014 not recommended as a training/evaluation target',
        'recommendation': 'not_recommended',
        'caption': (
            "Borrows information across genetically related individuals through the SAME "
            "markers the prediction models also use, which inflates agreement between the two "
            "and does so unevenly across model families (it favours rrBLUP/GBLUP-style models "
            "specifically). Keep this option available for users who want a breeder-facing "
            "shrinkage value for other purposes, but it should not be the basis for comparing "
            "model families against each other. (BLUEs_vs_BLUPs.md Q1, Q3)"
        ),
    },
}


def describe_genotype_effect_strategy(strategy: str) -> dict:
    """Requirements.md R2 - single-authority lookup for one strategy's
    GUI label/caption, read by `main_app.py` (to render the caption
    beneath each of the three radio options) and by
    `genomic_prediction.py::GP()`'s own preflight (to print the SAME
    text into the run log when `ASREML['genotype_effect_strategy']` is
    resolved), so the two can never independently drift on what a given
    strategy means (the same discipline architecture doc SS2's Layer 3d
    table cites for every other cross-cutting fact).

    Raises
    ------
    KeyError for an unrecognised `strategy` - never silently substitutes
    a different one's caption.
    """
    return GENOTYPE_EFFECT_DESCRIPTIONS[strategy]


def fit_unmasked_adjustment(
    raw_records: "pd.DataFrame", trait: str, genotype_col: str,
    asreml_cfg: dict, committed_syntax: str,
    *, response_unit: str = 'genotype', population_col: str = 'population',
    grm_inv=None, grm_ids: "Optional[Sequence[str]]" = None,
    workspace: str = '4gb', max_update_rounds: int = 20,
    log_prefix: str = '[ASReml]', mask_keys: "Optional[set]" = None,
    role: str = 'unmasked',
    step1_vparameters: "Optional[Mapping[str, float]]" = None,
    warm_start_vparameters: "Optional[Mapping[str, float]]" = None,
    threads: "Optional[float]" = None,
    return_vcov: bool = False,
    fixed_vparameters: "Optional[Mapping[str, float]]" = None,
    ai_sing_override: "Optional[str]" = None,
) -> AsremlResult:
    """Requirements.md R2 - the NEW default Step-6b engine: ONE fit per
    phenotype trait, over EVERY individual with a recorded value for
    this trait (train, validation and test alike by default - see
    `mask_keys` below), see `BLUEs_vs_BLUPs.md` Q2/Q3 and the
    EasiGP_ASReml_BLUE_BLUP_Redesign blueprint SS2.1-2.3. Produces a
    genuine BLUE/BLUP/GBLUP - an OBSERVATION of that individual (or of
    that (individual, population) cell when `response_unit ==
    'genotype_by_population'` - R3) when `mask_keys` is not supplied,
    or a genuine held-out PREDICTION for whichever individuals
    `mask_keys` names when it is (ver4-20 follow-up - see that
    parameter's own docstring).

    Unlike `_fit_single_asreml_model()` (the legacy masked engine, kept
    unchanged and reachable via `ASREML['legacy_masked_holdout'] =
    True`), this function:
      - has no notion of `id_train`/`id_valid`/`id_test` as physically
        separate partitions, and no two-fit/"BLUE cannot be held out"
        branching - every row is fitted in the SAME single call
        regardless of `mask_keys`; masking (when requested) only
        replaces the RESPONSE value of the named rows with NA before
        that one fit, exactly the way a genuinely missing plot already
        behaves under `na.action = na.method(y = "include")`
        (asreml_fit.R) - it never excludes a row or requires a second,
        independent model;
      - is intended to be called ONCE per phenotype trait and CACHED by
        the caller (`genomic_prediction.py`'s own per-trait cache,
        mirroring the `_bio_prior_merge_cache` precedent - architecture
        doc SS13.4) rather than once per `(population, ratio,
        replicate)` task, whenever `mask_keys` is `None` - the result
        then depends only on `(genotype universe, trait, strategy,
        response_unit)` (the blueprint's own SS2.4 framing). Once
        `mask_keys` is supplied the result ALSO depends on this task's
        own valid/test membership, so the caller must NOT cache across
        tasks in that case (see `genomic_prediction.py`'s own
        `_asreml_get_unmasked_result()` docstring/comment for how it
        switches between the two).

    Parameters
    ----------
    raw_records : every raw phenotype record this trait's fit should
        see - ordinarily `genomic_prediction.py`'s own
        `data_phenotype_raw_for_asreml` (the WHOLE trial, every
        population), already grid-padded if a grid-structured residual
        is configured. Never mutated - a working copy is always made,
        both for the response-column rename below and (when `mask_keys`
        is given) for the masking step itself.
    trait : this run's own phenotype/trait column name.
    genotype_col : the column identifying the individual/genotype.
    asreml_cfg : the resolved `ASREML` config dict.
    committed_syntax : the GUI-committed (or hand-written) ASReml-R call
        text for the CURRENTLY selected strategy - already rendered with
        the correct genotype term for `response_unit` (see
        `asreml_syntax.build_model_spec()`/`render_syntax()`) - still
        containing the `{TRAIT}`/`{DATA}`/(`{GRM}`) placeholders,
        substituted here once.
    response_unit : `'genotype'` (default) or `'genotype_by_population'`
        - already RESOLVED (never `'auto')` - see `resolve_response_unit()`
        above.
    population_col : only consulted when `response_unit ==
        'genotype_by_population'`.
    grm_inv, grm_ids : as `_fit_single_asreml_model()` - `grm_ids` is the
        actual id order `grm_inv`'s rows/columns are aligned to (the
        widened genotype universe from `resolve_grm_genotype_universe()`
        - Requirements.md R1).
    mask_keys : ver4-20 follow-up (`Requirements.md`'s "simplify the
        GUI ... Genotype effect / genomic relationship matrix
        configuration subsection", item 1) - `None` (the default,
        reproducing ver4-19's original unmasked behaviour exactly) or a
        `set` of `(genotype_id, population_or_None)` tuples naming this
        task's own validation/test individuals whose raw response
        should be set to NA before fitting. When `response_unit ==
        'genotype_by_population'`, a row is masked only when BOTH its
        `genotype_col` value and its `population_col` value match one
        of the tuples (so the SAME genotype's real, unmasked record in
        a DIFFERENT population - not this task's own - still
        contributes to the fit); otherwise only the genotype id half of
        each tuple is consulted. Masking never removes a row - it only
        overwrites that row's own response with NA, exactly like a
        genuinely missing plot.
    role : ver4-25, R9-S0 - see `_fit_single_asreml_model()`'s own
        docstring for this parameter's meaning; used identically here.
    step1_vparameters : ver4-25, R7 (Holland & Piepho 2024, Fig. 2) -
        `None` (the default, reproducing every pre-ver4-25 caller's
        behaviour exactly - I11) or a `{component_raw: estimate}` dict
        from a PRIOR step-(i) fit (genotype random, GRM-informed - see
        `genomic_prediction.py::_asreml_get_blue_refinement_step1()`) on
        this SAME trait's genotype universe. When given AND `asreml_cfg`
        configures `genotype_effect == 'fixed'` (BLUE), this function
        routes through `Preprocess/asreml_fit.R::asreml_fit_fixed_with_
        constrained_vparameters()` instead of the ordinary single call -
        fixing every NON-genetic variance parameter at step (i)'s own
        converged value (`Constraint = "F"`) before re-estimating the
        genotype term as FIXED (step (ii)) - rather than
        `asreml_fit_and_predict()`. Ignored (with one `[GP]`-style NOTE,
        printed by the CALLER, not here) whenever `genotype_effect` is
        `'random'` - the refinement is specifically a BLUE-quality
        improvement (Fig. 2's whole point is stabilising the non-genetic
        variance parameters BEHIND a fixed genotype term); a random
        genotype term already estimates its own variance jointly with
        every other term in one fit, so there is nothing to refine.
    warm_start_vparameters : ver4-25, R9-S4 (design blueprint stage 5,
        "Warm start per-task fits") - `None` (default, I11) or a
        `{component_raw: estimate}` dict of a PREVIOUSLY converged fit's
        own variance parameters for this SAME trait, used to SEED (not
        fix) this fit's own starting point via the identical R-side
        mechanism `step1_vparameters` uses (`fix_values=False` - see
        `Preprocess/asreml_fit.R::asreml_fit_fixed_with_constrained_
        vparameters()`). Ignored whenever `step1_vparameters` is also
        given (R7 takes precedence - the two seed the same underlying
        mechanism and are never combined). Only ever consulted by the
        CALLER when `ASREML['warm_start_per_task']=True` - disclosed as
        NOT bit-identical to a cold start in general (a different,
        equally valid REML optimum can be reached from a different
        starting point, though it should converge to the same value
        within ordinary REML tolerance).
    fixed_vparameters : ver4-58 - `None` (default, I11) or a
        `{component_raw: estimate}` dict from a larger, HEALTHY fit of the
        same model (`fit_per_set_grm_adjustment()` passes the train
        split's own variance components). Every matched variance
        parameter is HELD FIXED at that value (`Constraint = 'F'`) via
        `asreml_fit_fixed_with_constrained_vparameters(fix_values=TRUE)`,
        so the fit only solves for the random effects - no Average
        Information matrix is needed for fixed parameters, which is what
        removes a "N singularities in the Average Information matrix"
        abort caused by a split too small to identify its own variance
        components. Unlike `step1_vparameters` it applies to a random
        genotype term too. Takes precedence over `step1_vparameters` and
        `warm_start_vparameters`. The result carries
        `borrowed_variance=True`.
    ai_sing_override : ver4-58 - `None` (default: use
        `ASREML['ai_sing']`) or `'auto'`/`'on'`/`'off'` for THIS call
        only. `fit_per_set_grm_adjustment()` passes `'off'` for its strict
        first attempt so an AI singularity surfaces as an error it can
        handle instead of silently becoming a degenerate `ai.sing` fit.
    threads : ver4-25, R9-S5 (design blueprint stage 5, "ASReml
        threading") - forwarded to `asreml.options(threads=...)` when not
        `None` (default `None` - untouched, I11); silently ignored by the
        R side if the installed ASReml-R version doesn't support it (see
        `Preprocess/asreml_fit.R::asreml_set_options()`'s own comment).
    return_vcov : ver4-25, R8 (design blueprint SS2, Stage 6) - see
        `_fit_single_asreml_model()`'s own identical parameter docstring.
        Forwarded to whichever of the two R call paths this function
        actually takes (the R7 refined-BLUE path, or the ordinary path) -
        both R functions now accept it identically.

    Returns
    -------
    `AsremlResult` whose `values` has columns `id`, `population` (only
    when `response_unit == 'genotype_by_population'`), `value`,
    `estimand`, `std_error` - one row per individual (or per
    (individual, population) cell) in `raw_records`, keyed however
    `output_level` classifies. `values['split']` is always `'all'`
    (there is no PHYSICAL train/valid/test partition in this fit at all,
    masked or not - the caller's own per-task `apply_to_splits()` call
    is what later slices this result down to one task's own individuals).

    Raises
    ------
    ValueError for the same structural problems `_fit_single_asreml_model()`
    already guards (empty records, a missing `{TRAIT}` placeholder, a
    genotype id with no adjusted value - see the masked-vs-unmasked
    distinction in this function's own NaN post-condition check below).
    RuntimeError if the R fit itself raises - the caller's own
    `ASREML['on_nonconvergence']` policy decides whether this aborts the
    run or skips the task, same as the legacy engine.

    ver4-21 BUGFIX: a NaN adjusted value for a genotype that has ZERO
    non-missing raw observations for `trait` anywhere in `raw_records` is
    NOT an error - it is left in the returned `values` as NaN (see the
    NaN post-condition check below for why this is provably harmless: the
    same genotype is already excluded from every task's own
    `id_train`/`id_valid`/`id_test` for this trait by the ordinary geno x
    phenotype `dropna()`, so this NaN is never actually looked up by
    `apply_to_splits()`). Only a genotype that DOES have at least one
    non-missing raw observation for `trait` yet still comes back NaN
    (an aliased/inestimable effect - the case this check was written to
    catch) still raises `ValueError`.
    """
    if not committed_syntax or '{TRAIT}' not in committed_syntax:
        raise ValueError(
            f"{log_prefix} fit_unmasked_adjustment: committed_syntax is empty or missing the "
            f"{{TRAIT}} placeholder - commit a syntax from the GUI's ASReml section before "
            f"running with ASREML enabled."
        )
    if response_unit not in ('genotype', 'genotype_by_population'):
        raise ValueError(
            f"{log_prefix} fit_unmasked_adjustment: response_unit={response_unit!r} must "
            f"already be resolved to 'genotype' or 'genotype_by_population' (never 'auto') - "
            f"see resolve_response_unit()."
        )
    _by_population = response_unit == 'genotype_by_population'

    working = raw_records.copy()
    # ver4-18's grid-completion is idempotent - see
    # `_fit_single_asreml_model()`'s own identical call for why this is
    # safe (and cheap) to repeat defensively here even when the caller
    # already padded `raw_records` once, up front, for the whole run.
    working = asreml_syntax.pad_residual_grid(working, asreml_cfg)
    working['ID'] = working['ID'].astype(str)

    if working.shape[0] == 0:
        raise ValueError(f"{log_prefix} fit_unmasked_adjustment: no raw phenotype record was "
                          f"supplied for trait {trait!r} - cannot fit.")

    # Same fixed, always-bare response-column alias the legacy engine
    # uses (module docstring, "ver4-19 BUGFIX... unexpected symbol") -
    # applied unconditionally so a trait name with a space or parenthesis
    # never reaches R in any form.
    if _ASREML_SAFE_RESPONSE_COL in working.columns and _ASREML_SAFE_RESPONSE_COL != trait:
        raise ValueError(
            f"{log_prefix} The phenotype file already has a column literally named "
            f"{_ASREML_SAFE_RESPONSE_COL!r}, which this ASReml integration reserves "
            f"internally for the response variable during a fit - rename that column in the "
            f"phenotype file and try again."
        )
    working = working.rename(columns={trait: _ASREML_SAFE_RESPONSE_COL})
    response_col = _ASREML_SAFE_RESPONSE_COL

    # ver4-20 follow-up - `mask_keys` is the ONLY thing that ever touches
    # a row's own response in this engine. When it is `None` (the
    # original ver4-19 behaviour, and still the recommended default),
    # every row's own response is used exactly as recorded - a NaN
    # response, whether a genuinely missing plot or a ver4-18
    # grid-filler row, already contributes nothing to the likelihood
    # under `na.action = na.method(y = "include")` (asreml_fit.R). When
    # `mask_keys` names some rows, those rows' response is overwritten
    # with NA HERE, before the (single, unchanged) fit call below - see
    # this function's own docstring for the exact matching rule.
    _n_masked_rows = 0
    if mask_keys:
        if _by_population:
            _row_keys = list(zip(
                working[genotype_col].astype(str), working[population_col].astype(str)
            ))
            _mask_key_set = {(str(_gid), str(_pop)) for _gid, _pop in mask_keys if _pop is not None}
        else:
            _row_keys = list(working[genotype_col].astype(str))
            _mask_key_set = {str(_gid) for _gid, _ in mask_keys}
        _mask_row_bool = pd.Series(
            [k in _mask_key_set for k in _row_keys], index=working.index
        )
        _n_masked_rows = int(_mask_row_bool.sum())
        working.loc[_mask_row_bool, response_col] = np.nan
        print(f"{log_prefix} [mask=on/{response_unit}] {_n_masked_rows} raw row(s) (of "
              f"{working.shape[0]}) matching {len(mask_keys)} validation/test key(s) had "
              f"trait {trait!r} set to NA before fitting ("
              f"Requirements.md, optional masking).")

    factor_cols, order_cols = _collect_factor_and_order_columns(asreml_cfg)
    if _by_population and population_col not in factor_cols:
        factor_cols = sorted(set(factor_cols) | {population_col})

    output_level = list(asreml_cfg.get('output_level', []) or [])
    if genotype_col not in output_level:
        output_level = output_level + [genotype_col]
    if _by_population and population_col not in output_level:
        output_level = output_level + [population_col]

    call_text = committed_syntax.replace(
        '{TRAIT}', asreml_syntax.r_safe_name(response_col)
    ).replace('{DATA}', 'dat')
    if grm_inv is not None:
        call_text = call_text.replace('{GRM}', 'Ginv')

    import rpy2.robjects as robjects
    ensure_r_native_symbols_global()
    r_source = robjects.r['source']
    r_source(_R_FIT_FILE)
    _push_ai_sing_option_to_r(asreml_cfg, override=ai_sing_override)
    asreml_fit_and_predict = robjects.globalenv['asreml_fit_and_predict']

    ginv_ids_r = robjects.NULL
    if grm_inv is not None:
        _ginv_ids = [str(v) for v in grm_ids] if grm_ids is not None else sorted(
            set(working[genotype_col].dropna().astype(str))
        )
        ginv_ids_r = robjects.StrVector(_ginv_ids)

    # ver4-25, R7: the two-step refined-BLUE path is only meaningful (and
    # only ever attempted) for a FIXED genotype term - see this
    # function's own docstring on `step1_vparameters`.
    _use_refinement = bool(step1_vparameters) and asreml_cfg.get('genotype_effect', 'random') == 'fixed'
    # ver4-25, R9-S4 (design blueprint stage 5, "Warm start per-task
    # fits"): only consulted when R7's OWN refinement isn't already
    # active for this call (the two are mutually exclusive uses of the
    # same underlying R mechanism - `fix_values=True` vs. `False` - see
    # `asreml_fit_fixed_with_constrained_vparameters()`'s own R-side
    # comment). `warm_start_vparameters` seeds the optimiser's own
    # starting point without constraining anything - NOT bit-identical to
    # a cold start in general (disclosed in this function's own
    # docstring), only ever used when `ASREML['warm_start_per_task']=True`
    # (default `False` - I11).
    _use_warm_start = (not _use_refinement) and bool(warm_start_vparameters)
    # ver4-58: borrowed (fixed) variance components take precedence over
    # both of the above and apply to a RANDOM genotype term as well.
    _use_fixed = bool(fixed_vparameters)
    if _use_fixed:
        _use_refinement = False
        _use_warm_start = False

    print(f"{log_prefix} [{'mask=on' if mask_keys else 'unmasked'}/{response_unit}"
          f"{'/refined-BLUE' if _use_refinement else ''}"
          f"{'/warm-start' if _use_warm_start else ''}"
          f"{'/borrowed-VC' if _use_fixed else ''}] Fitting "
          f"trait {trait!r} on {working.shape[0]} raw row(s) covering "
          f"{working['ID'].nunique()} individual(s) - "
          + (f"{_n_masked_rows} row(s) masked to NA"
             if mask_keys else "NOTHING masked (Requirements.md R2)")
          + f"; factor column(s) {factor_cols}, order column(s) {order_cols or '(none)'}, "
            f"classify {':'.join(output_level)!r} (GRM {'on' if grm_inv is not None else 'off'})."
          + (f" Non-genetic variance parameters fixed at step-(i)'s own {len(step1_vparameters)} "
             f"estimate(s) (Holland & Piepho 2024, Fig. 2)." if _use_refinement else "")
          + (f" ALL variance parameters held FIXED at {len(fixed_vparameters)} value(s) "
             f"estimated on a larger, healthy fit (this split's own data cannot "
             f"identify them)." if _use_fixed else "")
          + (f" Seeded from {len(warm_start_vparameters)} previously-converged variance "
             f"parameter estimate(s) for this trait - free to move, not fixed."
             if _use_warm_start else ""))
    _fit_t0 = time.time()
    try:
        if _use_refinement or _use_warm_start or _use_fixed:
            _vp_source = (fixed_vparameters if _use_fixed
                          else step1_vparameters if _use_refinement else warm_start_vparameters)
            fixed_vp_r = robjects.FloatVector(list(_vp_source.values()))
            fixed_vp_r.names = robjects.StrVector([str(k) for k in _vp_source.keys()])
            asreml_fit_fixed_with_constrained_vparameters = robjects.globalenv[
                'asreml_fit_fixed_with_constrained_vparameters'
            ]
            r_result = asreml_fit_fixed_with_constrained_vparameters(
                working, call_text, str(response_col), robjects.StrVector(factor_cols),
                robjects.StrVector(order_cols), robjects.StrVector(output_level),
                fixed_vp_r, str(workspace), str(asreml_cfg.get('pworkspace', workspace)),
                (grm_inv if grm_inv is not None else robjects.NULL),
                ginv_ids_r,
                (str(genotype_col) if (grm_inv is not None and genotype_col) else robjects.NULL),
                int(max_update_rounds), bool(_use_refinement or _use_fixed),
                (float(threads) if threads is not None else robjects.NULL),
                bool(return_vcov),
            )
        else:
            r_result = asreml_fit_and_predict(
                working, call_text, str(response_col), robjects.StrVector(factor_cols),
                robjects.StrVector(order_cols), robjects.StrVector(output_level),
                int(max_update_rounds), str(workspace), str(asreml_cfg.get('pworkspace', workspace)),
                (grm_inv if grm_inv is not None else robjects.NULL),
                ginv_ids_r,
                (str(genotype_col) if (grm_inv is not None and genotype_col) else robjects.NULL),
                (float(threads) if threads is not None else robjects.NULL),
                bool(return_vcov),
            )
    except Exception as exc:
        raise RuntimeError(f"{log_prefix} R-side fit failed: {exc!r}") from exc
    _fit_seconds = time.time() - _fit_t0

    pvals = r_list_get(r_result, 'pvals')
    varcomp = r_list_get(r_result, 'varcomp')
    converged = bool(r_list_get(r_result, 'converged')[0])
    iterations = int(r_list_get(r_result, 'iterations')[0])
    # ver4-58: did this fit only succeed through the ai.sing retry? (The R
    # side only reports the key from asreml_fit_and_predict(); the
    # constrained-fit path never retries.) Defensive: any shape -> False.
    try:
        _ai_flag = r_list_get(r_result, 'ai_sing_used', default=None)
        ai_sing_used = bool(_ai_flag[0]) if _ai_flag is not None else False
    except Exception:
        ai_sing_used = False
    timing = _record_fit_timing(
        role, _fit_scope_hash(trait, working['ID'].astype(str).unique().tolist(), [], []),
        working.shape[0], working['ID'].nunique(), _fit_seconds, iterations, converged,
    )

    id_series = _extract_estimand_column(pvals, (genotype_col,), 'genotype/ID', log_prefix).astype(str)
    value_series = _extract_estimand_column(pvals, _VALUE_COL_CANDIDATES, 'predicted value', log_prefix)
    try:
        se_series = _extract_estimand_column(pvals, _SE_COL_CANDIDATES, 'standard error', log_prefix)
    except KeyError:
        se_series = pd.Series([np.nan] * len(pvals), index=pvals.index)

    _genotype_effect = asreml_cfg.get('genotype_effect', 'random')
    estimand_label = asreml_syntax.estimand(asreml_cfg)

    values_cols = {
        'id': id_series,
        'value': value_series.to_numpy(),
        'estimand': estimand_label,
        'std_error': se_series.to_numpy(),
        'split': 'all',
    }
    if _by_population:
        population_series = _extract_estimand_column(
            pvals, (population_col,), 'population', log_prefix
        ).astype(str)
        values_cols['population'] = population_series
    values = pd.DataFrame(values_cols)
    if _by_population:
        values = values[['id', 'population', 'split', 'value', 'estimand', 'std_error']]
    else:
        values = values[['id', 'split', 'value', 'estimand', 'std_error']]

    # ver4-25, R8: `pred$vcov` (when `return_vcov`) is aligned to
    # `pvals`'s own row order - the SAME order `id_series`/
    # `population_series` were just built in. `vcov_ids` carries the
    # SAME row-identity `values` itself uses: a plain id string when
    # `response_unit == 'genotype'`, or an `(id, population)` tuple when
    # `response_unit == 'genotype_by_population'` (ids alone are not
    # unique across populations in that case) - `restrict_vcov_to_
    # population()` below is the single place that later resolves either
    # shape down to a plain, population-restricted id list a
    # `BlueWeightContext` can use (mirrors `_asreml_values_lookup()`'s/
    # `_asreml_se_lookup()`'s own "filter by population column when
    # present, then key by plain id" convention in genomic_prediction.py).
    vcov_arr, vcov_ids = None, None
    if return_vcov:
        raw_vcov = r_list_get(r_result, 'vcov')
        vcov_arr = _maybe_mmap_vcov(np.asarray(raw_vcov, dtype=float), log_prefix=log_prefix)
        vcov_ids = (list(zip(id_series.tolist(), population_series.tolist())) if _by_population
                    else id_series.tolist())
        if vcov_arr.shape[0] != len(vcov_ids) or vcov_arr.shape[1] != len(vcov_ids):
            raise ValueError(
                f"{log_prefix} fit_unmasked_adjustment: predict(..., vcov=TRUE) returned a "
                f"{vcov_arr.shape} matrix, but predict()$pvals has {len(vcov_ids)} row(s) - "
                f"these should always match (ASReml-R's own documented predict() contract) - "
                f"refusing to use a covariance matrix that cannot be reliably aligned to "
                f"individual IDs."
            )

    # RK-2-style post-condition: never let a NaN adjusted value reach any
    # model silently. ver4-20 follow-up: when `mask_keys` is in effect, a
    # NaN is expected to be POSSIBLE for a masked key (see this
    # function's own docstring - a FIXED genotype term with no other,
    # unmasked observation anywhere is genuinely inestimable) - but it is
    # never acceptable for an UNMASKED key, which would indicate the same
    # kind of structural problem this check has always guarded against.
    if values['value'].isna().any():
        _nan_bool = values['value'].isna()
        if mask_keys:
            if _by_population:
                _val_keys = list(zip(values['id'].astype(str), values['population'].astype(str)))
                _mask_key_set_check = {(str(g), str(p)) for g, p in mask_keys if p is not None}
            else:
                _val_keys = list(values['id'].astype(str))
                _mask_key_set_check = {str(g) for g, _ in mask_keys}
            _nan_keys = [k for k, is_nan in zip(_val_keys, _nan_bool) if is_nan]
            _nan_unmasked = [k for k in _nan_keys if k not in _mask_key_set_check]
            if _nan_unmasked:
                raise ValueError(
                    f"{log_prefix} fit_unmasked_adjustment: {len(_nan_unmasked)} UNMASKED "
                    f"individual(s)/cell(s) - {_nan_unmasked[:10]}{'...' if len(_nan_unmasked) > 10 else ''} "
                    f"- have a NaN adjusted phenotype value for trait {trait!r} - refusing to "
                    f"let this reach any model. This is the same class of problem as the "
                    f"unmasked engine's original check (a genuinely NaN raw response with no "
                    f"other contributing record for that genotype, or an aliased/inestimable "
                    f"effect) - investigate this trait's own design before proceeding."
                )
            raise ValueError(
                f"{log_prefix} fit_unmasked_adjustment: {len(_nan_keys)} MASKED validation/test "
                f"individual(s)/cell(s) have an inestimable (NaN) adjusted phenotype for trait "
                f"{trait!r} once their own response is set to NA (ASREML['mask_valid_test']="
                f"True). This is expected for a FIXED (BLUEs) genotype term whenever that "
                f"genotype has no OTHER, unmasked observation anywhere in the raw phenotype "
                f"file - ASReml cannot estimate a fixed effect from zero informative rows. "
                f"Fix by disabling 'Mask validation/test phenotypes with NA' for this run, "
                f"switching to a BLUPs strategy (which predicts a masked individual from its "
                f"genomic relationship/replication structure instead of requiring its own "
                f"data), or confirming this genotype also appears, unmasked, in another "
                f"population/replicate."
            )
        # ver4-21 BUGFIX (production failure - "60 individual(s)/cell(s)
        # have a NaN adjusted phenotype value ... unlikely once
        # na.action='include' is in effect"): that assumption was wrong,
        # and this hard-fail was firing on ordinary, harmless data rather
        # than a genuine design problem. `raw_records` is deliberately the
        # WHOLE trial's raw file across every trait (this function's own
        # docstring, "ordinarily genomic_prediction.py's own
        # data_phenotype_raw_for_asreml") - it routinely contains
        # genotypes that were phenotyped for OTHER traits but have ZERO
        # raw rows with a non-missing response for THIS trait at all. For
        # such a genotype the sparse/fixed genotype term is genuinely,
        # correctly inestimable - exactly the same "no informative rows"
        # situation as an ordinary missing-phenotype individual anywhere
        # else in the pipeline (architecture doc SS4.2's geno x phenotype
        # `dropna()`), not an aliasing/design defect. And it is provably
        # harmless downstream: `collapse_raw_to_analysed_level()`'s own
        # placeholder mean is ALSO NaN for such a genotype (a mean of zero
        # values), so the normal geno x phenotype merge already drops it
        # from every task's `id_train`/`id_valid`/`id_test` for THIS
        # trait, long before `apply_to_splits()` would ever look this key
        # up - a NaN left here for it is simply never read.
        #
        # Distinguish that expected case from a genotype that DOES have
        # at least one non-missing raw observation for this trait yet
        # still comes back NaN - THAT remains the real "investigate the
        # design" signal this check exists to catch (e.g. a factor level
        # confounded with the genotype term), and still raises.
        if _by_population:
            _nan_val_keys = list(zip(
                values.loc[_nan_bool, 'id'].astype(str),
                values.loc[_nan_bool, 'population'].astype(str),
            ))
            _has_obs_keys = {
                (str(_g), str(_p)) for _g, _p in
                working.loc[working[response_col].notna(), [genotype_col, population_col]]
                       .drop_duplicates().itertuples(index=False, name=None)
            }
        else:
            _nan_val_keys = list(values.loc[_nan_bool, 'id'].astype(str))
            _has_obs_keys = set(
                working.loc[working[response_col].notna(), genotype_col].astype(str)
            )
        _nan_with_data = [k for k in _nan_val_keys if k in _has_obs_keys]
        if _nan_with_data:
            raise ValueError(
                f"{log_prefix} fit_unmasked_adjustment: {len(_nan_with_data)} "
                f"individual(s)/cell(s) - {_nan_with_data[:10]}{'...' if len(_nan_with_data) > 10 else ''} "
                f"- have a NaN adjusted phenotype value for trait {trait!r} EVEN THOUGH at "
                f"least one non-missing raw observation exists for them - refusing to let this "
                f"reach any model. Since this is NOT a genotype with zero raw data for this "
                f"trait, this points to an aliased/inestimable effect (e.g. a factor level "
                f"represented only by this genotype, confounded with the genotype term) rather "
                f"than genuinely missing data - investigate this trait's own design before "
                f"proceeding."
            )
        _n_nan = int(_nan_bool.sum())
        print(
            f"{log_prefix} fit_unmasked_adjustment: {_n_nan} individual(s)/cell(s) have NO "
            f"non-missing raw observation at all for trait {trait!r} (phenotyped for other "
            f"trait(s)/population(s) only, or genuinely never recorded) - their adjusted "
            f"phenotype is genuinely inestimable and is left as NaN, exactly like an ordinary "
            f"missing-phenotype individual elsewhere in the pipeline. This is harmless: the "
            f"same placeholder-mean collapse that feeds this trait's own train/valid/test "
            f"membership (pipeline_utils.collapse_raw_to_analysed_level()) is NaN for them too, "
            f"so they are already excluded from every task that predicts this trait, long "
            f"before this value would ever be looked up."
        )
    _dupe_key = ['id', 'population'] if _by_population else ['id']
    _dupe_counts = values.groupby(_dupe_key).size()
    _dupes = _dupe_counts[_dupe_counts > 1]
    if not _dupes.empty:
        raise ValueError(
            f"{log_prefix} fit_unmasked_adjustment: {_dupes.shape[0]} {_dupe_key} combination(s) "
            f"appear more than once in predict()$pvals for trait {trait!r} - this usually means "
            f"ASREML['output_level'] does not take a single value per {' x '.join(_dupe_key)}."
        )

    heritability, genetic_row_index = _estimate_heritability(varcomp, genotype_col, _genotype_effect)
    # Classic heritability from THIS fit's own variance components and the
    # records it actually used (`working` - response already masked to NA
    # where `mask_keys` applied) - reported next to the Cullis heritability
    # computed from this same fit (genomic_prediction.py's accuracy-
    # metrics context -> Metric.csv `Heritability (classic)`). Computed
    # from the RAW varcomp, BEFORE the 'heritability' column is added
    # below, purely diagnostic, `None` for a fixed (BLUE) genotype term.
    heritability_classic = (
        classic_heritability_from_fit(varcomp, working, response_col, genotype_col, log_prefix)
        if _genotype_effect == 'random' else None
    )
    varcomp = varcomp.copy()
    varcomp['heritability'] = np.nan
    if genetic_row_index is not None:
        varcomp.loc[genetic_row_index, 'heritability'] = heritability

    return AsremlResult(values=values, varcomp=varcomp, heritability=heritability,
                         iterations=iterations, converged=converged, timing=timing,
                         vcov=vcov_arr, vcov_ids=vcov_ids,
                         heritability_classic=heritability_classic,
                         ai_sing_used=ai_sing_used, borrowed_variance=_use_fixed)


def _harmonic_mean(counts: "pd.Series") -> "Optional[float]":
    """Harmonic mean of a `pandas.Series` of positive counts - `None`
    (never raises, never returns 0/inf) for an empty or all-non-positive
    input. Shared by `fit_concurrent_variance_model()`'s `reps_harmonic_
    mean`/GxE-level-count computations below - both are the SAME
    "harmonic mean of an unbalanced per-genotype count" operation, just
    over different underlying counts."""
    counts = counts[counts > 0]
    if counts.empty:
        return None
    return float(len(counts) / (1.0 / counts).sum())


def _harmonic_mean_ge_levels(raw_records: "pd.DataFrame", ids_train: Sequence[str],
                              factor_raw_label: "Optional[str]") -> "Optional[float]":
    """Harmonic mean, across TRAINING genotypes only, of each genotype's
    own number of DISTINCT levels of `factor_raw_label` (e.g.
    'Environment') actually observed in `raw_records` - the `ē_h` R10/Q4
    needs to divide a genotype-by-<factor> variance component by, when
    computing `H2_classic`'s optional GxE term. `factor_raw_label` is
    matched against `raw_records`' own columns CASE-INSENSITIVELY (it
    comes from splitting an ASReml varcomp component's own raw label on
    `':'`, which does not necessarily preserve the original column's
    exact case). Returns `None` (never raises) when no matching column is
    found, or nothing can be computed - the caller treats that as "omit
    the GxE term, with a NOTE" (design blueprint R10, Q4's own documented
    fallback), never as a fatal error."""
    if not factor_raw_label:
        return None
    match = next((c for c in raw_records.columns if c.lower() == str(factor_raw_label).lower()), None)
    if match is None:
        return None
    subset = raw_records[raw_records['ID'].astype(str).isin(set(str(v) for v in ids_train))]
    if subset.empty:
        return None
    level_counts = subset.groupby(subset['ID'].astype(str))[match].nunique()
    return _harmonic_mean(level_counts)


def _classic_heritability_from_components(
    canonical: "pd.Series", components_raw: "pd.Series", estimates: "pd.Series",
    genetic_variance: "Optional[float]", residual_variance: "Optional[float]",
    reps_harmonic_mean: "Optional[float]", ge_records: "pd.DataFrame",
    ids_train: Sequence[str], genotype_col: "Optional[str]", log_prefix: str,
    unaffected_note: str = "h2_plot/H2_cullis_train are unaffected",
) -> "Optional[float]":
    """The single authority for the CLASSIC heritability formula (design
    blueprint R10/Q4): `H2_classic = sigma_g^2 / (sigma_g^2 + sigma_e^2 /
    r_bar [+ sigma_ge^2 / e_bar])`, where `r_bar` is the harmonic mean of
    each genotype's own record count (`reps_harmonic_mean`) and `e_bar`
    the harmonic mean of each genotype's own number of distinct levels of
    the ONE genotype-by-<factor> term, when exactly one exists.

    Extracted verbatim from `fit_concurrent_variance_model()` (which now
    calls this) so the `H2_classic` in `ASReml_variance_components.csv`
    and the `Heritability (classic)` column in `Metric.csv`
    (`classic_heritability_from_fit()` below) can never disagree about
    what "classic" means - the same one-place-decides discipline
    `_classify_vparameter()` already applies.

    Returns `None` (never raises) when a required quantity is missing or
    the denominator is non-positive. Every NOTE printed here is the same
    text `fit_concurrent_variance_model()` always printed;
    `unaffected_note` only names which sibling statistics stay unaffected
    when the GxE contribution has to be omitted.
    """
    ge_terms = sorted(set(canonical[canonical.str.startswith('genotype_by_')])) \
        if canonical.shape[0] else []
    H2_classic = None
    if genetic_variance is not None and residual_variance is not None and reps_harmonic_mean:
        denominator = genetic_variance + (residual_variance / reps_harmonic_mean)
        if len(ge_terms) == 1:
            ge_mask = canonical == ge_terms[0]
            ge_variance = float(estimates[ge_mask].iloc[0])
            ge_raw_examples = components_raw[ge_mask]
            ge_other_token = None
            if not ge_raw_examples.empty:
                parts = [p.strip() for p in str(ge_raw_examples.iloc[0]).split(':')]
                others = [p for p in parts if str(genotype_col).lower() not in p.lower()]
                if len(others) == 1:
                    ge_other_token = others[0]
            ge_harmonic = _harmonic_mean_ge_levels(ge_records, ids_train, ge_other_token)
            if ge_harmonic:
                denominator += ge_variance / ge_harmonic
            else:
                print(f"{log_prefix} NOTE: could not compute a harmonic-mean level count for "
                      f"the identified genotype-by-environment term {ge_terms[0]!r} - "
                      f"H2_classic omits its variance contribution for this task "
                      f"({unaffected_note}).")
        elif len(ge_terms) > 1:
            print(f"{log_prefix} NOTE: {len(ge_terms)} distinct genotype-by-<factor> random "
                  f"terms were identified ({ge_terms}) - H2_classic (design blueprint Q4) is "
                  f"only defined when EXACTLY ONE such term exists, so its GxE contribution is "
                  f"omitted for this task ({unaffected_note}).")
        if denominator > 0:
            H2_classic = genetic_variance / denominator
    return H2_classic


def classic_heritability_from_fit(
    varcomp: "pd.DataFrame", working: "pd.DataFrame", response_col: str,
    genotype_col: "Optional[str]", log_prefix: str = '[ASReml]',
) -> "Optional[float]":
    """Classic heritability for ONE already-fitted model, from its own
    raw `summary(fit)$varcomp` frame (`varcomp`: `component`, `estimate`,
    ...) and the exact records that fit used (`working`: the response-
    masked working frame `fit_unmasked_adjustment()` builds, with the
    response in `response_col` and NA wherever a row was masked or is a
    grid filler).

    Replicate counts come only from rows with a NON-missing response, so
    a masked validation/test individual (response NA) contributes nothing
    - the same "training individuals only" rule
    `fit_concurrent_variance_model()` applies through `id_train`. The
    genotype-by-<factor> level count, when that term exists, uses those
    same individuals' rows, mirroring `fit_concurrent_variance_model()`.

    Purely diagnostic: any failure is swallowed and returned as `None`
    (never raises), so it can never fail a fit or a task. `None` also for
    a FIXED (BLUE) genotype term, which has no genetic variance
    component to build a heritability from.
    """
    try:
        if varcomp is None or varcomp.shape[0] == 0 or not genotype_col:
            return None
        components_raw = varcomp['component'].astype(str).reset_index(drop=True)
        estimates = varcomp['estimate'].astype(float).reset_index(drop=True)
        canonical = components_raw.apply(lambda c: _classify_vparameter(c, genotype_col))
        genetic_mask = canonical == 'genotype'
        residual_mask = canonical == 'residual'
        genetic_variance = float(estimates[genetic_mask].iloc[0]) if genetic_mask.any() else None
        residual_variance = float(estimates[residual_mask].iloc[0]) if residual_mask.any() else None
        if genetic_variance is None or residual_variance is None:
            return None

        id_col = 'ID' if 'ID' in working.columns else genotype_col
        fitted = working[working[response_col].notna() & working[id_col].notna()]
        if fitted.shape[0] == 0:
            return None
        fitted_ids = sorted(set(fitted[id_col].astype(str)))
        reps_per_genotype = fitted.groupby(fitted[id_col].astype(str)).size()
        reps_harmonic_mean = _harmonic_mean(reps_per_genotype)
        ge_records = working if id_col == 'ID' else working.rename(columns={id_col: 'ID'})
        return _classic_heritability_from_components(
            canonical, components_raw, estimates, genetic_variance, residual_variance,
            reps_harmonic_mean, ge_records, fitted_ids, genotype_col, log_prefix,
            unaffected_note="the Cullis heritability and every other metric are unaffected",
        )
    except Exception as exc:  # diagnostic only - never fail a fit over it
        print(f"{log_prefix} NOTE: classic heritability could not be computed for this fit "
              f"({exc!r}) - its 'Heritability (classic)' value will be NaN; the Cullis "
              f"heritability and every other metric are unaffected.")
        return None


def fit_blue_refinement_step1(
    raw_records: "pd.DataFrame", trait: str, genotype_col: str,
    asreml_cfg: dict, universe_ids: Sequence[str],
    *, grm_inv, grm_ids: Sequence[str],
    workspace: str = '4gb', max_update_rounds: int = 20,
    log_prefix: str = '[ASReml]', threads: "Optional[float]" = None,
) -> Dict[str, float]:
    """ver4-25, R7 (design blueprint SS "Chosen approach", Holland &
    Piepho 2024 Fig. 2) - step (i) of the two-step refined-BLUE
    procedure: fit genotype RANDOM with a genomic relationship matrix
    (`A ~ N(0, Gσ²_A)`, Q3's confirmed decision) on every individual in
    `universe_ids` (ordinarily every genotype the run's own GRM was built
    for - see `genomic_prediction.py::_asreml_build_grm_for_universe()`),
    UNMASKED, reusing the SAME non-genetic design terms `asreml_cfg`
    already configures for the eventual genotype-FIXED main fit (only
    the genotype term itself differs - random+GRM here, fixed there).

    This is deliberately a THIN, single-purpose wrapper around `_fit_
    single_asreml_model()` - it does not itself decide caching (the
    caller hoists/caches this once per trait, exactly like the main
    unmasked fit - see `genomic_prediction.py::_asreml_get_blue_
    refinement_step1()`) or excluded-fraction preflighting (the caller's
    own `_asreml_build_grm_for_universe()` already produces those
    diagnostics for the main GRM build, and R7 reuses the SAME universe/
    diagnostics rather than re-deriving them - design blueprint K8).

    Returns
    -------
    `{component_raw: estimate}` - EVERY row of the converged fit's own
    `summary(fit)$varcomp`, by its RAW (not canonical) component label,
    since step (ii)'s own R-side `sv_table$Component` match must be
    against ASReml-R's own free-text names, not this module's
    classification of them. The genotype term's own row IS included
    (step (ii)'s own R function simply finds no matching row to fix for
    it, since step (ii) has no variance parameter of that name at all -
    see `Preprocess/asreml_fit.R::asreml_fit_fixed_with_constrained_
    vparameters()`'s own "left exactly as asreml() itself proposed it"
    comment).

    Raises
    ------
    ValueError/RuntimeError on a genuine fit failure - the caller is
    responsible for catching these and falling back to the UNREFINED
    single fit with a `[GP] NOTE:` (this is a best-effort quality
    improvement, never a hard requirement for BLUE to be computable at
    all).
    """
    step1_cfg = dict(asreml_cfg)
    step1_cfg['genotype_effect'] = 'random'
    step1_cfg['use_grm'] = True
    spec = asreml_syntax.build_model_spec(step1_cfg, trait)
    step1_syntax = asreml_syntax.render_syntax(spec, data_symbol='{DATA}', grm_symbol='{GRM}')

    ids = sorted(set(str(v) for v in universe_ids))
    result = _fit_single_asreml_model(
        raw_records, trait, ids, [], [], step1_cfg, step1_syntax,
        grm_inv=grm_inv, workspace=workspace, max_update_rounds=max_update_rounds,
        log_prefix=f"{log_prefix} [blue_refinement/step1]", grm_ids=grm_ids,
        role='blue_refinement_step1', threads=threads,
    )
    varcomp = result.varcomp
    return dict(zip(varcomp['component'].astype(str), varcomp['estimate'].astype(float)))


def fit_concurrent_variance_model(
    train_raw_records: "pd.DataFrame", trait: str, genotype_col: str,
    asreml_cfg: dict, covariance: str = 'iid',
    *, grm_inv=None, grm_ids: "Optional[Sequence[str]]" = None,
    workspace: str = '4gb', max_update_rounds: int = 20,
    log_prefix: str = '[ASReml]',
    id_train: "Optional[Sequence[str]]" = None,
    full_raw_records: "Optional[pd.DataFrame]" = None,
    mask_non_training: "Optional[bool]" = None,
    role: str = 'concurrent_iid', threads: "Optional[float]" = None,
):
    """Requirements.md R2 SS2.6 / ver4-25 R10 redefinition (design
    blueprint stage 3) - the training-only side model that reports
    `ASReml_variance_components.csv`'s ONE row-set per task. Fits
    genotype as RANDOM - `covariance='iid'` (no GRM, R10's own new
    default and the ONLY mode `genomic_prediction.py` now calls this
    with) or `'grm'` (kept for a hand-written caller; not used by the
    redefined varcomp file) - on training individuals' responses ONLY,
    reusing the SAME fixed/random/residual design terms `asreml_cfg`
    already configures (only the genotype term itself differs).

    ver4-25, R10 (design blueprint stage 3, "Variance-components file
    redefined"): this is now called for EVERY strategy/task whenever
    ASREML is enabled at all - not just `blue_fixed` behind an explicit
    opt-in - and its OWN varcomp is the entire content of
    `ASReml_variance_components.csv` (the previous main-fit/companion
    varcomp rows are no longer written at all - see `genomic_
    prediction.py`'s own Step 6b comment). This function's returned
    `AsremlResult.varcomp` is therefore no longer the bare `summary(fit)
    $varcomp` frame `_fit_single_asreml_model()` produces - it is
    ENRICHED, here, into the file's own final schema (canonical
    `component`, `proportion_of_total`, `h2_plot`, `H2_classic`,
    `H2_cullis_train`, and the training counts those two heritabilities
    are computed from), so `genomic_prediction.py` only has to broadcast
    the scenario columns onto it and concatenate - exactly the same
    "the source module owns its own file's shape" discipline `_fit_
    single_asreml_model()` already applies to `heritability`.

    Grid-structured residuals (`asreml_syntax.residual_requires_full_
    grid()`): previously refused outright (a `ValueError` the caller
    turned into a skip-with-NOTE). Now handled by reusing the EXACT SAME
    "keep the full grid, mask non-training responses with NA" mechanism
    `_fit_single_asreml_model()` already implements for `id_valid`/
    `id_test` (ver4-17's own "exterior rows keep design, response
    masked" principle) - `full_raw_records` (the WHOLE trial, every
    population) is required whenever this applies; `mask_non_training`
    (`None` -> auto-detect from the grid flag) can force either
    behaviour explicitly.

    Parameters
    ----------
    train_raw_records : this task's own TRAINING individuals' raw
        records only (used directly when no grid residual is
        configured, and as the fallback source of `id_train`/the
        harmonic-mean/genotype-count statistics either way).
    id_train : this task's own training individual ids. `None` (the
        pre-ver4-25 default) falls back to every distinct `'ID'` in
        `train_raw_records` - unchanged behaviour for the non-grid case.
    full_raw_records : the WHOLE trial's raw records (every population) -
        required when `mask_non_training` resolves `True`; every
        individual OUTSIDE `id_train` has their own response masked to
        NA before fitting (design columns untouched), exactly like a
        validation/test individual elsewhere in this module.
    mask_non_training : `None` (default) -> `True` iff `asreml_syntax.
        residual_requires_full_grid(asreml_cfg)`; an explicit `True`/
        `False` overrides the auto-detection.
    role : ver4-25, R9-S0 - forwarded to `_fit_single_asreml_model()`.

    This function is purely a SIDE, diagnostic computation - it never
    returns a value that overwrites any model's own training target, and
    a failure here must never fail the whole task (see this function's
    own caller for the non-fatal `try`/`except` wrapping - mirroring the
    "informational side computation, never blocks the run" precedent
    architecture doc SS10.3 already documents for the surrogate
    interaction routes).

    Returns
    -------
    `AsremlResult` whose `.varcomp` is the file-ready, enriched frame
    described above.

    Raises
    ------
    ValueError if `mask_non_training` resolves `True` but `full_raw_
    records` is not supplied (a caller contract violation, not a data
    problem - `genomic_prediction.py` always supplies it). RuntimeError
    on a genuine R fit failure - the caller's own non-fatal wrapper turns
    this into a `[GP] NOTE:` rather than failing the task.
    """
    if covariance not in ('iid', 'grm'):
        raise ValueError(
            f"{log_prefix} fit_concurrent_variance_model: covariance={covariance!r} must be "
            f"'iid' or 'grm'."
        )
    ids_train = sorted(set(
        str(v) for v in (id_train if id_train is not None else train_raw_records.get('ID', []))
    ))

    needs_grid = asreml_syntax.residual_requires_full_grid(asreml_cfg)
    use_mask_non_training = needs_grid if mask_non_training is None else bool(mask_non_training)

    if use_mask_non_training:
        if full_raw_records is None:
            raise ValueError(
                f"{log_prefix} fit_concurrent_variance_model: this task's residual structure "
                f"needs the trial's complete physical grid (or mask_non_training=True was "
                f"forced), but no full_raw_records was supplied - this is a caller contract "
                f"violation (genomic_prediction.py should always pass the whole-trial table in "
                f"this case), not a data problem."
            )
        fit_records = full_raw_records
        non_training_ids = sorted(
            set(str(v) for v in fit_records['ID'].astype(str)) - set(ids_train)
        )
    else:
        fit_records = train_raw_records
        non_training_ids = []

    concurrent_cfg = dict(asreml_cfg)
    concurrent_cfg['genotype_effect'] = 'random'
    concurrent_cfg['use_grm'] = (covariance == 'grm')
    # response_unit is deliberately NOT threaded through here - the
    # concurrent model's only job is a training-only variance-component
    # estimate, which needs one genetic variance term, not a per-
    # population breakdown (R3 is about the MAIN adjustment's response
    # unit, not this side computation).
    spec = asreml_syntax.build_model_spec(concurrent_cfg, trait)
    concurrent_syntax = asreml_syntax.render_syntax(
        spec, data_symbol='{DATA}', grm_symbol='{GRM}' if covariance == 'grm' else None
    )

    result = _fit_single_asreml_model(
        fit_records, trait, ids_train, [], non_training_ids, concurrent_cfg, concurrent_syntax,
        grm_inv=grm_inv, workspace=workspace, max_update_rounds=max_update_rounds,
        log_prefix=f"{log_prefix} [concurrent/{covariance}]", grm_ids=grm_ids, role=role,
        threads=threads,
    )

    # --- R10: enrich the raw varcomp frame into the file's own schema --- #
    raw_varcomp = result.varcomp
    components_raw = (
        raw_varcomp['component'].astype(str) if raw_varcomp.shape[0] else pd.Series(dtype=str)
    )
    estimates = raw_varcomp['estimate'].astype(float) if raw_varcomp.shape[0] else pd.Series(dtype=float)
    canonical = components_raw.apply(lambda c: _classify_vparameter(c, genotype_col))

    genetic_mask = canonical == 'genotype'
    residual_mask = canonical == 'residual'
    genetic_variance = float(estimates[genetic_mask].iloc[0]) if genetic_mask.any() else None
    residual_variance = float(estimates[residual_mask].iloc[0]) if residual_mask.any() else None

    h2_plot = None
    if genetic_variance is not None and residual_variance is not None \
            and (genetic_variance + residual_variance) > 0:
        h2_plot = genetic_variance / (genetic_variance + residual_variance)

    # Training-only counts (I7: never derived from valid/test rows).
    train_records_for_counts = train_raw_records[
        train_raw_records['ID'].astype(str).isin(set(ids_train))
    ] if 'ID' in train_raw_records.columns else train_raw_records
    if trait in train_records_for_counts.columns:
        train_records_for_counts = train_records_for_counts[
            train_records_for_counts[trait].notna()
        ]
    n_train_genotypes = len(ids_train)
    n_train_records = int(train_records_for_counts.shape[0])
    reps_per_genotype = (
        train_records_for_counts.groupby(train_records_for_counts['ID'].astype(str)).size()
        if n_train_records else pd.Series(dtype=int)
    )
    reps_harmonic_mean = _harmonic_mean(reps_per_genotype)

    H2_classic = _classic_heritability_from_components(
        canonical, components_raw, estimates, genetic_variance, residual_variance,
        reps_harmonic_mean, train_raw_records, ids_train, genotype_col, log_prefix,
    )

    train_values = result.values[
        result.values['id'].astype(str).isin(set(ids_train))
    ] if result.values.shape[0] else result.values
    H2_cullis_train = asreml_accuracy_metrics.cullis_heritability(
        raw_varcomp, train_values, genotype_col,
    )

    total_estimate = float(estimates.sum()) if estimates.shape[0] else 0.0
    proportion_of_total = (
        (estimates / total_estimate) if total_estimate > 0 else pd.Series([np.nan] * len(estimates))
    )

    enriched_varcomp = pd.DataFrame({
        'component': canonical.reset_index(drop=True),
        'component_raw': components_raw.reset_index(drop=True),
        'estimate': estimates.reset_index(drop=True),
        'std_error': (
            raw_varcomp['std_error'].astype(float).reset_index(drop=True)
            if 'std_error' in raw_varcomp.columns else pd.Series([np.nan] * len(estimates))
        ),
        'bound': (
            raw_varcomp['bound'].reset_index(drop=True)
            if 'bound' in raw_varcomp.columns else pd.Series([None] * len(estimates))
        ),
        'proportion_of_total': proportion_of_total.reset_index(drop=True),
    })
    enriched_varcomp['converged'] = result.converged
    enriched_varcomp['iterations'] = result.iterations
    enriched_varcomp['n_train_genotypes'] = n_train_genotypes
    enriched_varcomp['n_train_records'] = n_train_records
    enriched_varcomp['reps_harmonic_mean'] = reps_harmonic_mean
    enriched_varcomp['h2_plot'] = h2_plot
    enriched_varcomp['H2_classic'] = H2_classic
    enriched_varcomp['H2_cullis_train'] = H2_cullis_train

    return AsremlResult(
        values=result.values, varcomp=enriched_varcomp, heritability=h2_plot,
        iterations=result.iterations, converged=result.converged, timing=result.timing,
    )
