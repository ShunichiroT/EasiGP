"""
Preprocess/asreml_license.py
==============================
Update ID ver4-12, R13 (design blueprint SS2.13).

`asreml` is a commercial R package - absent from CRAN and conda-forge,
never pinned in `environment_linux.yml` (invariant I12), and requiring
per-machine licence activation on top of ordinary installation. Without
this module, a missing installation or an unlicensed one surfaces as a
raw `rpy2`/R error in the middle of a run - often on an HPC compute node,
hours into a job, after every other model has already finished.

This module follows `Preprocess/flash_p_integration.py::preflight_check()`'s
established shape exactly (a near-free sanity check performed BEFORE the
expensive thing it protects), and is Layer 2 of R13's three gating layers:

  Layer 1 - main_app.py's Tab 2 ASReml section: a live status badge next
            to the master checkbox, and an activation panel when
            unlicensed (never a silent failure).
  Layer 2 - THIS module's preflight_check(), called from
            genomic_prediction.py::GP() before task 0, exactly where the
            existing model_registry.is_available() Tier-2-optional-
            dependency check already runs (same "fail fast, before any
            task starts" principle) - gated by ASREML_STRICT (default
            True), mirroring MODEL_AVAILABILITY_STRICT.
  Layer 3 - Preprocess/asreml_blup.py raises with an explicit message if
            called anyway, despite Layers 1/2 somehow being bypassed
            (e.g. a hand-edited config run directly against GP()).

Deliberately NOT folded into model_registry.OPTIONAL_DEPENDENCY (which
maps MODEL NAMES to importable PYTHON packages - `asreml` is neither a
model name nor merely importable, it is also licensed) - see that
module's own one-line cross-reference comment pointing here.

VERSION CAVEAT (Appendix B, check B3)
--------------------------------------------------------------------------
The exact ASReml-R 4 licence-status API (`asreml.license.status()` /
`asreml.license.activate()`) is asserted here as documented, but this
module could NOT be executed against a real, licensed installation in
this environment (none is available here). `license_status()` never
raises regardless of whether these exact calls exist in a given
installation - a `TypeError`/`AttributeError` from a genuinely different
API surface is caught and folded into the returned status dict's own
`message`, so a version mismatch degrades to "could not determine license
status" rather than crashing the preflight check itself. Phase 3 must
confirm the exact call names against the licensed build in use (see the
ver4-12 Change Summary SS8) before relying on the `licensed`/`expiry`
fields for anything beyond that degraded fallback.
"""

from __future__ import annotations

from typing import Optional


def _run_r_snippet(r_code: str, r_path: "Optional[str]" = None):
    """Run a short R snippet through the SAME rpy2 bridge every other R
    interaction in this codebase uses, returning whatever R object the
    snippet's last expression evaluates to. Raises on any failure (import,
    R-side error) - callers here are all internally wrapped in their own
    try/except, per this module's own "license_status() never raises"
    contract; `preflight_check()` is the one function that intentionally
    lets an exception propagate (see its own docstring).

    Requirements.md item 3 (bug fix): every OTHER R entry point in this
    codebase (`genomic_prediction.py::GP()`, every headless runner) calls
    `pipeline_utils.init_rpy2_conversion()` once, near the top, before any
    `rpy2.robjects` call - see the architecture doc's "R environment
    configuration must precede the genomic_prediction import" note. This
    module's own Layer 1 (the GUI's live status badge in Tab 2) is called
    directly from Streamlit's own script-execution thread, though, with
    no guarantee that thread has ever run `init_rpy2_conversion()` itself
    - rpy2's R<->Python conversion rules are held in a
    `contextvars.ContextVar`, which is NOT automatically inherited by a
    new thread (Streamlit re-runs the app script in its own thread per
    session), so the very first ASReml-section render in a fresh session
    hit rpy2 with no conversion rules registered in ITS thread/context at
    all - regardless of whether `asreml` itself was installed correctly.
    That surfaced as `NotImplementedError('Conversion rules for
    rpy2.robjects appear to be missing ...')`, reported unconditionally
    as "the R bridge itself failed", which was misleading (nothing about
    the R installation or the `asreml` package itself was actually
    broken). Calling `init_rpy2_conversion()` here - immediately before
    the only `rpy2.robjects` use in this module, in the SAME call stack/
    thread that is about to use it - makes this function self-sufficient
    regardless of which thread Streamlit happens to run it in, and
    regardless of whether some other entry point has already initialised
    conversion in a different thread. Cheap and idempotent - safe to call
    on every invocation.
    """
    import os
    if r_path:
        os.environ['R_HOME'] = r_path
    import rpy2.robjects as robjects
    try:
        from pipeline_utils import init_rpy2_conversion, ensure_r_native_symbols_global
        init_rpy2_conversion()
        # BUGFIX (undefined symbol: R_NilValue) - apply the SAME libR.so
        # global-symbol-visibility fix this preflight check's own real
        # counterpart (Preprocess/asreml_blup.py, immediately before its
        # `library(asreml)`-triggering r_source() call) applies, so that
        # THIS check's `requireNamespace('asreml', ...)` result reflects
        # what will actually happen at fit time on the same machine,
        # rather than passing here (requireNamespace() swallows a load
        # failure and returns FALSE, so it would actually have correctly
        # reported 'not installed' - but only once this fix makes its own
        # verdict deterministic rather than a race against whatever else
        # happens to have already loaded a shared object in this process)
        # and failing later, per task, inside Step 6b - see
        # pipeline_utils.ensure_r_native_symbols_global()'s own docstring.
        ensure_r_native_symbols_global()
    except ImportError:
        # pipeline_utils not importable from wherever this process's
        # sys.path currently points (e.g. this module imported standalone,
        # outside the normal easigp/ package layout) - fall back to rpy2's
        # own default conversion context rather than failing this check
        # outright; the NotImplementedError this guards against is the
        # much more common failure mode.
        pass
    return robjects.r(r_code)


def license_status(r_path: "Optional[str]" = None) -> dict:
    """Best-effort, NEVER-RAISING check of whether `asreml` is installed
    and licensed in the R environment `configure_r_environment()`/
    `r_path` would point at.

    Returns
    -------
    dict with:
        'installed' : bool - `requireNamespace('asreml', quietly=TRUE)`
            succeeded.
        'licensed'  : bool or None - True/False if a licence-status call
            succeeded and reported a definite answer; None if the package
            isn't installed, or the licence-status call itself failed
            (see 'message').
        'version'   : str or None - the installed package version, if
            determinable.
        'expiry'    : str or None - licence expiry date, if the installed
            build's licence-status API reports one in a recognisable
            shape (see the module-level version-caveat note - not
            asserted to be reliable across every ASReml-R build).
        'message'   : human-readable summary, always populated - what was
            found, or what went wrong trying to find out.
    """
    result = {'installed': False, 'licensed': None, 'version': None,
              'expiry': None, 'message': ''}
    try:
        installed = _run_r_snippet("requireNamespace('asreml', quietly = TRUE)", r_path)
        result['installed'] = bool(installed[0])
    except Exception as exc:
        result['message'] = (
            f"Could not even attempt to check for the 'asreml' R package - the R bridge itself "
            f"failed: {exc!r}. Check R_PATH / your R installation before troubleshooting ASReml "
            f"specifically."
        )
        return result

    if not result['installed']:
        result['message'] = (
            "The 'asreml' R package is not installed in this R environment. It is a commercial "
            "package (not on CRAN/conda-forge) - install it manually following VSNi's own "
            "instructions (https://www.vsni.co.uk/software/asreml-r/), then activate your "
            "licence (see activate() in this module, or the GUI's ASReml section)."
        )
        return result

    try:
        version_r = _run_r_snippet("as.character(utils::packageVersion('asreml'))", r_path)
        result['version'] = str(version_r[0])
    except Exception:
        pass  # Non-fatal - version is a diagnostic nicety, not required to proceed.

    try:
        # Appendix B check B3: ASReml-R 4's documented licence-status entry
        # point. Wrapped defensively - see this module's own version-
        # caveat docstring note - so a differently-shaped API on some
        # other licensed build degrades this ONE field, not the whole
        # preflight check.
        status_r = _run_r_snippet(
            "tryCatch({s <- asreml::asreml.license.status(quiet = TRUE); "
            "as.character(s)}, error = function(e) NA_character_)",
            r_path,
        )
        status_text = str(status_r[0]) if status_r is not None else None
        if status_text and status_text.lower() != 'na':
            result['licensed'] = 'expired' not in status_text.lower() and 'invalid' not in status_text.lower()
            result['message'] = f"asreml {result['version'] or '(unknown version)'} - licence status: {status_text}."
        else:
            result['message'] = (
                f"asreml {result['version'] or '(unknown version)'} is installed, but its licence "
                f"status could not be determined via asreml.license.status() - this may indicate a "
                f"different ASReml-R API generation than this module was written against (Appendix "
                f"B check B3). Proceed with caution; the first real fit will surface a licence "
                f"error directly if one exists."
            )
    except Exception as exc:
        result['message'] = (
            f"asreml {result['version'] or '(unknown version)'} is installed, but checking its "
            f"licence status raised {exc!r} - this may indicate a different ASReml-R API "
            f"generation than this module was written against (Appendix B check B3)."
        )

    return result


def activate(activation_code: str, r_path: "Optional[str]" = None) -> dict:
    """Attempt to activate `asreml` with `activation_code`, via the SAME
    ASReml-R 4 documented API (`asreml::asreml.license.activate()`).
    Never raises - returns `{'success': bool, 'message': str}`.

    Activation is per-MACHINE, not per-installation of EasiGP - if this is
    run from a Streamlit session on a submission/login host, it does NOT
    license any separate HPC compute node the actual pipeline will later
    run on in every ASReml-R deployment. The returned message says so
    explicitly whenever activation succeeds, so a person doesn't discover
    this the hard way mid-job.
    """
    if not activation_code or not activation_code.strip():
        return {'success': False, 'message': "No activation code given."}
    try:
        result_r = _run_r_snippet(
            f"tryCatch({{asreml::asreml.license.activate('{activation_code.strip()}'); 'OK'}}, "
            f"error = function(e) paste('ERROR:', conditionMessage(e)))",
            r_path,
        )
        result_text = str(result_r[0])
    except Exception as exc:
        return {'success': False, 'message': f"Activation attempt failed - the R bridge itself "
                                              f"raised: {exc!r}."}

    if result_text == 'OK':
        return {
            'success': True,
            'message': (
                "Activation succeeded on THIS machine. Remember: ASReml-R licence activation is "
                "per-machine, not per-EasiGP-installation - if your actual pipeline run will "
                "execute on a different host (e.g. an HPC compute node reached via a job "
                "scheduler, rather than this submission/login host), you must activate there "
                "too before submitting the job."
            ),
        }
    return {'success': False, 'message': f"Activation failed: {result_text}"}


def preflight_check(cfg: dict) -> None:
    """Layer 2 of R13's three gating layers - called from
    `genomic_prediction.py::GP()` before task 0, alongside the existing
    `model_registry.is_available()` Tier-2-optional-dependency fail-fast
    block. Raises `RuntimeError` with an actionable message if `asreml`
    is not installed or not licensed; a caller with
    `cfg.get('ASREML_STRICT', True) == False` is expected to catch this
    and downgrade it to a warning itself (mirroring
    `MODEL_AVAILABILITY_STRICT`'s own caller-side handling) - this
    function always raises on a genuine problem; it never silently
    downgrades its own severity.

    A no-op (returns immediately) if `cfg.get('ASREML')` is falsy - this
    check only ever matters for a run that has actually enabled the
    ASReml stage.
    """
    asreml_cfg = cfg.get('ASREML')
    if not asreml_cfg:
        return

    r_path = cfg.get('R_PATH')
    status = license_status(r_path)

    if not status['installed']:
        raise RuntimeError(
            f"[ASReml] ASREML is enabled in this run's configuration, but the 'asreml' R "
            f"package is not installed in this R environment. {status['message']} Set "
            f"ASREML_STRICT=False to downgrade this to a warning and let the run proceed until "
            f"Step 6b actually needs it (not recommended for an unattended HPC job - the same "
            f"failure will simply occur later, after other work has already been done)."
        )

    if status['licensed'] is False:
        raise RuntimeError(
            f"[ASReml] ASREML is enabled in this run's configuration, and the 'asreml' R "
            f"package is installed, but it does not appear to be licensed on this machine. "
            f"{status['message']} Activate your licence (via the GUI's ASReml section, or "
            f"Preprocess.asreml_license.activate() directly) on the machine that will actually "
            f"execute this run - remember that activation is per-MACHINE, so activating on a "
            f"submission/login host does not license a separate HPC compute node."
        )

    # status['licensed'] is True, or None (undeterminable - degraded, but
    # not treated as a hard failure here, since the alternative is
    # refusing to run ASReml at all on every build whose licence-status
    # API this module's own version caveat doesn't recognise, which would
    # be strictly worse than proceeding and letting the real fit surface
    # any genuine licence problem directly).
