# Preprocess/asreml_fit.R
# =========================
# Update ID ver4-12, R5/R8/R9/R12 (design blueprint SS2.5 "Module layout").
#
# The R-side half of the ASReml-R bridge, sourced into the R global
# environment by Preprocess/asreml_blup.py via rpy2, exactly the same way
# genomic_prediction.py already sources models/rrBLUP.R, models/GBLUP.R,
# models/BayesB.R and models/RKHS.R (see that module's own r_source() calls
# and their comment on why an ABSOLUTE path, resolved relative to this
# file's own location, is required rather than a bare relative path - a
# headless HPC array job's current working directory is not guaranteed to
# be the EasiGP install directory).
#
# This file exposes exactly two functions:
#   asreml_prepare_ginv(...)       - attach the row/col names and the
#                                     ASReml-R 4.x INVERSE attribute
#                                     convention to a plain numeric
#                                     genomic-relationship-matrix inverse
#                                     built in Python (Preprocess/
#                                     asreml_blup.py::build_or_load_grm(),
#                                     itself backed by genomic_prediction.
#                                     py::_build_grm() - invariant I12, no
#                                     new R dependency such as AGHmatrix).
#   asreml_fit_and_predict(...)    - eval the caller's own already-
#                                     substituted ASReml-R call text,
#                                     iterate update.asreml() to
#                                     convergence (bounded), predict at the
#                                     configured output level, and return
#                                     (pvals, varcomp, converged,
#                                     iterations) as a plain named list -
#                                     the exact shape
#                                     Preprocess/asreml_blup.py expects
#                                     back through r_list_get()
#                                     (pipeline_utils.py), the same helper
#                                     already used to unpack the four
#                                     BGLR models' own R return lists.
#
# ASReml-R API SURFACE - VERSION CAVEAT (Appendix B, checks B1-B4)
# --------------------------------------------------------------------------
# This file was written and reviewed against ASReml-R 4's PUBLICLY
# DOCUMENTED API (asreml(), update.asreml(), predict.asreml(), vm(),
# na.method()) - it could NOT be executed against a real, licensed
# installation in this environment (no licensed copy is available here -
# see Preprocess/asreml_license.py's own docstring). Three specific points
# are flagged, rather than silently assumed, exactly per this project's
# "name it rather than assume" coding standard:
#   B1 - whether predict.asreml(..., classify=<level>)$pvals returns rows
#        for individuals masked to NA under na.action=na.method(y="include")
#        is the ENTIRE mechanism R8 depends on. It is believed correct
#        (this is ASReml-R's own documented purpose for na.method(y=
#        "include")), but Phase 3 MUST confirm this against the reference
#        dataset with a deliberate mask before trusting Step 6b's output
#        on real data - see the ver4-12 Change Summary SS8.
#   B2 - fit$converge is used as the convergence flag below (ASReml-R's
#        own documented field for this). If a licensed build exposes a
#        differently-named field, only asreml_convergence_flag() below
#        needs to change - the bounded-loop structure around it is
#        unaffected.
#   B4 - the INVERSE-attribute convention for a user-supplied genomic
#        relationship matrix differs across ASReml-R versions
#        (genotype_random_effect_model_GRM.R's own documented caveat,
#        lines 171-178 of that reference script). asreml_prepare_ginv()
#        below implements the ASReml-R 4.x convention (attr(G, "INVERSE")
#        <- TRUE, consumed via vm(<term>, source = G)) and PROBES for an
#        older asreml-R 3.x-style grm()/ginverse= convention as a fallback,
#        logging which branch it took either way (never silently guessing).


# ---------------------------------------------------------------------------
# asreml_convergence_flag(fit)
#
# Isolated into its own one-line function specifically so Appendix B check
# B2 (a different field name in a particular licensed build) is a ONE-LINE
# fix here, not a hunt through the loop logic below.
# ---------------------------------------------------------------------------
asreml_convergence_flag <- function(fit) {
  isTRUE(fit$converge)
}


# ---------------------------------------------------------------------------
# asreml_densify_vcov(vcov_matrix)
#
# ver4-25, R8 bugfix (Requirements2.md, Error #2): `predict.asreml(...,
# vcov = TRUE)` does not always hand back a plain base-R `matrix`. For a
# fit whose prediction equations ASReml-R itself elects to solve using
# its sparse machinery (e.g. a model with `sparse = ~ <term>`, or any
# residual/dispersion structure ASReml-R internally represents sparsely),
# `pred$vcov` can come back as an S4-classed `Matrix` package object
# (e.g. a `dsCMatrix`/`dspMatrix`) instead. The process-wide rpy2
# `numpy2ri` converter (`pipeline_utils.init_rpy2_conversion()`, the
# Python side of this bridge) only knows how to convert a plain base-R
# matrix/array into a numpy array automatically - an S4 object crosses
# back into Python UNCONVERTED, as an `rpy2.robjects.methods.RS4`
# wrapper, and `Preprocess/asreml_blup.py`'s own
# `np.asarray(raw_vcov, dtype=float)` (immediately downstream of
# `r_list_get(r_result, 'vcov')`) then fails with `TypeError: float()
# argument must be a string or a real number, not 'RS4'` - reproduced
# exactly against a real "full variance-covariance matrix" run (see the
# ver4-25 Requirements document's Error #2, raised from
# `_asreml_get_unmasked_result -> fit_unmasked_adjustment ->
# _maybe_mmap_vcov`).
#
# Densifying HERE, before the value ever leaves R, means the Python side
# keeps receiving exactly the plain dense matrix its own docstring
# already documents ("pred$vcov is a dense matrix over exactly the SAME
# rows pvals reports") - no Python-side change needed, and every OTHER
# caller (return_vcov = FALSE, vcov_matrix always NULL) is completely
# untouched (I11: `is.null(vcov_matrix)` short-circuits before this
# function is ever called for them).
#
# NULL passes straight through - the caller's own `is.null(vcov_matrix)`
# `stop()` (immediately after this call, at both call sites) still fires
# on a genuine "vcov unavailable" NULL, never on a coerced value.
# ---------------------------------------------------------------------------
asreml_densify_vcov <- function(vcov_matrix) {
  if (is.null(vcov_matrix)) {
    return(vcov_matrix)
  }
  if (isS4(vcov_matrix) || methods::is(vcov_matrix, "Matrix")) {
    vcov_matrix <- as.matrix(vcov_matrix)
  }
  # Belt-and-braces: force plain double storage even for an
  # already-dense `matrix`, so a `TypeError` one layer later (character/
  # factor artefact from an unusual predict() configuration) can never
  # reproduce itself under a different guise - a numeric matrix's own
  # storage mode is always already "double", so this is a strict no-op
  # for the common case.
  storage.mode(vcov_matrix) <- "double"
  vcov_matrix
}


# ---------------------------------------------------------------------------
# asreml_prepare_ginv(ginv_matrix, ids)
#
# Attach row/column names and the inverse-relationship-matrix marker
# convention ASReml-R expects to a PLAIN numeric matrix built in Python
# (already symmetrised and bent towards identity - see
# Preprocess/asreml_blup.py::build_or_load_grm()'s own docstring for the
# numerics; this function does no numerical work at all, only R-side
# metadata attachment).
#
# `ids` must be in the SAME row/column order as `ginv_matrix` - the caller
# (asreml_blup.py) is responsible for that alignment; this function trusts
# it rather than re-deriving it, since re-deriving an ID order from inside
# R would require re-implementing exactly the alignment logic the Python
# side already owns.
# ---------------------------------------------------------------------------
asreml_prepare_ginv <- function(ginv_matrix, ids) {
  rownames(ginv_matrix) <- ids
  colnames(ginv_matrix) <- ids

  # ASReml-R 4.x convention (Appendix B check B4): a user-supplied
  # inverse-relationship matrix is marked via the "INVERSE" attribute and
  # consumed in a model formula via vm(<term>, source = <this object>).
  # asreml-R 3.x instead used a separate `ginverse=` list argument to
  # asreml() itself and a plain (non-attributed) matrix - probed for here
  # as a fallback ONLY so this file fails loudly with a clear message
  # rather than silently mis-attributing the matrix, if EasiGP is ever run
  # against that older API generation.
  asreml_version <- tryCatch(
    as.character(utils::packageVersion("asreml")),
    error = function(e) NA_character_
  )
  is_v4_or_later <- is.na(asreml_version) || !startsWith(asreml_version, "3.")

  if (is_v4_or_later) {
    attr(ginv_matrix, "INVERSE") <- TRUE
    cat(sprintf(
      "[ASReml] asreml_prepare_ginv: using the ASReml-R 4.x INVERSE-attribute "
      %+% "convention (attr(G, 'INVERSE') <- TRUE, consumed via vm()) - "
      %+% "detected package version: %s.\n",
      ifelse(is.na(asreml_version), "unknown (assuming 4.x)", asreml_version)
    ))
  } else {
    cat(sprintf(
      "[ASReml] asreml_prepare_ginv: detected asreml-R %s (pre-4.x) - the "
      %+% "INVERSE-attribute/vm() convention this file implements does NOT "
      %+% "apply to this version. EasiGP's ASReml integration was "
      %+% "written and reviewed against the ASReml-R 4.x API only; running "
      %+% "it against a 3.x installation requires adapting this function to "
      %+% "that version's own ginverse= argument convention first (Appendix "
      %+% "B check B4) - stopping rather "
      %+% "than silently producing a wrongly-attributed matrix.\n",
      asreml_version
    ))
    stop("[ASReml] Unsupported asreml-R major version for the GRM route - see the message above.")
  }

  ginv_matrix
}

# Small helper: R has no built-in string-concatenation infix operator: the
# `%+%` used just above is defined here, once, purely for the multi-line
# cat() calls' own readability (paste0() would work identically).
`%+%` <- function(a, b) paste0(a, b)


# ---------------------------------------------------------------------------
# asreml_fit_and_predict(dat, call_text, response_col, factor_cols,
#                         order_cols, output_level, max_update_rounds,
#                         workspace, pworkspace, ginv = NULL,
#                         ginv_ids = NULL, genotype_col = NULL)
#
# Parameters
# ----------
# dat              : the data.frame for this task's fit - already row-
#                    subsetted to this task's individuals (train + masked
#                    valid + masked test) by Preprocess/asreml_blup.py, and
#                    already carrying `response_col`'s masked (NA-for-
#                    valid/test) values. Converted from pandas via rpy2's
#                    pandas2ri; every design column may therefore arrive as
#                    an R character vector or a numeric vector depending on
#                    its pandas dtype - the coercion loop below (mirroring
#                    the reference scripts' own "CRITICAL FIX" block) is
#                    what turns the ones ASReml needs to treat as factors
#                    into TRUE R factors.
# call_text        : the already-{TRAIT}/{DATA}/{GRM}-substituted ASReml-R
#                    call, as a single string, e.g.
#                    "asreml(fixed = GrainYield ~ 1, sparse = ~ Genotype,
#                     random = ~ Row + Range, residual = ~ ar1(Row):
#                     ar1(Range), data = dat, na.action = na.method(y =
#                     'include', x = 'include'), trace = TRUE)" - `dat` in
#                    the substituted text always refers to THIS function's
#                    own `dat` argument (assigned into this function's
#                    local environment below before the eval), and (when a
#                    GRM is configured) `Ginv` always refers to this
#                    function's own `ginv` argument.
# response_col     : name of the (already-masked) response column in `dat`
#                    - coerced to numeric here (never left as whatever
#                    dtype rpy2 happened to convert it to).
# factor_cols      : character vector of column names in `dat` that must
#                    be coerced to R factors before the fit (every design
#                    column referenced anywhere in the model - fixed,
#                    sparse, random, residual, interaction/nested terms -
#                    the genotype column itself included).
# order_cols       : character vector of 0, 1 or 2 column names to sort
#                    `dat` by before fitting - required whenever the
#                    residual structure names an ordered dimension (e.g.
#                    ar1(Row):ar1(Range) needs `dat` sorted by (Row, Range)
#                    - the reference scripts' own
#                    `dat[order(dat$Row, dat$Range), ]` step, generalised
#                    to whatever dimension names this task's own residual
#                    structure configured). Empty vector = no re-ordering.
# output_level     : character vector of column name(s) predict(classify=)
#                    groups by (R2/R6's own "output level" - e.g. a single
#                    genotype column, or several columns for a
#                    population-defining group).
# max_update_rounds: integer upper bound on additional update.asreml()
#                    rounds (R9) - a hard stop so a mis-specified model
#                    cannot hang an HPC job indefinitely.
# workspace, pworkspace : passed straight to asreml.options() (character,
#                    e.g. "4gb") - mirrors the reference scripts' own
#                    setting.
# ginv             : optional RAW (dimname-less) inverse relationship
#                    matrix, as it crosses the Python->R boundary via
#                    rpy2's numpy2ri converter (ver4-15 bugfix - see this
#                    file's own Update ID note above: numpy2ri cannot
#                    carry dimnames, so this arrives as a plain matrix,
#                    NOT yet asreml_prepare_ginv()-attributed). Attributed
#                    HERE, inside this function, using `ginv_ids` (see
#                    below), before anything else touches it - never
#                    handed back to Python in between. NULL (the default)
#                    when ASREML['use_grm'] is False - `call_text` then
#                    simply never references `Ginv` at all.
# ginv_ids         : character vector of individual IDs, in exactly the
#                    row/column order `ginv` was built in (Preprocess/
#                    asreml_blup.py::build_or_load_grm()'s own contract -
#                    `id_train + id_valid + id_test`) - the ID vector
#                    `asreml_prepare_ginv()` needs to attach as `ginv`'s
#                    dimnames. NULL whenever `ginv` is NULL.
# genotype_col     : name of the genotype/individual-id column in `dat`
#                    (Requirements.md item 1 fix). When both this AND
#                    `ginv` are supplied, this column's factor level ORDER
#                    is forced to exactly `rownames(ginv)` (see the
#                    coercion loop below for why a default, alphabetically-
#                    ordered `factor(dat[[col]])` is not sufficient here).
#                    NULL is fine when `ginv` is NULL (no GRM configured -
#                    the genotype column is still factor()-ed normally,
#                    just with no forced level order).
#
# Returns
# -------
# A plain named list: `pvals` (predict.asreml()'s own data.frame, one row
# per output-level combination - see Appendix B check B1 for what this
# contains for a MASKED individual), `varcomp` (a data.frame built from
# summary(fit)$varcomp, with the component name as its own column rather
# than a row name - row names do not survive rpy2's pandas2ri round trip
# cleanly), `converged` (logical) and `iterations` (integer, the number of
# ADDITIONAL update.asreml() rounds beyond the initial fit - 0 means the
# very first fit already satisfied fit$converge).


library(asreml)

# ---------------------------------------------------------------------------
# `%||%` - a genuinely absent summary(fit)$varcomp column degrades to NA
# rather than raising (Appendix B - the exact column set that table
# returns is one more place flagged as version-dependent). Defined once,
# at file scope, so both asreml_fit_and_predict() and ver4-25's new
# asreml_fit_fixed_with_constrained_vparameters() below share it rather
# than each declaring their own local copy.
# ---------------------------------------------------------------------------
`%||%` <- function(col_name, table) {
  if (col_name %in% names(table)) table[[col_name]] else rep(NA, nrow(table))
}

# ---------------------------------------------------------------------------
# asreml_varcomp_frame(fit)
#
# Builds the plain data.frame(component, estimate, std_error, bound) both
# fitting functions below return - factored out ver4-25 (R7) purely so a
# future change to summary(fit)$varcomp's own column handling only needs
# to happen once. This is the literal body asreml_fit_and_predict() had
# before ver4-25 - no numeric change.
# ---------------------------------------------------------------------------
asreml_varcomp_frame <- function(fit) {
  varcomp_raw <- summary(fit)$varcomp
  data.frame(
    component = rownames(varcomp_raw),
    estimate = "component" %||% varcomp_raw,
    std_error = "std.error" %||% varcomp_raw,
    bound = as.character("bound" %||% varcomp_raw),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------------
# asreml_set_options(workspace, pworkspace, threads)
#
# ver4-25, R9-S5 (design blueprint stage 5, "ASReml threading") - the
# SINGLE place `asreml.options()` is ever called from this file, so both
# fitting functions below configure ASReml identically. `threads` (design
# blueprint's own `ASREML['threads']`, default `NULL` = untouched, exactly
# reproducing every pre-ver4-25 call) is passed to `asreml.options()` ONLY
# when non-NULL, and wrapped in its own `tryCatch` - not every installed
# ASReml-R version exposes a `threads` option (design blueprint V9.3: "if
# the installed ASReml-R exposes it" - unverified in this environment, no
# licensed install available to check against). A version that rejects it
# degrades to the ordinary options() call, with one WARNING explaining why,
# rather than failing the whole fit over a speed-only setting.
# ---------------------------------------------------------------------------
asreml_set_options <- function(workspace, pworkspace, threads = NULL) {
  if (is.null(threads)) {
    asreml.options(workspace = workspace, pworkspace = pworkspace, trace = FALSE)
    return(invisible(NULL))
  }
  tryCatch({
    asreml.options(workspace = workspace, pworkspace = pworkspace, trace = FALSE, threads = threads)
  }, error = function(e) {
    cat(sprintf("[ASReml] WARNING: asreml.options(threads=%s) failed (%s) - this ASReml-R "
                %+% "install may not support the 'threads' option (design blueprint V9.3, "
                %+% "unverified without a licensed install). Falling back to the default "
                %+% "options() call without it; ASREML['threads'] has no further effect this "
                %+% "run.\n", threads, conditionMessage(e)))
    asreml.options(workspace = workspace, pworkspace = pworkspace, trace = FALSE)
  })
  invisible(NULL)
}

# ---------------------------------------------------------------------------
# asreml_prepare_dat(dat, response_col, factor_cols, order_cols, ginv,
#                     ginv_ids, genotype_col)
#
# ver4-25, R7 (design blueprint stage 4) - EXTRACTED from what used to be
# the top half of asreml_fit_and_predict()'s own body, so ver4-25's new
# asreml_fit_fixed_with_constrained_vparameters() (below) can reuse the
# IDENTICAL data-preparation steps (GRM attribution, factor coercion,
# response numeric coercion, ordered-residual sorting) rather than a
# second, independently-maintained copy that could silently drift from
# this one. Pure data preparation only - no fitting happens here.
#
# Returns a list(dat = <prepared data.frame>, ginv = <attributed matrix or
# NULL>) - `ginv` is returned alongside `dat` because it is mutated by
# asreml_prepare_ginv() (dimnames/the INVERSE attribute are attached to
# it) and the caller needs the ATTRIBUTED version, not its own original
# argument, for the eventual vm(<term>, source = Ginv) call.
# ---------------------------------------------------------------------------
asreml_prepare_dat <- function(dat, response_col, factor_cols, order_cols,
                                ginv = NULL, ginv_ids = NULL, genotype_col = NULL) {
  # ver4-15 BUGFIX ("'source' is missing a 'dimnames' attribute"): attach
  # the row/col names and the ASReml-R 4.x INVERSE attribute to the raw
  # matrix HERE, R-side, using the ID vector crossed over alongside it -
  # never round-tripped back through rpy2 (which would strip them again).
  # See this file's own top-of-file Update ID note and asreml_prepare_
  # ginv()'s own docstring for the full root-cause explanation.
  if (!is.null(ginv)) {
    ginv <- asreml_prepare_ginv(ginv, ginv_ids)
  }

  # "CRITICAL FIX" (mirrors genotype_fixed_effect_model.R /
  # genotype_random_effect_model.R's own comment of the same name):
  # rpy2's pandas2ri conversion of an object-dtype pandas column reaches R
  # as a plain character vector, not a factor - ASReml silently treats an
  # un-factored character vector very differently from a factor (often as
  # a single, un-interpretable covariate, or raises an opaque error deep
  # inside its own C code), so every design column this task's model
  # actually references is explicitly factor()-ed here, R-side, regardless
  # of what dtype it arrived as.
  #
  # Requirements.md item 1 FIX: for the genotype column specifically, a
  # plain `factor(dat[[col]])` is NOT enough whenever a genomic
  # relationship matrix is also configured. `factor()` on its own assigns
  # levels in ALPHABETICAL/lexical order of whatever genotype ids happen
  # to appear in `dat` - which, in general, is NOT the same order as
  # `rownames(ginv)` (the order `Preprocess/asreml_blup.py::
  # build_or_load_grm()` built the matrix in). `vm(<genotype col>,
  # source = Ginv)` associates the genotype factor's levels with Ginv's
  # rows/columns positionally, so a mismatched level order silently wires
  # each genotype up to the WRONG row of the relationship matrix (or, for
  # a genotype whose default alphabetical level has no counterpart at the
  # same position, effectively no informative relationship at all) - the
  # observed symptom is every validation/test individual's own BLUP/GBLUP
  # collapsing to the same (population-mean) constant, because none of
  # them ends up correctly linked to its own row of Ginv. Forcing
  # `levels = rownames(ginv)` guarantees the genotype factor's own level
  # order is IDENTICAL to the relationship matrix's row/column order for
  # every training, validation AND test individual (rownames(ginv) is
  # built from exactly that union - see build_or_load_grm()'s own
  # docstring), which is what this fix is about.
  for (col in factor_cols) {
    if (col %in% names(dat)) {
      if (!is.null(ginv) && !is.null(genotype_col) && identical(col, genotype_col)) {
        # ver4-18: a row that exists ONLY to complete a grid-structured
        # residual's own Cartesian product (asreml_syntax.pad_residual_
        # grid()) already carries NA in this column BEFORE factor()
        # ever runs - that is expected and intentional (module docstring,
        # "ver4-18 BUGFIX"), not a genotype id failing to match the GRM.
        # Counted separately here so the warning below stays meaningful
        # for its ORIGINAL purpose (a real id/GRM mismatch) instead of
        # firing on every grid-completion run regardless of whether one
        # exists.
        was_na_before <- is.na(dat[[col]])
        dat[[col]] <- factor(dat[[col]], levels = rownames(ginv))
        n_unmatched <- sum(is.na(dat[[col]]) & !was_na_before)
        if (n_unmatched > 0) {
          cat(sprintf(
            "[ASReml] WARNING: %d row(s) have a genotype id in %s that is NOT among the %d "
            %+% "id(s) the genomic relationship matrix was built for - their genotype level is "
            %+% "now NA and they cannot receive a GRM-informed value. This usually means the "
            %+% "GRM was built from a different individual set than this fit's own data.\n",
            n_unmatched, col, length(rownames(ginv))
          ))
        }
      } else {
        dat[[col]] <- factor(dat[[col]])
      }
    }
  }
  dat[[response_col]] <- as.numeric(dat[[response_col]])

  # AR1 (and other ordered-dimension) residual structures require the data
  # sorted along that dimension before fitting - the reference scripts' own
  # `dat[order(dat$Row, dat$Range), ]` step, generalised to whichever 1-2
  # columns this task's own residual structure configured.
  if (length(order_cols) > 0) {
    ordering <- do.call(order, as.list(dat[order_cols]))
    dat <- dat[ordering, ]
    cat(sprintf("[ASReml] Sorted %d row(s) by (%s) before fitting - required by this model's "
                %+% "residual structure.\n", nrow(dat), paste(order_cols, collapse = ", ")))
  }

  list(dat = dat, ginv = ginv)
}


# ---------------------------------------------------------------------------
# asreml_eval_fit_call(call_text, envir)          [ver4-55 BUGFIX]
#
# Production failure (cotton, per_set_grm, trait STR, task 1/25):
#
#     Error in asreml(fixed = ... ~ 1, random = ~vm(ID, source = Ginv) + ...) :
#       Error : 1 singularities in the Average Information matrix.
#             : To help identifying the issue, try running it again with
#             : argument ai.sing = TRUE; ...
#
# ASReml-R's default is to ABORT the moment its Average Information matrix
# is singular (a variance parameter that the data cannot identify - typical
# for the small, physically-restricted validation / test split of a
# per-set-GRM fit, where one record per individual leaves sigma_g^2 and
# sigma_e^2 only weakly separable, or where a random design factor has too
# few levels inside that split). Nothing in EasiGP ever retried, so one
# unidentifiable parameter in one split stopped the whole HPC array task.
#
# Behaviour (controlled by options("easigp.asreml.ai_sing"), which the
# Python side sets from ASREML['ai_sing']):
#   "auto" (default) - fit exactly as before. ONLY if that fit raises an
#                      AI-singularity error, retry once with ai.sing = TRUE
#                      (ASReml's own recommended remedy) and print a loud
#                      WARNING plus the variance components whose `bound`
#                      flag is not 'P' (i.e. the ones that were fixed /
#                      singular), so the user can see which term the data
#                      could not support.
#   "off"            - never retry; the original ASReml error is raised
#                      (the pre-ver4-55 behaviour).
#   "on"             - fit with ai.sing = TRUE from the first attempt.
# A fit that never hits the singularity is byte-identical to before (I11).
#
# Two mechanisms are tried for ai.sing because the exact place ASReml-R 4.x
# accepts it (asreml.options() vs. an argument of asreml()) was never
# verified against a licensed build here: first asreml.options(ai.sing =
# TRUE), then, if that is rejected or still fails, the argument is spliced
# into the asreml() call text. Returns list(fit = <asreml object>,
# ai_sing_used = <logical>).
# ---------------------------------------------------------------------------
asreml_eval_fit_call <- function(call_text, envir) {
  mode <- getOption("easigp.asreml.ai_sing", "auto")
  if (!mode %in% c("auto", "on", "off")) mode <- "auto"

  run_plain <- function() eval(parse(text = call_text), envir = envir)

  run_with_ai_sing <- function() {
    # Mechanism 1: global ASReml option, restored afterwards by the caller.
    opt_ok <- tryCatch({
      asreml.options(ai.sing = TRUE)
      TRUE
    }, error = function(e) FALSE)
    if (opt_ok) {
      res <- tryCatch(run_plain(), error = function(e) e)
      if (!inherits(res, "error")) return(res)
      cat(sprintf("[ASReml] ai.sing retry via asreml.options() still failed: %s\n",
                  conditionMessage(res)))
    }
    # Mechanism 2: splice the argument into the asreml() call itself.
    if (grepl("ai\\.sing", call_text)) stop("ai.sing already present in call text")
    spliced <- sub("\\basreml\\s*\\(", "asreml(ai.sing = TRUE, ", call_text, perl = TRUE)
    eval(parse(text = spliced), envir = envir)
  }

  if (identical(mode, "on")) {
    fit <- run_with_ai_sing()
    return(list(fit = fit, ai_sing_used = TRUE))
  }

  first <- tryCatch(run_plain(), error = function(e) e)
  if (!inherits(first, "error")) {
    return(list(fit = first, ai_sing_used = FALSE))
  }
  msg <- conditionMessage(first)
  if (identical(mode, "off") || !grepl("singularit", msg, ignore.case = TRUE)) {
    stop(first)
  }

  cat(sprintf(paste0(
    "[ASReml] WARNING: %s\n[ASReml] The data cannot identify every variance parameter of this ",
    "model (AI-matrix singularity). Retrying ONCE with ai.sing = TRUE as ASReml suggests. The ",
    "affected component(s) will be flagged in the variance-component table; interpret this fit ",
    "with care. Set ASREML['ai_sing'] = 'off' to make this a hard error again, or simplify the ",
    "random terms / use a larger split.\n"), gsub("\\s+", " ", msg)))
  fit <- tryCatch(run_with_ai_sing(), error = function(e) {
    stop(paste0(msg, "\n[ASReml] Retry with ai.sing = TRUE also failed: ", conditionMessage(e)),
         call. = FALSE)
  })
  list(fit = fit, ai_sing_used = TRUE)
}

# ---------------------------------------------------------------------------
# asreml_release_and_gc(env)                                     [ver4-59, opt-in]
#
# DIAGNOSTIC / possible mitigation for intermittent native crashes ("corrupted
# double-linked list", "Segmentation fault") that surface in the FIRST large R
# allocation burst after a task's ASReml fits (e.g. rpy2's data.frame() call
# for the marker table), not inside ASReml itself. If the memory damage comes
# from the native finalisers of dropped asreml fit objects, forcing them to run
# HERE - right at the end of the fit that created them - moves the crash to a
# place the Python stack names ("asreml_fit_and_predict"), or avoids it.
# Active only when the environment variable EASIGP_ASREML_GC is set to a
# non-empty value other than "0"; otherwise a no-op (behaviour unchanged).
# ---------------------------------------------------------------------------
asreml_release_and_gc <- function(env) {
  flag <- Sys.getenv("EASIGP_ASREML_GC", "")
  if (!nzchar(flag) || identical(flag, "0")) return(invisible(NULL))
  drop <- intersect(c("fit", "fit_info", "sv_fit", "first", "pred", "Ginv", "dat",
                      "EasiGP_R7_start_values"), ls(envir = env, all.names = TRUE))
  if (length(drop) > 0) rm(list = drop, envir = env)
  invisible(gc(verbose = FALSE))
  cat("[ASReml] EASIGP_ASREML_GC: fit objects released and R garbage collection run.\n")
  invisible(NULL)
}

# ---------------------------------------------------------------------------
asreml_fit_and_predict <- function(dat, call_text, response_col, factor_cols,
                                    order_cols, output_level, max_update_rounds,
                                    workspace = "4gb", pworkspace = "4gb",
                                    ginv = NULL, ginv_ids = NULL, genotype_col = NULL,
                                    threads = NULL, return_vcov = FALSE) {

  asreml_set_options(workspace, pworkspace, threads)
  .easigp_fn_env <- environment()
  on.exit(asreml_release_and_gc(.easigp_fn_env), add = TRUE)

  prepared <- asreml_prepare_dat(dat, response_col, factor_cols, order_cols,
                                  ginv, ginv_ids, genotype_col)
  dat <- prepared$dat
  ginv <- prepared$ginv

  # Bind the substituted call's own free variables into THIS function's
  # local environment, then eval() the caller-supplied text directly - the
  # exact mechanism that makes a freely hand-edited ASReml-R call (R7's
  # "generate-then-edit-then-use" contract) reusable without EasiGP ever
  # having to parse the person's own R.
  environment_for_eval <- environment()
  assign("dat", dat, envir = environment_for_eval)
  if (!is.null(ginv)) {
    assign("Ginv", ginv, envir = environment_for_eval)
  }

  # ver4-55: remember asreml.options()'s ai.sing so a retry cannot leak into
  # the next fit made by this long-lived R session.
  prior_ai_sing <- tryCatch(asreml.options()$ai.sing, error = function(e) NULL)
  if (is.null(prior_ai_sing) || !is.logical(prior_ai_sing)) prior_ai_sing <- FALSE
  if (!identical(getOption("easigp.asreml.ai_sing", "auto"), "off")) {
    on.exit(try(asreml.options(ai.sing = prior_ai_sing), silent = TRUE), add = TRUE)
  }

  fit_info <- asreml_eval_fit_call(call_text, environment_for_eval)
  fit <- fit_info$fit

  converged <- asreml_convergence_flag(fit)
  iterations <- 0L
  while (!converged && iterations < max_update_rounds) {
    fit <- update.asreml(fit)
    converged <- asreml_convergence_flag(fit)
    iterations <- iterations + 1L
  }
  if (!converged) {
    cat(sprintf("[ASReml] WARNING: did not reach convergence after %d update.asreml() round(s) "
                %+% "(max_update_rounds=%d).\n", iterations, max_update_rounds))
  }
  if (isTRUE(fit_info$ai_sing_used)) {
    vc <- tryCatch(summary(fit)$varcomp, error = function(e) NULL)
    if (!is.null(vc) && "bound" %in% names(vc)) {
      flagged <- vc[!(as.character(vc$bound) %in% c("P", "")), , drop = FALSE]
      if (nrow(flagged) > 0) {
        cat("[ASReml] ai.sing fit - variance components NOT at a regular estimate:\n")
        print(flagged)
      }
    }
  }

  classify_term <- paste(output_level, collapse = ":")
  # ver4-25, R8 (design blueprint SS2, Stage 6, V8.2): `vcov = TRUE` asks
  # the SAME predict() call for the full variance-covariance matrix of the
  # predicted values, in ADDITION to pvals - no second predict() call is
  # made (avoids fitting/predicting twice for the 'full' BLUE-reliability-
  # weighting path). `return_vcov = FALSE` (the default - every pre-
  # ver4-25 caller) is a byte-identical `predict(fit, classify=classify_
  # term, pworkspace=pworkspace)` call - I11.
  pred <- predict(fit, classify = classify_term, pworkspace = pworkspace, vcov = return_vcov)
  pvals <- pred$pvals
  varcomp_df <- asreml_varcomp_frame(fit)

  # ver4-25, R8: `pred$vcov` is a dense matrix over exactly the SAME rows
  # `pvals` reports (ASReml-R's own documented `predict(..., vcov=TRUE)`
  # contract - the returned matrix is aligned to predict()'s own row
  # order), so no separate id-alignment vector is returned here - the
  # Python side (Preprocess/asreml_blup.py::_fit_single_asreml_model())
  # already extracts the genotype/ID column from `pvals` in that SAME row
  # order (see its own `id_series` - built before this function is ever
  # touched) and uses it as `vcov_ids`. V8.2 (Phase 1 pre-flight, still
  # unverified against a licensed install per this delivery's own
  # disclosure - see the ver4-25 Change Summary): if `return_vcov` was
  # requested but ASReml-R's own `vcov=TRUE` support is unavailable for
  # this fit (e.g. a residual/dispersion structure it does not support
  # for prediction covariances), `pred$vcov` comes back NULL - stop()
  # here with an actionable message rather than silently falling back to
  # an identity/diagonal covariance, per the blueprint's own "never
  # silently diagonal" requirement.
  vcov_matrix <- NULL
  if (isTRUE(return_vcov)) {
    # ver4-25, R8 bugfix (Requirements2.md, Error #2): densify BEFORE the
    # NULL check and before this value ever crosses back to Python - see
    # asreml_densify_vcov()'s own docstring for why `pred$vcov` is not
    # always a plain base-R matrix.
    vcov_matrix <- asreml_densify_vcov(pred$vcov)
    if (is.null(vcov_matrix)) {
      stop("[ASReml] asreml_fit_and_predict: ASREML['blue_weighting']='full' requested " %+%
           "predict(..., vcov=TRUE), but this fit's own predict() call returned no vcov " %+%
           "matrix - this residual/dispersion structure may not support prediction " %+%
           "covariances in this ASReml-R version. Use ASREML['blue_weighting']='diagonal' " %+%
           "instead, or consult ASReml-R's own predict.asreml() documentation for this " %+%
           "model's residual structure.")
    }
  }

  # ver4-25, R7: the full vparameters vector (named numeric - component
  # label -> current estimate), used by asreml_fit_fixed_with_constrained_
  # vparameters() below as step (i)'s own converged values to fix step
  # (ii)'s non-genetic terms at. Harmless to compute and return for every
  # OTHER caller too (a small named numeric vector) - r_list_get() on the
  # Python side simply ignores a key nobody asked for.
  # ver4-58: `ai_sing_used` tells the Python side whether this fit only
  # succeeded through the ai.sing = TRUE retry (a variance parameter was not
  # identifiable and was left at a boundary/fixed value).
  list(pvals = pvals, varcomp = varcomp_df, converged = converged, iterations = iterations,
       vparameters = fit$vparameters, vcov = vcov_matrix,
       ai_sing_used = isTRUE(fit_info$ai_sing_used))
}

# ---------------------------------------------------------------------------
# asreml_fit_fixed_with_constrained_vparameters(dat, call_text,
#     response_col, factor_cols, order_cols, output_level,
#     fixed_vparameters, workspace, pworkspace, ginv, ginv_ids,
#     genotype_col, max_update_rounds)
#
# ver4-25, R7 (design blueprint stage 4, Holland & Piepho 2024 Fig. 2,
# step (ii)) - fits `call_text` (the caller's own genotype-FIXED committed
# syntax, unchanged) with every NON-genetic variance parameter FIXED
# ("Constraint = 'F'") at `fixed_vparameters`'s own converged values from
# a PRIOR step-(i) fit (genotype random, GRM-informed - see Preprocess/
# asreml_blup.py::fit_blue_refinement_step1()). This is what "refines"
# the BLUE: the non-genetic variance structure is estimated once, well,
# from a model that can see genetic covariance information (the GRM), and
# then held fixed while the genotype term itself is finally estimated as
# a clean set of fixed effects (BLUEs) - rather than every non-genetic
# variance parameter being re-estimated, less precisely, in a model whose
# genotype term (fixed) carries no covariance information at all.
#
# `fixed_vparameters` : a NAMED numeric vector (component label -> value)
#     - exact NAME match against THIS call's own `start.values = TRUE`
#     vparameters.table, never by position (row order is not guaranteed
#     identical between two different models' own tables). A component
#     name present in `fixed_vparameters` but ABSENT from this fit's own
#     table (should not normally happen - the non-genetic terms are
#     identical between step (i) and step (ii) by construction) is simply
#     never matched, and a component present in the table but ABSENT from
#     `fixed_vparameters` (the genotype term itself, which has no
#     variance parameter of its own once FIXED) is left exactly as
#     asreml() itself proposed it - untouched, unconstrained.
#
# Returns the SAME shape asreml_fit_and_predict() does, plus
# `n_constrained` (integer - how many of `fixed_vparameters`' own entries
# were actually matched and fixed - genomic_prediction.py logs this so a
# silent mismatch between step (i)'s and step (ii)'s own non-genetic
# terms is visible rather than merely producing an unconstrained refit
# with no explanation).
#
# Raises (via stop()) if `start.values = TRUE` does not return a
# $vparameters.table (an unexpected ASReml-R API shape - Appendix B), or
# if NONE of `fixed_vparameters`' own names match this model's own table
# at all (refusing to silently fall back to a fully unconstrained refit,
# which would defeat the entire point of Fig. 2's method without any
# visible symptom other than a slightly different BLUE).
# ---------------------------------------------------------------------------
asreml_fit_fixed_with_constrained_vparameters <- function(
    dat, call_text, response_col, factor_cols, order_cols, output_level,
    fixed_vparameters, workspace = "4gb", pworkspace = "4gb",
    ginv = NULL, ginv_ids = NULL, genotype_col = NULL, max_update_rounds = 20,
    fix_values = TRUE, threads = NULL, return_vcov = FALSE) {

  asreml_set_options(workspace, pworkspace, threads)
  .easigp_fn_env <- environment()
  on.exit(asreml_release_and_gc(.easigp_fn_env), add = TRUE)

  prepared <- asreml_prepare_dat(dat, response_col, factor_cols, order_cols,
                                  ginv, ginv_ids, genotype_col)
  dat <- prepared$dat
  ginv <- prepared$ginv

  environment_for_eval <- environment()
  assign("dat", dat, envir = environment_for_eval)
  if (!is.null(ginv)) {
    assign("Ginv", ginv, envir = environment_for_eval)
  }

  # Step (ii), part A: fit the SAME call text with start.values = TRUE,
  # purely to obtain a vparameters.table whose row names/order match
  # THIS EXACT model - start.values = TRUE returns a table, not a
  # converged fit (ASReml-R's own documented behaviour), so this is cheap
  # and never itself iterates to convergence.
  sv_call_text <- sub("\\)\\s*$", ", start.values = TRUE)", call_text)
  sv_fit <- eval(parse(text = sv_call_text), envir = environment_for_eval)
  sv_table <- sv_fit$vparameters.table
  if (is.null(sv_table)) {
    stop("[ASReml] asreml_fit_fixed_with_constrained_vparameters: start.values = TRUE did not "
         %+% "return a $vparameters.table - unexpected ASReml-R API shape (Appendix B - report "
         %+% "this rather than guessing).")
  }

  # ver4-25, R9-S4 ("Warm start per-task fits"): `fix_values = FALSE` (the
  # design blueprint's own warm-start mode, `ASREML['warm_start_per_task']
  # =True`) sets ONLY `Value` from `fixed_vparameters` (here better read as
  # "seed values") and leaves `Constraint` exactly as `start.values = TRUE`
  # itself proposed - i.e. these seed the optimiser's own starting point
  # but remain free to move, unlike R7's `fix_values = TRUE` (the default,
  # and the ONLY mode R7 itself ever uses), which additionally sets
  # `Constraint <- "F"` (held fixed, never re-estimated). Both modes share
  # this one function because the "look up by exact Component name, then
  # edit the matched row(s)" mechanism is identical; only what gets
  # written into `Constraint` differs.
  matched <- 0L
  for (component_name in names(fixed_vparameters)) {
    row_match <- which(sv_table$Component == component_name)
    if (length(row_match) == 1) {
      sv_table$Value[row_match] <- fixed_vparameters[[component_name]]
      if (fix_values) {
        sv_table$Constraint[row_match] <- "F"
      }
      matched <- matched + 1L
    }
  }
  if (matched == 0L) {
    stop(sprintf(
      "[ASReml] asreml_fit_fixed_with_constrained_vparameters: none of the %d step-(i) variance "
      %+% "component name(s) matched this step-(ii) model's own vparameters.table component "
      %+% "names (%s) - refusing an UNCONSTRAINED refit that would silently re-estimate every "
      %+% "non-genetic variance parameter instead of fixing them at step (i)'s own values "
      %+% "(Holland & Piepho 2024, Fig. 2). This usually means the non-genetic model terms "
      %+% "differ between step (i) and step (ii) - they must be identical.",
      length(fixed_vparameters), paste(sv_table$Component, collapse = "; ")
    ))
  }

  # Step (ii), part B: the REAL fit, with every matched component now
  # fixed via R.param/G.param (ASReml-R's own documented mechanism for
  # supplying a start.values table back in with some rows constrained).
  assign("EasiGP_R7_start_values", sv_table, envir = environment_for_eval)
  final_call_text <- sub(
    "\\)\\s*$",
    ", R.param = EasiGP_R7_start_values, G.param = EasiGP_R7_start_values)",
    call_text
  )
  fit <- eval(parse(text = final_call_text), envir = environment_for_eval)

  # ver4-58: when EVERY variance parameter is held fixed (the per-set-GRM
  # "borrowed variance components" refit of a split too small to identify
  # its own variance parameters), there is nothing left for update.asreml()
  # to iterate on - the fit only solves the mixed-model equations for the
  # random effects - so the update loop is skipped and the fit is reported
  # as converged. A partially-fixed fit behaves exactly as before.
  all_fixed <- fix_values && all(as.character(sv_table$Constraint) == "F")
  if (all_fixed) {
    converged <- TRUE
    iterations <- 0L
    cat(sprintf("[ASReml] All %d variance parameter(s) held fixed - no variance iteration needed.\n",
                nrow(sv_table)))
  } else {
    converged <- asreml_convergence_flag(fit)
    iterations <- 0L
  }
  while (!converged && iterations < max_update_rounds) {
    fit <- update.asreml(fit)
    converged <- asreml_convergence_flag(fit)
    iterations <- iterations + 1L
  }
  if (!converged) {
    cat(sprintf("[ASReml] WARNING: step (ii) (fixed genotype, %d non-genetic variance "
                %+% "parameter(s) constrained) did not reach convergence after %d "
                %+% "update.asreml() round(s).\n", matched, iterations))
  }

  classify_term <- paste(output_level, collapse = ":")
  # ver4-25, R8 (design blueprint SS2, Stage 6) - see
  # asreml_fit_and_predict()'s own identical comment for the full
  # rationale; `return_vcov = FALSE` (default) is byte-identical (I11).
  pred <- predict(fit, classify = classify_term, pworkspace = pworkspace, vcov = return_vcov)
  pvals <- pred$pvals
  varcomp_df <- asreml_varcomp_frame(fit)

  vcov_matrix <- NULL
  if (isTRUE(return_vcov)) {
    # ver4-25, R8 bugfix (Requirements2.md, Error #2) - see
    # asreml_fit_and_predict()'s own identical call/comment above.
    vcov_matrix <- asreml_densify_vcov(pred$vcov)
    if (is.null(vcov_matrix)) {
      stop("[ASReml] asreml_fit_fixed_with_constrained_vparameters: ASREML['blue_weighting']=" %+%
           "'full' requested predict(..., vcov=TRUE), but this fit's own predict() call " %+%
           "returned no vcov matrix - see asreml_fit_and_predict()'s own identical message " %+%
           "for the likely cause.")
    }
  }

  list(pvals = pvals, varcomp = varcomp_df, converged = converged, iterations = iterations,
       vparameters = fit$vparameters, n_constrained = matched, vcov = vcov_matrix,
       all_fixed = all_fixed)
}
