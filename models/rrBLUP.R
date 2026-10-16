library(BGLR)
library(stringr)
library(data.table)
library(dplyr)

# Bug fix (Requirement.md follow-up to CHANGE_NOTES.md - "other places" in
# the model files themselves, not the generic tuning machinery in
# hparam_specs.py/models/hyperparameter_tuning.py). ROOT CAUSE: BGLR's own
# Gibbs sampler draws from R's GLOBAL random number generator, and this
# file never fixed it (genomic_prediction.py::_run_gblup_or_rkhs_for_trial's
# own docstring already says as much for GBLUP/RKHS: "BGLR does not fix a
# global RNG seed, so every worker's own main fit is an independent MCMC
# draw" - noted there as an accepted fact, never actually addressed). Every
# call to rrBLUP() - whether a hyperparameter-search TRIAL, the repeated
# inner resamples HP_TUNE_VALID_REPEATS draws to estimate a candidate's own
# score uncertainty, or the final CONFIRMATORY fit - is therefore an
# INDEPENDENT random MCMC realisation even when the data and hyperparameters
# are identical. Concretely, this means:
#   (a) two evaluations of the exact same candidate (e.g. the search's own
#       internal default-vs-winner comparison, or a Bayesian optimiser
#       revisiting a nearby point) can score differently purely from MCMC
#       noise, not from any real difference in generalisation - noise the
#       floor_at_baseline/1-SE-rule fixes in models/hyperparameter_tuning.py
#       were never designed to catch, since those operate on the SEARCH
#       SPACE and the SELECTION RULE, not on the model's own reproducibility;
#   (b) the final confirmatory fit's reported test-set performance is a
#       FRESH, independent MCMC draw, uncorrelated with whatever draw earned
#       the winning hyperparameters their validation-time "win" - so a
#       genuinely-at-least-as-good hyperparameter combination can still be
#       reported as scoring worse than a separately-run default purely
#       because the two runs' MCMC chains happened to land differently, with
#       nothing in the tuning safeguards able to detect or prevent it.
# This is a genomic-prediction-specific instance of "overfitting to the
# training/validation set" (Requirement.md's own suggested cause): the
# noise being overfit to is the MCMC sampler's own stochasticity, not just
# finite-sample validation noise.
#
# FIX: seed R's RNG, immediately before the main BGLR() fit, from a hash of
# the actual DATA being fit (the observed phenotypes and the marker
# matrix's own dimensions) - never from the hyperparameters themselves.
# This gives two properties simultaneously:
#   - Reproducibility: the SAME (data, hyperparameters) pair always yields
#     the SAME MCMC chain, so re-evaluating a candidate (or re-running the
#     confirmatory fit at hyperparameters the search already scored) can
#     never diverge from what was already seen purely by chance - exactly
#     the guarantee RF.py already gets for free from its own hardcoded
#     `random_state=0`, which rrBLUP/BayesB/RKHS never had an equivalent of.
#   - "Common random numbers": every hyperparameter CANDIDATE evaluated
#     against the SAME data (the same task, or the same HP_TUNE_VALID_REPEATS
#     inner-resampled fold) shares the SAME starting RNG state, so a
#     difference in validation score between two candidates reflects the
#     hyperparameters actually being searched (df0/R2/probIn/counts/h/nIter/
#     burnIn), not which one happened to draw a luckier MCMC chain - a
#     standard simulation-optimisation variance-reduction technique, applied
#     here to harden the search against exactly the kind of noise-driven
#     "winner" the margin/SE-rule fixes were introduced to catch. A
#     different task, replicate, or inner-resampled fold naturally changes
#     the observed phenotypes and/or row count, so it still gets its own,
#     independent (but equally reproducible) seed - nothing about this
#     collapses genuinely different data onto the same MCMC path.
.bglr_seed <- function(y, X) {
  y_obs <- y[is.finite(y)]
  basis <- length(y) * 2654435761 + length(y_obs) * 40503 +
    sum(y_obs) * 97 + sum(y_obs^2) + ncol(X) * 15487469 + nrow(X)
  as.integer(abs(basis) %% 2147483397) + 1L
}


rrBLUP <- function(train, valid, test, params, RESULT_NAME, weights = NULL){
  
  params <- unlist(params)
  nIter <- as.numeric(params[1])
  burnIn <- as.numeric(params[2])
  # Control how strongly BRR shrinks marker effects towards zero. df0 is the
  # prior degrees of freedom and R2 is the expected proportion of phenotypic
  # variance explained by the markers; together they set the prior variance
  # BGLR uses for the marker effects.
  df0 <- as.numeric(params[3])
  R2 <- as.numeric(params[4])

  # A process-unique id for BGLR's saveAt path - prevents concurrent tasks in
  # a parallel/HPC run (e.g. run_step1_batch.py array jobs) from colliding on
  # the same intermediate-file prefix, which otherwise only depended on
  # RESULT_NAME and was shared across every concurrent process.
  run_id <- paste0(Sys.getpid(), '_', sample.int(.Machine$integer.max, 1))

  # BGLR's saveAt writes several of its own diagnostic/trace files (MCMC
  # variance-component traces, etc.) as a side effect of every call - these
  # aren't read back by anything else in the pipeline, but shouldn't be left
  # loose directly in Result/<RESULT_NAME>/ alongside the pipeline's own
  # output files. showWarnings=FALSE because this is called once per task
  # (potentially many times, including concurrently across parallel/HPC
  # processes) - the directory already existing on a later call is expected,
  # not a problem to warn about.
  bglr_output_dir <- paste0('./Result/', RESULT_NAME, '/BGLR_output')
  dir.create(bglr_output_dir, showWarnings = FALSE, recursive = TRUE)

  data <- rbind(train, valid, test)
  data_qtl <- data.frame(lapply(data[,1:(ncol(data)-1)], as.numeric))
  data_pheno <- data[,ncol(data):ncol(data)]
  
  X <- scale(data_qtl)/sqrt(ncol(data_qtl))
  #X <- X[ , colSums(is.na(X)) == 0]
  X[is.na(X)] <- -1
  y <- as.numeric(unlist(data_pheno))
  
  y_test <- y
  y_test[(nrow(data)-(nrow(valid)+nrow(test))+1):nrow(data)] <- NA
  
  # Bug fix (see module-level comment above): deterministic, data-derived
  # seed immediately before the fit - never before, and never based on
  # anything else, so nothing upstream of this line (train_test_split,
  # marker-pool routing, etc.) is affected.
  set.seed(.bglr_seed(y_test, X))
  # ver4-25, R8 (design blueprint SS2, Stage 6): weights=NULL (every
  # pre-ver4-25 caller) reproduces the unweighted fit exactly (I11) -
  # w_full is a vector of 1s in that case. When given, `weights` is
  # this task's own shared TRAIN-split fitting weight (aligned to
  # train's own row order) - BGLR's residual variance for row i is
  # sigma^2/weights_i^2, so sqrt(w) is passed (V8.1). valid/test rows
  # get weight 1 - they are masked (NA response) and contribute no
  # likelihood term either way.
  if (is.null(weights)) {
    w_full <- rep(1, nrow(data))
  } else {
    w_full <- c(sqrt(as.numeric(unlist(weights))), rep(1, nrow(valid) + nrow(test)))
  }
  fm <- BGLR(y=y_test,ETA=list(mrk=list(X=X,model='BRR',df0=df0,R2=R2)),
             nIter=nIter,burnIn=burnIn,verbose=FALSE,weights=w_full,
             saveAt=paste(bglr_output_dir,'/brr_',run_id,'_', sep = ""))
  
  y_predicted <- fm$yHat[(nrow(data)-nrow(test)+1):nrow(data)]
  y_actual <- y[(nrow(data)-nrow(test)+1):nrow(data)]

  pearson <- cor(y_predicted, y_actual, method = c("pearson"))
  MSE <- mean((y_predicted - y_actual)^2)
  
  y_predicted_train <- fm$yHat[1:nrow(train)]
  y_actual_train <- y[1:nrow(train)] 
  
  if(nrow(valid)!=0){
      y_predicted_valid <- fm$yHat[(nrow(data)-(nrow(valid)+nrow(test))+1):(nrow(train)+nrow(valid))]
      y_actual_valid <- y[(nrow(data)-(nrow(valid)+nrow(test))+1):(nrow(train)+nrow(valid))] 
  }else{
      y_predicted_valid <- data.frame()
      y_actual_valid <- data.frame()
  }
  
  effect <- data.frame(t(fm[["ETA"]][["mrk"]][["b"]]))

  return(list(r_pearson=pearson, 
              r_MSE=MSE, 
              r_effect=effect, 
              r_y_predicted=y_predicted, 
              r_y_predicted_valid=y_predicted_valid, 
              r_y_predicted_train=y_predicted_train))           
}