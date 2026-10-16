library(BGLR)
library(stringr)
library(data.table)
library(dplyr)

# Bug fix (Requirement.md follow-up to CHANGE_NOTES.md) - see rrBLUP.R's
# identical helper for the full root-cause explanation: BGLR's Gibbs
# sampler draws from R's global RNG, and this file never fixed it, so two
# calls with identical data/hyperparameters (a re-evaluated search
# candidate, or the final confirmatory fit vs. whatever earned that
# candidate its validation-time "win") are independent MCMC draws rather
# than reproductions of each other - noise the search-space/selection-rule
# fixes in models/hyperparameter_tuning.py cannot catch, because they
# operate one level above the model's own reproducibility. Seeding from the
# data being fit (never the hyperparameters) both restores reproducibility
# and gives every candidate scored against the SAME data a shared MCMC
# starting point ("common random numbers"), so a difference in validation
# score reflects probIn/counts/nIter/burnIn actually changing something,
# not which candidate got MCMC-lucky.
.bglr_seed <- function(y, X) {
  y_obs <- y[is.finite(y)]
  basis <- length(y) * 2654435761 + length(y_obs) * 40503 +
    sum(y_obs) * 97 + sum(y_obs^2) + ncol(X) * 15487469 + nrow(X)
  as.integer(abs(basis) %% 2147483397) + 1L
}


BayesB <- function(train, valid, test, params, RESULT_NAME, weights = NULL){
  
  params <- unlist(params)
  nIter <- as.numeric(params[1])
  burnIn <- as.numeric(params[2])
  # probIn is the prior probability that a marker has a nonzero effect
  # (BayesB's defining hyperparameter - it sets the sparsity assumption),
  # and counts controls how strongly that prior is held.
  probIn <- as.numeric(params[3])
  counts <- as.numeric(params[4])

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
  # seed immediately before the fit.
  set.seed(.bglr_seed(y_test, X))
  # ver4-25, R8 - see rrBLUP.R's identical comment.
  if (is.null(weights)) {
    w_full <- rep(1, nrow(data))
  } else {
    w_full <- c(sqrt(as.numeric(unlist(weights))), rep(1, nrow(valid) + nrow(test)))
  }
  fm <- BGLR(y=y_test,ETA=list(mrk=list(X=X,model='BayesB',probIn=probIn,counts=counts)),
             nIter=nIter,burnIn=burnIn,verbose=FALSE,weights=w_full,
             saveAt=paste(bglr_output_dir,'/bayes_',run_id,'_', sep = ""))
  
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