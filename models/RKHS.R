library(BGLR)
library(stringr)
library(data.table)
library(dplyr)
library(iml)

# ver4-4 R4.h (blueprint §2.4.2): squared Euclidean distance via the
# algebraic identity D^2[i,j] = ||x_i||^2 + ||x_j||^2 - 2*(x_i . x_j),
# computed as ONE matrix product (X %*% t(X)) instead of R's own dist(),
# which is single-threaded and does not benefit from R3.c's BLAS
# threading - this identity, expressed as a GEMM, does. Always on (not
# gated by GPU_KERNEL_PRECOMPUTE - a numerics-affecting change at
# float-rounding level, announced via the ver4-4 [GP] NUMERICS: line,
# unconditional per the blueprint's own R4.h design record). Verified
# directly against R's own dist(method="euclidean")^2 across several
# random-matrix trials (max abs difference ~1e-14, floating-point
# rounding order only - see the ver4-4 Stage 5 design record's own test
# evidence). Tiny negative values that can arise from floating-point
# cancellation for a point's distance to itself (mathematically exactly
# 0) are clamped to 0 - dist() itself never returns a negative value, so
# this only ever corrects rounding noise, never a real distance.
.squared_euclidean_dist_gemm <- function(mat) {
  X <- as.matrix(mat)
  xxt <- X %*% t(X)
  sq_norms <- diag(xxt)
  n <- nrow(X)
  D <- outer(sq_norms, rep(1, n)) + outer(rep(1, n), sq_norms) - 2 * xxt
  D[D < 0] <- 0
  D
}

# Computing Shapley marker effects here is far more expensive than for a
# typical ML model: `true_model()` below does not just re-evaluate an
# already-fitted model - because RKHS is a transductive kernel model (BGLR
# fits train+valid+test jointly with test labels set to NA), scoring a single
# perturbed row requires rebuilding the whole distance kernel and re-running
# the ENTIRE BGLR Markov-chain fit (nIter iterations) from scratch. With M
# markers, the iml::Shapley estimator (sample.size=1) makes roughly 2*M such
# calls per explained test sample, and Shapley_num samples are explained -
# i.e. roughly 2*M*Shapley_num full BGLR refits in total. For an 8,000-marker
# dataset that is hundreds of thousands of refits, which is what turns this
# into a multi-day (or worse) run.
#
# The fixes below only touch the Shapley computation - r_pearson, r_MSE, and
# all three r_y_predicted* outputs are computed from the single main BGLR fit
# exactly as before, with the user's chosen nIter/burnIn, completely untouched.
#  - restrict which markers are perturbed/explained to the top-importance
#    markers (by absolute correlation with the trait - the same cheap,
#    model-agnostic prefilter used in RF.py/SVR.py/KNN.py). This cuts the
#    number of refits, and the size of the per-refit distance computation,
#    proportionally (e.g. 500/8000 markers = ~16x fewer/faster).
#  - use a separate, much smaller nIter/burnIn just for these perturbation
#    refits - they only need a stable point estimate for a marginal-
#    contribution comparison, not full posterior precision, so they don't
#    need anywhere near as many MCMC iterations as the main fit.
# Both are optional/user-configurable; the defaults are deliberately far
# below the main model's nIter/burnIn. Since the Shapley
# values were already a Monte Carlo approximation (sample.size=1) even before
# this change, this is a faster approximation of the same quantity rather
# than an exact-vs-approximate change in kind - increase max_shap_features
# and/or the Shapley nIter/burnIn towards the main model's values for a
# closer (but slower) match to the original behaviour.
RKHS <- function(train, valid, test, params, RESULT_NAME, K_precomputed=NULL){
  
  params <- unlist(params)
  nIter <- as.numeric(params[1])
  burnIn <- as.numeric(params[2])
  # Gaussian kernel bandwidth (K = exp(-h*D)). Larger h makes the kernel
  # decay faster with distance (more locally-focused similarity); smaller h
  # treats more distant samples as similar. 1 matches the previous
  # hardcoded behaviour.
  h <- as.numeric(params[3])
  get_effect <- as.logical(params[4])
  Shapley_num <- as.numeric(params[5])
  max_shap_features <- params[6]
  Shapley_nIter <- as.numeric(params[7])
  Shapley_burnIn <- as.numeric(params[8])
  # ver4-4 R3.d - see GBLUP.R's identical note (params[9]/[10] here,
  # since RKHS's own base params list is one element longer than
  # GBLUP's).
  shap_row_offset <- suppressWarnings(as.numeric(params[9]))
  if (is.na(shap_row_offset)) shap_row_offset <- 0
  shap_row_count <- suppressWarnings(as.numeric(params[10]))
  if (is.na(shap_row_count)) shap_row_count <- -1

  # A process- and call-unique id for BGLR's saveAt path. BGLR writes
  # intermediate eigendecomposition/MCMC files to this path; the *same*
  # './Result/<RESULT_NAME>/eig_' prefix is otherwise shared across every
  # task and - critically - across every *concurrent* task in a parallel/HPC
  # run (e.g. run_step1_batch.py array jobs), so two processes can collide
  # on the same files. Sys.getpid() prevents cross-process collisions; a
  # per-call counter (see call_counter below) additionally prevents the many
  # Shapley perturbation re-fits within a single task from reusing each
  # other's files.
  run_id <- paste0(Sys.getpid(), '_', sample.int(.Machine$integer.max, 1))
  call_counter <- 0
  # BGLR's saveAt writes several of its own diagnostic/trace files (MCMC
  # variance-component traces, etc.) as a side effect of every call - these
  # aren't read back by anything else in the pipeline, but shouldn't be left
  # loose directly in Result/<RESULT_NAME>/ alongside the pipeline's own
  # output files. showWarnings=FALSE because this directory gets (re)created
  # once per task (potentially many times, including concurrently across
  # parallel/HPC processes) - it already existing on a later call is
  # expected, not a problem to warn about.
  bglr_output_dir <- paste0('./Result/', RESULT_NAME, '/BGLR_output')
  dir.create(bglr_output_dir, showWarnings = FALSE, recursive = TRUE)
  next_save_prefix <- function(tag) {
    call_counter <<- call_counter + 1
    paste0(bglr_output_dir, '/', tag, '_', run_id, '_', call_counter, '_')
  }

  data <- rbind(train, valid, test)
  data_qtl <- data.frame(lapply(data[,1:(ncol(data)-1)], as.numeric))
  data_pheno <- data[,ncol(data):ncol(data)]
  
  # ver4-4 R4.h: an optional precomputed squared-distance matrix - built
  # in Python, optionally on GPU, over train+valid+test in the SAME row
  # order rbind(train,valid,test) produces here - skips this function's
  # own GEMM-identity distance build entirely. Unlike GBLUP's G, RKHS's
  # own final kernel K = exp(-h*D) depends on h (a per-call, potentially
  # TUNED hyperparameter - see hparam_specs.py), so what can safely be
  # precomputed ONCE outside a hyperparameter search is only D (h-
  # independent), never K itself; K_precomputed here therefore names a
  # precomputed D, not a precomputed K, despite sharing GBLUP.R's
  # parameter name for call-site symmetry. D/mean(D) and exp(-h*D) are
  # always still computed here, using THIS call's own h.
  if (!is.null(K_precomputed)) {
    D <- as.matrix(K_precomputed)
  } else {
    D <- .squared_euclidean_dist_gemm(data_qtl)
  }
  D <- D/mean(D)
  K <- exp(-h*D)
  
  y <- as.numeric(unlist(data_pheno))
  
  y_test <- y
  y_test[(nrow(data)-(nrow(valid)+nrow(test))+1):nrow(data)] <- NA
  
  fm <- BGLR(y=y_test,ETA=list(list(K=K,model='RKHS')),
             nIter=nIter,burnIn=burnIn,verbose=FALSE,saveAt=next_save_prefix('eig'))
  
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
  
  if (get_effect == TRUE){
    all_marker_names <- colnames(data)[1:(ncol(data)-1)]
    n_features <- length(all_marker_names)

    # Cheap, model-agnostic importance proxy: absolute correlation with the
    # trait, computed on the training portion only (O(N*M), no extra model
    # fitting needed to rank candidates).
    #
    # Deliberately uses POSITIONS (column indices), never names, throughout
    # this block. R's data.frame() constructor silently re-validates/renames
    # columns (via make.names(), triggered by the default check.names=TRUE)
    # whenever a data.frame is rebuilt - which is exactly what
    # `data_qtl <- data.frame(lapply(...))` does above. If marker IDs aren't
    # valid R identifiers (e.g. numeric-starting, common for real marker
    # IDs), data_qtl's actual column names can end up different from
    # all_marker_names (colnames(data)) - this only matters once this
    # top-k-by-correlation branch is reached (i.e. once n_features exceeds
    # max_shap_features), which is exactly the "only breaks with >500
    # markers" symptom this was causing. Using positions sidesteps the whole
    # class of bug regardless of the exact renaming mechanism.
    if (!identical(max_shap_features, 'all') && n_features > as.numeric(max_shap_features)) {
      k <- as.numeric(max_shap_features)
      train_idx <- 1:nrow(train)
      correlations <- abs(sapply(data_qtl[train_idx, ], function(col) {
        r <- suppressWarnings(cor(col, y[train_idx]))
        if (is.na(r)) 0 else r
      }))
      top_positions <- order(correlations, decreasing = TRUE)[1:k]
    } else {
      top_positions <- 1:n_features
    }

    data_qtl_shap <- data_qtl[, top_positions, drop = FALSE]
    # Diagnostic check: catches any column-count mismatch here, at the exact
    # point it would first occur, rather than much later on the Python side
    # where it's far harder to trace back to the cause.
    if (ncol(data_qtl_shap) != length(top_positions)) {
      stop(sprintf(
        "RKHS marker-effect bug: selected %d marker positions but got %d columns back (max_shap_features=%s, n_features=%d).",
        length(top_positions), ncol(data_qtl_shap), as.character(max_shap_features), n_features
      ))
    }

    true_model <- function(newdata) {
      qtl <- rbind(data_qtl_shap,newdata)
      pred <- c(y_test,rep(NA, nrow(newdata)))
      len_beg <- nrow(data_qtl_shap)+ 1
      len_end <- nrow(data_qtl_shap)+ nrow(newdata)
      
      D <- .squared_euclidean_dist_gemm(qtl)
      D <- D/mean(D)
      K <- exp(-h*D)
      
      # Deliberately using the smaller, fast Shapley_nIter/Shapley_burnIn here
      # (not the main model's nIter/burnIn) - see note above. Each call gets
      # its own unique saveAt prefix (see run_id/next_save_prefix above) -
      # this loop can call BGLR thousands of times per task, and reusing a
      # single fixed path here previously risked one perturbation query's
      # eigendecomposition being silently reused for another, or colliding
      # with a concurrent HPC task's files.
      this_save_prefix <- next_save_prefix('eig_shap')
      f <- BGLR(y=pred,ETA=list(list(K=K,model='RKHS')),
                nIter=Shapley_nIter,burnIn=Shapley_burnIn,verbose=FALSE,saveAt=this_save_prefix)
      unlink(paste0(this_save_prefix, '*'))
      
      return(f[["yHat"]][len_beg:len_end])
    }
    
    predictor <- Predictor$new(NULL, data = data_qtl_shap, y=fm[["yHat"]], predict.fun=true_model)
    
    effect <- data.frame()
    if(nrow(test) < Shapley_num){len <- nrow(test)}else{len <- Shapley_num}
    # ver4-4 R3.d - see GBLUP.R's identical note.
    effective_row_count <- if (shap_row_count < 0) len else max(0, min(shap_row_count, len - shap_row_offset))
    row_start <- shap_row_offset + 1
    row_end <- shap_row_offset + effective_row_count
    if (effective_row_count > 0) {
      for(j in row_start:row_end){
        shapley <- Shapley$new(predictor, x.interest = data_qtl_shap[j+nrow(train)+nrow(valid), ], sample.size = 1)
        tmp <- data.frame(t(shapley$results[,1:2]))
        # Label these columns by POSITION within top_positions (1, 2, 3, ...)
        # rather than by marker name - avoids relying on iml::Shapley's
        # internal feature-name handling matching data_qtl_shap's names
        # exactly, which is itself subject to the same R renaming risk.
        colnames(tmp) <- as.character(seq_along(top_positions))
        effect <- dplyr::bind_rows(effect, tmp[2,])
      }
    }
    
    if (nrow(effect) == 0) {
      # ver4-4 R3.d - see GBLUP.R's identical note.
      effect <- setNames(rep(0, length(top_positions)), as.character(seq_along(top_positions)))
    } else {
      effect <- effect %>% mutate_all(as.numeric)
      effect <- colSums(abs(effect))
    }
    # effect is now a plain numeric vector indexed 1..length(top_positions),
    # in the SAME order as top_positions/top_marker_names (colnames(tmp) was
    # set to seq_along(top_positions) every iteration, so column j always
    # corresponds to top_positions[j]).
    effect <- as.numeric(effect)[order(as.numeric(names(effect)))]

    # Reassemble into a full-width vector (one entry per marker, in the
    # original column order, with 0 for any marker outside the shortlist
    # above) - genomic_prediction.py assigns column names positionally from
    # the full marker list, so this must always have exactly n_features
    # columns regardless of how many markers were actually explained.
    # Positional assignment throughout - no name matching involved.
    effect_full <- rep(0, n_features)
    effect_full[top_positions] <- effect
    effect <- data.frame(t(effect_full))
    colnames(effect) <- all_marker_names
    # Final diagnostic check: this is the exact object returned to Python,
    # so if it's ever the wrong width, this pinpoints it precisely instead
    # of failing three function calls later with a hard-to-trace error.
    if (ncol(effect) != n_features) {
      stop(sprintf(
        "RKHS marker-effect bug: final effect table has %d columns but should have %d (n_features). top_positions had %d entries; all_marker_names had %d entries.",
        ncol(effect), n_features, length(top_positions), length(all_marker_names)
      ))
    }
  }else{
    effect <- data.frame()
  }
  
  return(list(r_pearson=pearson, 
              r_MSE=MSE, 
              r_effect=effect, 
              r_y_predicted=y_predicted, 
              r_y_predicted_valid=y_predicted_valid, 
              r_y_predicted_train=y_predicted_train))
  
}