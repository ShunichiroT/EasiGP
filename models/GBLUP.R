library(BGLR)
library(stringr)
library(data.table)
library(dplyr)
library(iml)

# Computing Shapley marker effects here is far more expensive than for a
# typical ML model: `true_model()` below does not just re-evaluate an
# already-fitted model - because GBLUP is a transductive kernel model (BGLR
# fits train+valid+test jointly with test labels set to NA), scoring a single
# perturbed row requires rebuilding the whole genomic relationship matrix and
# re-running the ENTIRE BGLR Markov-chain fit (nIter iterations) from scratch.
# With M markers, the iml::Shapley estimator (sample.size=1) makes roughly
# 2*M such calls per explained test sample, and Shapley_num samples are
# explained - i.e. roughly 2*M*Shapley_num full BGLR refits in total. For an
# 8,000-marker dataset that is hundreds of thousands of refits, which is what
# turns this into a multi-day (or worse) run. (Same issue, and same fix, as
# RKHS.R.)
#
# The fixes below only touch the Shapley computation - r_pearson, r_MSE, and
# all three r_y_predicted* outputs are computed from the single main BGLR fit
# exactly as before, with the user's chosen nIter/burnIn, completely untouched.
#  - restrict which markers are perturbed/explained to the top-importance
#    markers (by absolute correlation with the trait - the same cheap,
#    model-agnostic prefilter used in RF.py/SVR.py/KNN.py/RKHS.R). This cuts
#    the number of refits, and the size of the per-refit relationship-matrix
#    computation, proportionally (e.g. 500/8000 markers = ~16x fewer/faster).
#  - use a separate, much smaller nIter/burnIn just for these perturbation
#    refits - they only need a stable point estimate for a marginal-
#    contribution comparison, not full posterior precision, so they don't
#    need anywhere near as many MCMC iterations as the main fit.
# Both are optional/user-configurable (params[5:7]); the defaults below are
# deliberately far below the main model's nIter/burnIn. Since the Shapley
# values were already a Monte Carlo approximation (sample.size=1) even before
# this change, this is a faster approximation of the same quantity rather
# than an exact-vs-approximate change in kind - increase max_shap_features
# and/or the Shapley nIter/burnIn towards the main model's values for a
# closer (but slower) match to the original behaviour.
GBLUP <- function(train, valid, test, params, RESULT_NAME, K_precomputed=NULL){
  
  params <- unlist(params)
  nIter <- as.numeric(params[1])
  burnIn <- as.numeric(params[2])
  get_effect <- as.logical(params[3])
  Shapley_num <- as.numeric(params[4])
  max_shap_features <- params[5]
  Shapley_nIter <- as.numeric(params[6])
  Shapley_burnIn <- as.numeric(params[7])
  # ver4-4 R3.d (blueprint §2.3.2): appended, never inserted (I5). Both
  # default to today's exact behaviour when absent - either because an
  # older, shorter HPARAMETERS['GBLUP'] list was padded with these
  # defaults on the Python side (genomic_prediction.py, RK-8), or because
  # R's own out-of-bounds vector indexing returns NA here, which the
  # is.na() guards below catch identically either way.
  #   params[8] shap_row_offset : first row (0-based) of the len-row
  #     Shapley block below to explain in THIS call - lets Python split
  #     that block's rows across worker PROCESSES (never threads - BGLR's
  #     PID-based saveAt prefix depends on distinct PIDs, invariant I10).
  #   params[9] shap_row_count  : how many rows, starting at
  #     shap_row_offset, THIS call explains. -1 (default) means "every
  #     row of the len-row block", reproducing today's single-process
  #     1:len loop exactly.
  shap_row_offset <- suppressWarnings(as.numeric(params[8]))
  if (is.na(shap_row_offset)) shap_row_offset <- 0
  shap_row_count <- suppressWarnings(as.numeric(params[9]))
  if (is.na(shap_row_count)) shap_row_count <- -1

  # A process- and call-unique id for BGLR's saveAt path - see RKHS.R for the
  # full explanation. In short: the same './Result/<RESULT_NAME>/GBLUP_'
  # prefix would otherwise be shared across every task and every *concurrent*
  # task in a parallel/HPC run, risking one process's intermediate files
  # colliding with another's.
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
  
  # ver4-4 R4.h (blueprint §2.4.2): an optional precomputed genomic
  # relationship matrix - built in Python, optionally on GPU, over
  # train+valid+test in the SAME row order rbind(train,valid,test)
  # produces here - skips this function's own O(N^2*M) scale()+tcrossprod
  # build entirely. Symmetry/finiteness are the CALLER's own
  # responsibility (genomic_prediction.py validates before ever passing
  # K_precomputed - see that module's own R4.h wiring); this function
  # trusts a given K_precomputed exactly as it already trusted a self-
  # built G, with no additional validation here.
  if (!is.null(K_precomputed)) {
    G <- as.matrix(K_precomputed)
  } else {
    X <- scale(data_qtl, center = T, scale = T)
    G <- as.matrix((X %*% t(X)) / ncol(data_qtl))  # same as tcrossprod(X) / p
  }

  #X <- scale(data_qtl)/sqrt(ncol(data_qtl))
  #X <- X[ , colSums(is.na(X)) == 0]
  #X[is.na(X)] <- -1
  y <- as.numeric(unlist(data_pheno))
  
  y_test <- y
  y_test[(nrow(data)-(nrow(valid)+nrow(test))+1):nrow(data)] <- NA
  
  fm <- BGLR(y=y_test,ETA=list(list(K = G, model = "RKHS")),
             nIter=nIter,burnIn=burnIn,verbose=FALSE,saveAt=next_save_prefix('GBLUP'))
  
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
    # this block - see RKHS.R for the full explanation (R's data.frame()
    # constructor silently renames columns whose names aren't valid R
    # identifiers whenever a data.frame is rebuilt, which data_qtl above is;
    # this only matters once this top-k-by-correlation branch is reached).
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
        "GBLUP marker-effect bug: selected %d marker positions but got %d columns back (max_shap_features=%s, n_features=%d).",
        length(top_positions), ncol(data_qtl_shap), as.character(max_shap_features), n_features
      ))
    }

    true_model <- function(newdata) {
      qtl <- rbind(data_qtl_shap,newdata)
      pred <- c(y_test,rep(NA, nrow(newdata)))
      len_beg <- nrow(data_qtl_shap)+ 1
      len_end <- nrow(data_qtl_shap)+ nrow(newdata)
      
      X <- scale(qtl, center = T, scale = T)
      G <- as.matrix((X %*% t(X)) / ncol(qtl))  # same as tcrossprod(X) / p

      # Deliberately using the smaller, fast Shapley_nIter/Shapley_burnIn here
      # (not the main model's nIter/burnIn) - see note above. Each call gets
      # its own unique saveAt prefix (see run_id/next_save_prefix above) -
      # this loop can call BGLR thousands of times per task, and reusing a
      # single fixed path here previously risked one perturbation query's
      # eigendecomposition being silently reused for another, or colliding
      # with a concurrent HPC task's files.
      this_save_prefix <- next_save_prefix('GBLUP_shap')
      f <- BGLR(y=pred,ETA=list(list(K = G, model = "RKHS")),
                nIter=Shapley_nIter,burnIn=Shapley_burnIn,verbose=FALSE,saveAt=this_save_prefix)
      unlink(paste0(this_save_prefix, '*'))
      
      return(f[["yHat"]][len_beg:len_end])
    }
    
    predictor <- Predictor$new(NULL, data = data_qtl_shap, y=fm[["yHat"]], predict.fun=true_model)
    
    effect <- data.frame()
    if(nrow(test) < Shapley_num){len <- nrow(test)}else{len <- Shapley_num}
    # ver4-4 R3.d: row-range fan-out. shap_row_count=-1 (default) explains
    # every row 1:len exactly as before - a positive shap_row_count
    # restricts this call to a contiguous sub-range [shap_row_offset+1,
    # shap_row_offset+shap_row_count] of that SAME len-row block, so N
    # worker processes can each explain a different slice in parallel.
    # colSums(abs(...)) below is then computed over ONLY this call's own
    # row subset - genomic_prediction.py sums each worker's own partial
    # colSums(abs(...)) vector together to reconstruct the SAME total a
    # single, unsplit 1:len call would have produced (colSums(abs(x)) is
    # linear/associative across ROW-disjoint partitions of x - verified
    # in the ver4-4 Stage 5 design record).
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
      # ver4-4 R3.d: this call's own row range explained zero rows (e.g.
      # a defensive shap_row_offset/shap_row_count combination that
      # doesn't overlap [1, len] at all) - report an all-zero contribution
      # for every top_positions column, rather than letting colSums() on
      # a genuinely EMPTY (0-column) data.frame collapse to a zero-length
      # vector, which would silently break the effect_full[top_positions]
      # <- effect assignment below.
      effect <- setNames(rep(0, length(top_positions)), as.character(seq_along(top_positions)))
    } else {
      effect <- effect %>% mutate_all(as.numeric)
      effect <- colSums(abs(effect))
    }
    # effect is now a plain numeric vector indexed 1..length(top_positions),
    # in the SAME order as top_positions (colnames(tmp) was set to
    # seq_along(top_positions) every iteration, so column j always
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
        "GBLUP marker-effect bug: final effect table has %d columns but should have %d (n_features). top_positions had %d entries; all_marker_names had %d entries.",
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