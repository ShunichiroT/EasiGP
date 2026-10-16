# Result

EasiGP writes the output of each run into its own sub-folder here, `Result/<run name>/` (the run name is set in the GUI). Everything in this folder except this file is ignored by git, so your results are never committed by accident.

## What a run folder contains

(The same list appears in section 14 of the [User Manual](../Manual/EasiGP_User_Manual.pdf).)

Names are for a single run; a Parallel-mode run writes one file per batch (`Metric_<batch>.csv` etc.) and the assemble step merges them into the names below. Large files may be gzip-compressed (`.csv.gz`) if you ticked that option.

| File | Contents |
|---|---|
| `Metric.csv` | One row per model, phenotype, population, split ratio and replicate: Pearson correlation, MSE, Spearman correlation and Top-k Coincidence (plus heritability and adjusted prediction accuracy when ASReml-R is used) |
| `Prediction_result_train.csv`, `Prediction_result_valid.csv`, `Prediction_result_test.csv` | Predicted and observed values for the training, validation and test individuals |
| `Marker_effect.csv` | Each model's estimated marker effects (the circos effect rings) |
| `Interaction.csv`, `Attention.csv` | Marker-pair interactions, and attention weights from the GAT models |
| `Weight.csv` | The weight each model received in each weighted ensemble |
| `hyperparameter.csv` | The hyperparameters used (or found by tuning) for each model |
| `Basic_stats.csv` | Basic per-task statistics: split sizes and number of markers |
| `Metric_summary.xlsx`, `Diversity_prediction_theorem.xlsx` | The two Excel summary reports (plain CSV files if `openpyxl` is not installed) |
| `circos_<phenotype>_<population>*.png`, `circos_legend.png` | Circos plots (one shared legend per run) |
| `Pearson correlation.png`, `MSE.png`, `Weight.png`, `Scatter_plot_<phenotype>.png`, ... | Violin, ensemble-weight and scatter plots (plus `.svg` / `.pdf` copies if requested) |
| `step1_config.json` and similar | The configuration the run used, saved so the run can be repeated or reloaded in the GUI |

If a run stops part-way, resubmitting the same job with the same run name resumes from the unfinished tasks using the hidden `.checkpoint_*.json` files and the partial result files in this folder - so don't delete the run folder in between.

Plots and tables for several ASReml-R options are written into one sub-folder per option, with comparison plots in the run folder itself.
