# Ensemble AnalySis with Interpretable Genomic Prediction (EasiGP): Computational Tool for Interpreting Ensembles of Genomic Prediction Models 
This code is used for "Ensemble AnalySis with Interpretable Genomic Prediction (EasiGP): Computational Tool for Interpreting Ensembles of Genomic Prediction Models". (https://doi.org/10.1002/tpg2.70138)

EasiGP analyses the ensemble of multiple diverse genomic prediction models at the genome level in crop breeding programs.
Circos plots are then constructed using the effect of each genomic marker region and the interactions of genomic markers for a target trait.
With a constructed circos plot, we can visually compare the inferred trait genetic architecture of genomic prediction models to deepen our understanding of their predictive behaviour at the genome level.
The comparison of the inferred genomic marker effects with known key genome regions also enables the discovery of potential new genome regions that have not been well-investigated in previous studies.

**New to EasiGP?** The [User Manual](Manual/EasiGP_User_Manual.pdf) (PDF, 23 pages) walks through every GUI field and the whole configuration flow, with a quick start on the example data.

## Table of Contents

- [What's new in this release](#whats-new-in-this-release)
- [Description](#description)
- [EasiGP setup procedure](#easigp-setup-procedure)
  - [Full version or Light version - which do I need?](#full-version-or-light-version---which-do-i-need)
  - [Ready-made image, or set up from source code - which method do I need?](#ready-made-image-or-set-up-from-source-code---which-method-do-i-need)
  - [Option A: Using a ready-made image (recommended)](#option-a-using-a-ready-made-image-recommended)
    - [Getting the image file](#getting-the-image-file)
    - [Getting a Claude Code access token (Full version only)](#getting-a-claude-code-access-token-full-version-only)
    - [For a local PC/laptop, using Docker](#for-a-local-pclaptop-using-docker)
    - [For HPC, using Apptainer](#for-hpc-using-apptainer)
      - [Bunya (UQ) or Gadi (NCI) users: an easier option using a virtual desktop](#bunya-uq-or-gadi-nci-users-an-easier-option-using-a-virtual-desktop)
    - [Building the image files yourself](#building-the-image-files-yourself)
  - [Option B: Setting up manually from source code](#option-b-setting-up-manually-from-source-code)
    - [For laptop](#for-laptop)
      - [Laptop: first-time installation (do once)](#laptop-first-time-installation-do-once)
      - [Laptop: every time you use EasiGP](#laptop-every-time-you-use-easigp)
    - [For HPC](#for-hpc)
      - [HPC: first-time installation (do once)](#hpc-first-time-installation-do-once)
      - [HPC: every time you use EasiGP](#hpc-every-time-you-use-easigp)
    - [Optional: ASReml-R mixed-model adjustment (needs a paid licence)](#optional-asreml-r-mixed-model-adjustment-needs-a-paid-licence)
- [User manual](#user-manual)
- [References](#references)

---

## What's new in this release

- **User manual.** A new illustrated PDF guide to every GUI field, the execution modes, the input files, the output files and troubleshooting: [`Manual/EasiGP_User_Manual.pdf`](Manual/EasiGP_User_Manual.pdf).
- **ASReml-R mixed-model adjustment (optional, needs a paid licence).** Fit BLUEs or BLUPs (without or with a genomic relationship matrix) and use them as the adjusted phenotypes for the models. Several options can be run side by side and compared, and EasiGP checks that ASReml-R is installed and licensed before a job starts. See [Optional: ASReml-R mixed-model adjustment](#optional-asreml-r-mixed-model-adjustment-needs-a-paid-licence).
- **Tool mode.** A third way to use the GUI, for finished runs: compare prediction performance across result folders (Pearson correlation / MSE, adjusted accuracy, Spearman correlation, across-result scatter plots) and run per-run diagnostics, without refitting anything.
- **More ways to judge accuracy.** Spearman correlation, Top-k Coincidence and, with ASReml-R, heritability and adjusted prediction accuracy, each with its own plots, plus an optional calibration-slope sheet in `Metric_summary.xlsx`.
- **Circos plot improvements.** A consensus ring across models, highlighting of interactions that come from genome regions you choose, and top-marker labelling.
- **Load a saved configuration.** Restore the settings of an earlier run in the GUI from a saved GUI state or a run's config file.
- **Better HPC planning.** Resource suggestions are calibrated from the measured time and memory of earlier runs, and a small pilot run (`run_step1_batch.py --pilot`) can measure your own data before you commit a large allocation.
- **Biological prior network.** Gene locations can be looked up automatically from an annotation file (GFF3/GTF) or Ensembl before any AI-agent step is used.

The previous release roughly doubled the codebase and added three selectable prediction models (ExtraTrees, XGBoost, EBM), a fourth weighted-ensemble method (Analytic least-squares), marker-pair interactions from twelve models (each drawn as its own circos ring), task-level and model-level parallelism, optional gzip compression of large result files, a streaming result-assembly path for large Parallel-mode runs, and the two Excel summary reports (`Metric_summary.xlsx`, `Diversity_prediction_theorem.xlsx`).

## Description
EasiGP is a web-based application (no coding required to use it day-to-day) that runs and compares multiple genomic prediction models, then visualises what each model has learned as a circos plot. It also includes several optional add-ons for preparing your data and for building on top of your results.

- **Models**: fifteen individual genomic prediction models, plus an ensemble that combines them, are available to select and run from the GUI.
   - ridge regression best linear unbiased prediction (rrBLUP), genomic best linear unbiased prediction (GBLUP), BayesB and reproducing kernel Hilbert Space (RKHS): BGLR (Pérez and de Los Campos, 2014) in R
   - Random forest (RF), Extra Trees, support vector regression (SVR) and K-nearest neighbours (KNN): Scikit-learn (Pedregosa et al., 2012) in Python
   - XGBoost and Explainable Boosting Machine (EBM): optional add-on models (need the `xgboost` / `interpret` packages respectively). If either package isn't installed, the GUI greys out that model with a note rather than failing silently, and a headless/HPC run stops with an actionable install message before any work starts.
   - Multilayer perceptron (MLP): PyTorch (Paszke et al., 2019) in Python
   - Four graph attention network (GAT) variants - GAT infinitesimal, GAT fully-connected, GAT prior-knowledge, and GAT biological prior-knowledge (which learns over genes rather than individual markers, using a curated or FLASH-P-generated gene-interaction network - see "Biological prior network" below) - all built with PyTorch Geometric (Fey et al., 2019) in Python
   - **ensemble**: combines the predictions, marker effects, and marker-pair interactions of every other selected model, optionally with a chosen weighting method (four are available: Linear transformation, Nelder Mead, Bayesian optimisation, or the closed-form Analytic least-squares) rather than a plain average
   - Any model's hyperparameters can either be set manually, or automatically tuned (a choice of search algorithms, e.g. grid, random, or Bayesian search, several of which can evaluate candidates in parallel) against a validation set

- **Interpretability**: twelve of the models above can report which pairs of markers they treat as interacting (via TreeSHAP, the Friedman H-statistic, Neural Interaction Detection, EBM's own native pairwise terms, or - for the four BGLR models, disclosed as approximate - a surrogate TreeSHAP/H-statistic route), and four of the GAT variants report their learned attention weights over marker pairs. Each emitting model gets its own ring on the circos plot, so different models' inferred architectures can be compared side by side rather than collapsed into one.

- **Data preprocessing** (optional, all configured from the GUI before running the models):
   - **LD pruning**: removes markers in high linkage disequilibrium, with an optional diagnostic LD-decay plot
   - **RF marker importance filtering**: narrows the marker set further, down to the top markers by Random Forest importance (by percentage or fixed count) - typically run after LD pruning
   - **Biological prior network**: builds the gene-level interaction network the GAT biological prior-knowledge model needs, either from a gene-interaction network file you already have, or generated automatically by [FLASH-P](https://flash-p.com/) (Mitsanis et al., 2026) (which needs access to Claude - see "Setup procedure" below); gene genomic coordinates can likewise be looked up automatically instead of hand-curated
   - **ASReml-R mixed-model adjustment** (optional; needs ASReml-R and a paid licence, which are *not* included - see "Optional: ASReml-R mixed-model adjustment" below): fits a mixed model and uses the resulting BLUE/BLUP values as the phenotypes for the models
   - **PLINK support**: genotype data can be supplied either as a CSV file or as a PLINK1 binary fileset (`.bed`/`.bim`/`.fam`), converted automatically. When only the biological prior-knowledge model is selected on a PLINK fileset, EasiGP now avoids materialising the full genotype matrix at all, extracting only the markers each gene actually needs

- **Circos plots** compare the inferred trait genetic architecture across models: each model's marker effects are shown as a ring, alongside a separate marker-pair interaction/attention ring per contributing model and (optionally) known key gene regions for comparison. Most of a circos plot's own display settings (label size, tick spacing, seam gap, how much to widen a marker/gene region so it's visible, etc.) are suggested automatically based on your data, and can be overridden manually if needed. One shared legend is produced per run, rather than one per plot. A consensus ring summarises the models, and interactions coming from genome regions you name can be highlighted.

- **Two ways to run a pipeline**:
   - **Sequential**: runs every task (population × phenotype × model, etc.) one after another, either directly in the GUI or as a single submitted HPC job
   - **Parallel**: splits the same work into a batch of independent HPC jobs that run at the same time, then assembles their results together afterwards - much faster for a large number of tasks on a cluster. Within that, two further and independently switchable levels of parallelism are available: fanning a batch's individual tasks across worker processes, and fanning a single task's models across worker processes - useful on a single large node even without a full HPC array. If any individual task fails partway through (e.g. a bad hyperparameter combination), completed tasks are automatically saved as they finish (optionally gzip-compressed) and a resubmitted job picks up only the unfinished ones, rather than starting over. A GUI panel suggests CPU/GPU/memory settings and an array layout for your job, based on your configuration - treat this as a starting point and pilot a small batch before committing a large allocation to it.

- **Tool mode**: a separate GUI mode for finished runs. It compares the prediction performance of several result folders and runs per-run diagnostics, reading the saved result files only - nothing is refitted.

- **Data**: example data files based on the TeoNAM dataset (Chen et al., 2019), MaizeNAM dataset (Buckler et al., 2009) and Arabidopsis dataset (the 1001 Genomes Consortium, 2016) to run this tool
  - Details are explained in "README.md" in the Data folder

- **Result**: folder used as storage for output files from this tool, including per-model prediction/effect/interaction/attention tables, circos plots (one legend per run, one interaction/attention ring per contributing model), an ensemble-weight bar chart, and two Excel summary reports (`Metric_summary.xlsx` for prediction accuracy, `Diversity_prediction_theorem.xlsx` for the Diversity Prediction Theorem breakdown of each ensemble) - both of which degrade to plain CSV files if `openpyxl` isn't installed

- Key files, for anyone working with the code directly (this isn't needed to use the GUI):
   - `Environments/environment_windows.yml` / `Environments/environment_linux.yml`: the packages needed to build EasiGP's own environment on Windows / Linux
   - `main_app.py`: the GUI application itself (this is what `streamlit run` launches - see "Setup procedure" below)
   - `genomic_prediction.py`: the genomic prediction pipeline itself (scenario enumeration, data splitting/preprocessing orchestration, model dispatch, checkpointing)
   - `Manual/EasiGP_User_Manual.pdf`: the user manual (see "User manual" below)
   - `Preprocess/`: the data preprocessing add-ons described above (LD pruning, RF marker filtering, biological prior network, PLINK conversion)
   - `models/`: the prediction models, the naive/weighted ensembles, and the hyperparameter-tuning engine
   - `model_registry.py` / `hparam_specs.py` / `circos_geometry.py`: single-source-of-truth reference modules - respectively, which models can emit interactions/attention and which need an optional package; every model's hyperparameter layout and tunable ranges; and circos ring/label geometry - each imported by several of the files above rather than re-implemented in each one
   - `checkpoint_utils.py` / `assemble.py` / `batch_reader.py`: result-file durability (checkpointing, optional compression) and, for Parallel mode, merging per-batch results back together (streamed, so large runs don't need to fit in memory at once)
   - `intra_batch_parallel.py` / `intra_task_parallel.py` / `resource_profiles.py`: the task-level and model-level parallelism described above, and the GUI's compute-resource/array-layout advisor
   - `circos_plot.py` / `scatter_plot.py` / `metric_plot.py` / `weight_plot.py` / `attention_histogram.py`: the plots described above
   - `metric_summary.py` / `diversity_summary.py`: the two Excel summary reports described above
   - `asreml_syntax.py` / `asreml_accuracy_metrics.py` / `Preprocess/asreml_blup.py` / `Preprocess/asreml_fit.R` / `Preprocess/asreml_license.py`: the optional ASReml-R adjustment (model syntax and validation, accuracy metrics, the R/ASReml engine, and the install/licence check); `blue_weights.py` / `strategy_results.py` / `reporting_pipeline.py` handle BLUE weighting and the combining and reporting of runs with several ASReml options
   - `accuracy_metrics_plot.py` / `plot_style.py` / `circos_highlight.py` / `interaction_reduce.py` / `stability_summary.py`: the accuracy-metric plots, shared plot styling, circos region highlighting, memory-bounded reduction of interaction tables, and selection-stability summaries
   - `result_comparison.py` / `tool_diagnostics.py` / `run_tool.py`: Tool mode (comparing and diagnosing finished runs)
   - `gui_config_loader.py`: restores the GUI from a saved GUI state or config file
   - `resource_advise.py` / `resource_calibration.py` / `resource_telemetry.py` / `async_lane.py`: measured resource use, calibrated resource suggestions, and the worker that lets CPU and GPU models run at the same time
   - `tools/`: stand-alone command-line tools (e.g. per-run diagnostics and resource-advice scripts); `tests/`: the automated tests
   - `run_sequential.py` / `run_step1_batch.py` / `run_step2_assemble.py`: the headless scripts an HPC job actually runs, generated for you by the GUI's own "Generate and save job files" option - you shouldn't need to edit these by hand

## EasiGP setup procedure

EasiGP comes in two forms, and there are two different ways to get either one running. Read the next two sections first to work out which combination is right for you - then jump straight to the matching set of steps.

### Full version or Light version - which do I need?

- **Full version** includes everything: the complete EasiGP toolkit, plus its Claude Code integration used by the "Biological prior network" preprocessing step to auto-generate a gene-interaction network with FLASH-P, instead of requiring one to be supplied manually. To use this you'll need access to Claude (an Anthropic account with API or subscription billing) - see "Getting a Claude Code access token" below.
- **Light version** includes the complete EasiGP toolkit only, with no Claude Code/FLASH-P integration and nothing extra to set up for it. You can still supply your own gene-interaction network file by hand with this version - you just can't have EasiGP generate one for you automatically. Choose this if you don't have (or don't need) Claude access, or want the simpler setup.

Both versions give you the same genomic prediction models, hyperparameter tuning, data preprocessing, and circos plot generation - the only difference is the automatic gene-network generation.

### Ready-made image, or set up from source code - which method do I need?

- **Option A - ready-made image (recommended for most people).** Someone (a colleague, a lab administrator, or you, following the "Building the image files yourself" note below) has already packaged EasiGP and everything it depends on into a single file, ready to run. You don't install Python, conda, or any packages yourself - you just load this file and run it. This is by far the easiest path if you don't have a coding background.
- **Option B - set up manually from the source code.** You clone the EasiGP code from GitHub yourself and build the environment it needs using conda. This gives you direct access to every file so you can inspect or modify the code, but involves more setup steps and some familiarity with the command line. Choose this option if you want to use the optional ASReml-R mixed-model adjustment, which needs a paid licence and is not included in the ready-made images.

If you're not sure, use Option A.

---

### Option A: Using a ready-made image (recommended)

A few concepts worth knowing before you start, if you haven't used these before:

- **Docker** and **Apptainer** are both tools that run a pre-packaged, self-contained copy of an application - all its software dependencies already included - without you needing to install any of those dependencies yourself. Docker is normally used on your own laptop/PC; Apptainer (sometimes still called by its older name, Singularity) is the equivalent commonly available on HPC/university computing clusters, since Docker itself usually isn't allowed there for security reasons.
- An **image** is that self-contained, ready-to-run package. A **tar file** (e.g. `EasiGP.tar`) is simply how that image is saved to a single file so it can be copied between computers - similar in spirit to a zip file.
- A **container** is a running copy of an image - i.e. the image is the "template", and the container is EasiGP actually running from it.

#### Getting the image file

EasiGP's ready-made images are published on [Docker Hub](https://hub.docker.com/r/shunichirot/easigp/tags) rather than distributed as downloadable files directly from GitHub - the images are several gigabytes each (over 2GB), too large for GitHub to host directly. That Docker Hub page shows the exact tags available for the Full and Light versions, and the exact `docker pull` command to use for each - you'll need this for the steps below, wherever you see `<version>` (Full version) or `<version>_light` (Light version).

- On a **local PC**, using Docker (below), you pull the image directly from that page - no separate download or `.tar` file needed at all.
- On **HPC**, using Apptainer (below), you can often build directly from Docker Hub too, the same way - a `.tar` file is only needed as a fallback, for an HPC whose login/build environment doesn't have internet access, or if you'd simply rather transfer the image over yourself. See that section for exactly how to create one, if you need it.

#### Getting a Claude Code access token (Full version only)

The Full version's Claude Code integration needs to authenticate with Anthropic, using an access token you generate once yourself. Since the computer running EasiGP (especially on HPC) often can't open a web browser to log in, generate this token on a separate machine where you *can* log in with a browser:

1. Install Claude Code there if you haven't already (see [Anthropic's Claude Code documentation](https://docs.claude.com) for installation instructions).
2. Run `claude setup-token` and follow the prompts to log in.
3. Copy the token it prints out - you'll paste this in below, wherever you see `YOUR TOKEN`. Keep it private; treat it like a password.

This token works with either an API-billed Anthropic account or a Claude subscription (Pro/Max/Team/Enterprise).

#### For a local PC/laptop, using Docker

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) if you don't already have it, and make sure it's running.
2. Open a terminal (**Command Prompt** or **PowerShell** on Windows; **Terminal** on Mac/Linux).
3. Pull the image directly from Docker Hub - see "Getting the image file" above for the exact tag to use in place of `<version>`/`<version>_light` below:
   - Full version: `docker pull shunichirot/easigp:<version>`
   - Light version: `docker pull shunichirot/easigp:<version>_light`

   This downloads several gigabytes, so it can take a while the first time - you only need to repeat it when a new version is released.
4. Run it:
   - Full version:
     ```
     docker run -p 8501:8501 -it -e CLAUDE_CODE_OAUTH_TOKEN=YOUR TOKEN shunichirot/easigp:<version> streamlit run main_app.py --server.address=0.0.0.0 --server.port=8501
     ```
   - Light version:
     ```
     docker run -p 8501:8501 -it shunichirot/easigp:<version>_light streamlit run main_app.py --server.address=0.0.0.0 --server.port=8501
     ```
   (If port 8501 is already in use on your machine, change *both* `8501`s before the colon-separated pair, e.g. `-p 8502:8501`, and use that new port number in the next step instead.)
5. Once you see EasiGP's own startup messages in the terminal, open your browser and go to `http://localhost:8501` (or whichever port you chose in step 4).
6. When you're done, press `Ctrl+C` in the terminal to stop the container.

Any files EasiGP creates (results, generated networks, etc.) live inside the container and are lost when it stops, unless you mount a folder from your own computer into it - add `-v "PATH ON YOUR COMPUTER":/workspace/data` to the `docker run` command above (before `shunichirot/easigp:<version>`) to keep a folder in sync, e.g. `-v "$(pwd)/data":/workspace/data` on Mac/Linux or `-v "${PWD}\data":/workspace/data` in PowerShell.

**Next time you use EasiGP (Docker):** the image is already on your computer, so skip step 3 (the pull) unless a new version has been released. Make sure Docker Desktop is running, run the same `docker run ...` command from step 4, open `http://localhost:8501`, and press `Ctrl+C` in the terminal when you are done.

#### For HPC, using Apptainer

Apptainer builds a **sandbox** - an extracted, folder-based copy of the image - then runs EasiGP from that sandbox. Do this once; you can reuse the same sandbox for every future session.

1. Log in to your HPC and make sure Apptainer is available (try `apptainer --version`; if that fails, check your HPC's documentation for how to load it, e.g. `module load apptainer`).
2. Build the sandbox - see "Getting the image file" above for the exact tag to use in place of `<version>`/`<version>_light` below. There are two ways to do this, depending on whether your HPC's login node has outbound internet access (many do; some, especially compute nodes, don't):

   **a. Directly from Docker Hub (simplest - try this first):**
   ```
   mkdir -p /scratch/user/$USER/EasiGP
   apptainer build --sandbox /scratch/user/$USER/EasiGP docker://shunichirot/easigp:<version>
   ```
   (Light version: replace both `EasiGP` in the paths with `EasiGP_light`, and `<version>` with `<version>_light`.) If this works, skip straight to step 3 below.

   **b. Via a `.tar` file (if step a fails, or your login node has no internet access):**
   - On a *different* computer that has both internet access and Docker installed - your own laptop is usually easiest - download and re-package the image as a single file:
     ```
     docker pull shunichirot/easigp:<version>
     docker save -o EasiGP.tar shunichirot/easigp:<version>
     ```
     (Light version: use `<version>_light` and name the file `EasiGP_light.tar` instead.) This file will be several gigabytes - the same size as the image itself.
   - Transfer that `.tar` file to your HPC account (e.g. via `scp`/`rsync`, or your HPC's own file-transfer tool) - it doesn't need to go anywhere special, just somewhere in your own storage.
   - Build the sandbox from it:
     ```
     mkdir -p /scratch/user/$USER/EasiGP
     apptainer build --sandbox /scratch/user/$USER/EasiGP docker-archive:///path/to/EasiGP.tar
     ```
     (Light version: replace `EasiGP` in the paths with `EasiGP_light`, and point the last part at `EasiGP_light.tar` instead.) Replace `/path/to/EasiGP.tar` with wherever you actually uploaded it.

   (Adjust `/scratch/user/$USER/...` if your HPC uses a different path for your own storage - `$USER` fills in your own username automatically, so you shouldn't need to type it yourself.)

   Either way, this can take a while the first time - it's unpacking the whole image.
3. Start an interactive job on a compute node first (running Apptainer directly on the login node is usually against HPC policy) - check your own HPC's documentation for the right command, since this varies between clusters. For example, for Slurm:
   ```
   salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem=10G \
     --job-name=EasiGP --time=01:00:00 \
     --partition=YOUR PARTITION --account=YOUR GROUP \
     srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l
   ```
4. Once your interactive job is running, run the container:
   - Full version - generate a Claude Code access token first (see "Getting a Claude Code access token" above), then:
     ```
     export APPTAINERENV_CLAUDE_CODE_OAUTH_TOKEN="YOUR TOKEN"
     apptainer run /scratch/user/$USER/EasiGP
     cd /scratch/user/$USER/EasiGP/workspace/EasiGP/
     streamlit run main_app.py --server.address=0.0.0.0 --server.port=8501
     ```
   - Light version:
     ```
     apptainer run /scratch/user/$USER/EasiGP_light
     cd /scratch/user/$USER/EasiGP_light/workspace/EasiGP/
     streamlit run main_app.py --server.address=0.0.0.0 --server.port=8501
     ```
   Note the compute node name shown by the scheduler (e.g. in your shell prompt, or via `squeue --me`) - you'll need it next.
5. From your **local machine** (not the HPC), open a new terminal and set up an SSH tunnel to that compute node:
   ```
   ssh -N -L 8501:COMPUTE_NODE_NAME:8501 YOUR_ACCOUNT@YOUR_HPC_LOGIN_ADDRESS
   ```
   For example, on UQ's Bunya cluster: `ssh -N -L 8501:bun128:8501 USERNAME@bunya.rcc.uq.edu.au`. Leave this terminal open for as long as you're using EasiGP.
6. Open `http://localhost:8501` in your browser.
7. When you're done, close the browser tab, stop the SSH tunnel (`Ctrl+C` in that terminal), and end your interactive job (`exit`, or however your scheduler expects you to release it).

**Next time you use EasiGP (Apptainer):** steps 1-2 (building the sandbox) are one-time only - skip them unless a new version has been released. Each time, start at step 3: start an interactive job, run steps 4-6 (run the container, start Streamlit, open the SSH tunnel and the browser), and finish with step 7.

##### Bunya (UQ) or Gadi (NCI) users: an easier option using a virtual desktop

If you're on UQ's Bunya or NCI's Gadi, there's a simpler alternative to steps 3-7 above: both offer a **virtual desktop** - a full graphical Linux desktop, running on a compute node, that opens directly in your own browser. Since your browser and the compute node running EasiGP end up in the same place, this skips the separate interactive-job and SSH-tunnel steps entirely - you just open a terminal *inside* the virtual desktop for step 2 and 4 above, then open a second browser tab (e.g. Firefox) *inside that same desktop* for step 6, instead of tunnelling back to your own machine.

- **Bunya**: go to [onBunya](https://bunya-ondemand.rcc.uq.edu.au/pun/sys/dashboard) and launch a desktop session from **Interactive Apps** (see UQ RCC's [OnDemand Guide](https://github.com/UQ-RCC/hpc-docs/blob/main/guides/OnDemand-Guide.md) for the full walkthrough). This already *is* your interactive job, so step 3 above isn't needed - open a terminal from within the desktop and continue from step 2.
  - If you've separately set up EasiGP's conda environment on Bunya (Option B below) and added conda's own initialisation to your `.bashrc`, onBunya's own desktop sessions won't start until that's removed (`conda init --reverse`) - see UQ RCC's guide for details. This doesn't apply to the Apptainer route described here on its own.
- **Gadi**: go to the [Australian Research Environment (ARE)](https://are.nci.org.au/) and launch a **VDI** (Virtual Desktop Infrastructure) session (see this [Gadi/ARE platform guide](https://acdguide.github.io/BigData/platforms/platforms-nci-gadi.html) for more background). Open a terminal from within the desktop and continue from step 2.
  - Gadi's compute nodes (including VDI sessions) typically don't have general internet access - this doesn't affect the Light version at all, but may prevent the Full version's Claude Code/FLASH-P (Mitsanis et al., 2026) features from reaching Anthropic's API from inside a VDI session, and may also mean option 2a above (building directly from Docker Hub) doesn't work from a VDI session either, even though it might from the login node - option 2b (via a `.tar` file) always works regardless. Check with NCI if you specifically need internet access from a VDI/compute-node session for this.

#### Building the image files yourself

If you're the one preparing/publishing these images (e.g. you have the EasiGP Dockerfile and want to build and push fresh ones to Docker Hub), see the Dockerfile in the [`Docker image/`](Docker%20image) folder - its own comments give the exact `docker build` commands, including the `--build-arg INSTALL_FLASHP=false` option used to produce the Light version.

---

### Option B: Setting up manually from source code

Option B has two parts for each of laptop and HPC:

- **First-time installation** - clone the code, build the conda environment and install `plink2`. Do this **once** (and again only when a new EasiGP version changes the packages it needs).
- **Every time you use EasiGP** - activate the environment, start the app and open it in your browser. Do this **each time** you want to use EasiGP; you do not repeat the installation.

#### For laptop

##### Laptop: first-time installation (do once)

1. Clone the EasiGP folder from GitHub (https://github.com/ShunichiroT/EasiGP) and store it locally.
2. Convert the format of your genotype/phenotype data into the format EasiGP requires (details: https://github.com/ShunichiroT/EasiGP/tree/main/Data).
3. Start Anaconda (Anaconda Prompt on Windows, Terminal on Mac / Linux) and change directory to the main EasiGP folder: `cd YOUR PATH TO EasiGP/EasiGP`. The environment files are in its `Environments/` subfolder, so run the next step from here.
4. Create an Anaconda environment using the provided yml file (this downloads many packages and can take a while):
   - Windows (use Anaconda Prompt): `conda env create -f Environments/environment_windows.yml`
   - Mac / Linux (use Terminal): `conda activate` and `conda env create -f Environments/environment_linux.yml`
5. Activate the environment you just created: `conda activate EasiGP`
6. If you want to use LD pruning: install `plink2` and make sure it is available on your PATH (or point the GUI at its executable path directly).
7. If you want to select the XGBoost or EBM models, make sure the optional `xgboost` / `interpret` packages are installed in your environment - the provided yml files already pin these, but if you're managing your own environment by hand, add them separately. Neither is required for any other model.
8. Optional - only if you want the ASReml-R mixed-model adjustment: install ASReml-R and activate your licence, see [Optional: ASReml-R mixed-model adjustment](#optional-asreml-r-mixed-model-adjustment-needs-a-paid-licence). This needs a paid licence; everything else in EasiGP works without it.

##### Laptop: every time you use EasiGP

Once the installation above is done, this is all you need to do each time:

1. Open Anaconda Prompt (Windows) or Terminal (Mac / Linux).
2. Activate the environment: `conda activate EasiGP`
3. Change directory to the main EasiGP folder: `cd YOUR PATH TO EasiGP/EasiGP`
4. Run: `streamlit run main_app.py --server.port 8501`
   a. If port 8501 is already in use, change it to another free port.
5. Your default browser should open automatically to the GUI.
   - If it doesn't open automatically, copy the local URL printed in the terminal (e.g. `http://localhost:8501`) into your browser.
6. Complete the configuration in the GUI and run the pipeline.
7. When you're done, close the browser tab and press `Ctrl+C` in the terminal to stop EasiGP.

**Updating to a new EasiGP version:** download or `git pull` the new code over your existing folder. If the new version changes the required packages, update the environment from the main EasiGP folder with `conda env update -f Environments/environment_windows.yml --prune` (Windows) or `conda env update -f Environments/environment_linux.yml --prune` (Mac / Linux); otherwise nothing needs reinstalling.

#### For HPC

##### HPC: first-time installation (do once)

Do these steps on the login node, which usually has internet access. (Compute nodes often don't - e.g. on NCI Gadi - and creating the environment needs to download packages. If your HPC does not allow installing software on the login node, do steps 3-5 inside an interactive job on a node that has internet access instead.)

1. Clone the EasiGP folder from GitHub (https://github.com/ShunichiroT/EasiGP), log in to your HPC, and store it there.
2. Convert the format of your genotype/phenotype data into the format EasiGP requires (details: https://github.com/ShunichiroT/EasiGP/tree/main/Data).
3. Activate Anaconda (e.g. `module load anaconda3`) and change directory to the main EasiGP folder: `cd YOUR PATH TO EasiGP/EasiGP`
4. Create an Anaconda environment using the provided yml file: `conda env create -f Environments/environment_linux.yml`
5. Activate the environment you just created: `conda activate EasiGP`
6. If you want to use LD pruning: install `plink2` and make sure it is available on your PATH (or point the GUI at its executable path directly).
   - You can also use a module on your HPC if available (e.g. `module load plink`). In that case you need to load the module again each time you use EasiGP (see below).
7. Optional - only if you want the ASReml-R mixed-model adjustment: install ASReml-R and activate your licence, see [Optional: ASReml-R mixed-model adjustment](#optional-asreml-r-mixed-model-adjustment-needs-a-paid-licence). This needs a paid licence; everything else in EasiGP works without it.

##### HPC: every time you use EasiGP

Once the installation above is done, this is all you need to do each time:

1. Submit a job for interactive mode, e.g. for Slurm:
   ```
   salloc --nodes=NODE_NUMBER; e.g. 1 --ntasks-per-node=TASK_NUMBER; e.g. 1 \
     --cpus-per-task=CPU_NUMBER; e.g. 1 --mem=MEMORY; e.g. 10G \
     --job-name=YOUR JOB NAME --time=ALLOCATED TIME; e.g. 01:00:00 \
     --partition=YOUR PARTITION; e.g. general --account=YOUR GROUP \
     srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l
   ```
2. Once the interactive job starts running, activate Anaconda (e.g. `module load anaconda3`) and then the environment: `conda activate EasiGP`
   - If you use `plink2` from a module, load it too (e.g. `module load plink`).
3. Change directory to the main EasiGP folder: `cd YOUR PATH TO EasiGP/EasiGP`
4. Run: `streamlit run main_app.py --server.port 8501 --server.headless true`
   a. If port 8501 is already in use, change it to another free port.
   b. Note the compute node name shown by the scheduler (e.g. in your shell prompt, or via `squeue --me`) - you'll need it in the next step.
5. From your **local machine**, open a new terminal and set up an SSH tunnel to that compute node:
   ```
   ssh -N -L 8501:COMPUTE_NODE_NAME:8501 YOUR_ACCOUNT@YOUR_HPC_LOGIN_ADDRESS
   ```
   For example, on UQ's Bunya cluster: `ssh -N -L 8501:bun128:8501 USERNAME@bunya.rcc.uq.edu.au`
6. Open the local URL in your browser (e.g. `http://localhost:8501`).
7. Complete the configuration in the GUI, then use the "Generate and save job files" option to create the submission script and its config JSON file. On this screen the GUI also suggests CPU/GPU/memory and array-layout settings for your job based on your configuration - review these before submitting, especially for a first run on new data.
   - The submitted job starts in a fresh shell on a compute node, so it needs to know how to find the environment. Use the "Additional setup lines (optional)" box on this screen to add the lines your job needs, e.g. `module load anaconda3` and `conda activate EasiGP` (plus `module load plink` if you use `plink2` from a module).
8. Log in to the HPC using another terminal (separate from the tunnel in step 5, which must stay open only as long as you're using the GUI).
9. Change directory to the main EasiGP folder: `cd YOUR PATH TO EasiGP/EasiGP`
10. Submit the job file created in step 7 (e.g. `sbatch script name.sh` for Slurm).
11. When you're done with the GUI, close the browser tab, stop the SSH tunnel (`Ctrl+C` in that terminal), and end your interactive job (`exit`, or however your scheduler expects you to release it). A submitted job keeps running without it.

**Updating to a new EasiGP version:** replace the code on the HPC (or `git pull`). If the new version changes the required packages, update the environment from the main EasiGP folder with `conda env update -f Environments/environment_linux.yml --prune`; otherwise nothing needs reinstalling.

#### Optional: ASReml-R mixed-model adjustment (needs a paid licence)

> **ASReml-R is commercial software and needs its own paid licence. EasiGP does not include it and cannot supply a licence.** You need to buy (or have access to) an ASReml-R licence from VSNi *before* you can use this feature. Without a licence, leave the ASReml-R option switched off - every other part of EasiGP works normally.

ASReml-R is only used by the optional "ASReml-R Mixed-Model Adjustment (BLUE/BLUP)" section on the Data Preprocessing tab (Tab 2), which fits a mixed model and uses the resulting BLUE/BLUP values in place of the raw phenotypes. ASReml-R works on plot-level records, so set **Phenotype file layout** to **Raw** on Tab 1 before ticking it. It is not on conda, CRAN or pip, so it is **not** installed by the yml files, and it is **not** included in the ready-made Docker/Apptainer images of Option A (it cannot be redistributed). To use it, set up EasiGP with Option B.

Install it **once**, after the first-time installation above:

1. **Get a licence and the installer.** Obtain an ASReml-R licence and download the ASReml-R package from VSNi (https://vsni.co.uk/software/asreml-r), following VSNi's own instructions. The package must be built for the R version inside EasiGP's environment (R 4.3.x - check with `R --version` after activating the environment) and for your operating system.
2. **Activate the EasiGP environment** (`conda activate EasiGP`) so that you install ASReml-R into *EasiGP's own R*, not into a different R on your system. Installing it into any other R will not work for EasiGP.
3. **Install the package** in that R, following the instructions that come with the download. Typically this means starting `R` in the activated environment and running `install.packages("PATH TO THE DOWNLOADED asreml FILE", repos = NULL)` (for the Windows `.zip` file add `type = "win.binary"`; for the Linux `.tar.gz` file add `type = "source"`). Install any package dependencies that VSNi lists.
4. **Activate your licence** with the activation code VSNi gave you. Either start EasiGP, tick the ASReml-R option on Tab 2 and use the "Activate ASReml-R licence" box that appears when ASReml-R is installed but not yet licensed, or run `asreml::asreml.license.activate()` in R as described by VSNi.
5. **Check it worked.** On Tab 2 the ASReml-R section shows a green status message once ASReml-R is installed and licensed (you can also run `asreml::asreml.license.status()` in R).

Things to know:

- **Activation is per machine.** On an HPC, a licence activated on the login node does not automatically cover the compute nodes your job runs on - activate on the machine that actually runs the job, and check with your HPC support / VSNi how licences are handled on your cluster.
- **EasiGP checks before it starts.** If ASReml-R is switched on for a run but is not installed or not licensed on the machine running it, EasiGP stops at the start with a clear message instead of failing hours into the job.
- **Use the same environment each time.** Whenever you run EasiGP with ASReml-R, activate the same `EasiGP` environment you installed it into (this is already part of the "every time you use EasiGP" steps above).

## User manual

Once EasiGP is installed, the [User Manual](Manual/EasiGP_User_Manual.pdf) (PDF, `Manual/EasiGP_User_Manual.pdf`) is the guide for day-to-day use. It covers:

- which input files you need and how to format them (section 3), and how to choose between Sequential, Parallel and Tool mode (section 4);
- the configuration flow through the seven tabs, with a quick start on the MaizeNAM example data that takes a few minutes on a laptop (section 5);
- every field of each tab, including LD pruning, RF marker filtering, the optional ASReml-R adjustment and the biological prior network (sections 6-12);
- Tool mode for comparing finished runs (section 13), the output files (section 14), and troubleshooting with a configuration checklist (section 15).

The manual summarises installation; the full step-by-step commands are the ones in this README.

## References
Buckler ES, Holland JB, Bradbury PJ, Acharya CB, Brown PJ, Browne C, Ersoz E, Flint-Garcia S, Garcia A, 464 Glaubitz JC et al. 2009. The genetic architecture of maize flowering time. Science. 325:714–718.

Chen Q, Yang CJ, York AM, Xue W, Daskalska LL, DeValk CA, Krueger KW, Lawton SB, Spiegelberg BG, Schnell JM et al. 2019. Teonam: A nested association mapping population for domestication and agronomic trait analysis in maize. Genetics. 213:1065–1078. 

Dominik G. Grimm, Damian Roqueiro, Patrice A. Salomé, Stefan Kleeberger, Bastian Greshake, Wangsheng Zhu, Chang Liu, Christoph Lippert, Oliver Stegle, Bernhard Schölkopf, Detlef Weigel, Karsten M. Borgwardt. 2017. easyGWAS: A Cloud-Based Platform for Comparing the Results of Genome-Wide Association Studies. The Plant Cell. 29. 5-19.

Dong Z, Danilevskaya O, Abadie T, Messina C, Coles N, Cooper M. 2012. A gene regulatory network model for floral transition of the shoot apex in maize and its dynamic modelling. PLoS ONE. 

Fey M, Lenssen JE. 2019. Fast graph representation learning with PyTorch Geometric. arXiv preprint arXiv:1903.02428.

Gibbs, Patrick M., Jefferson F. Paril, and Alexandre Fournier-Level. 2025. Trait genetic architecture and population structure determine model selection for genomic prediction in natural Arabidopsis thaliana populations. Genetics 229.3: iyaf003.

Lundberg SM, Lee SI. 2017. A unified approach to interpreting model predictions. Advances in neural information processing systems. 30.

Mitsanis C, Fortuna N, Beveridge C, Kainer D. 2026. FLASH-P: Turning decades of biology into accurate causal networks with AI agents. bioRxiv. doi: 10.64898/2026.06.13.731799.

Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chillemi G, Antiga L, Desmaison A, Tejani A, Chilamkurthy S et al . 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems. 32.

Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, Blondel M, Prettenhofer P, Weiss R, Dubourg V, Vanderplas J, Passos A, Cournapeau D, Brucher M, Perrot M, Duchesnay E. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830.

Pérez P, de Los Campos G. 2014. Genome-wide regression and prediction with the bglr statistical package. Genetics. 198:483–495. 

The 1001 Genomes Consortium. 2016. 1,135 Genomes Reveal the Global Pattern of Polymorphism in Arabidopsis thaliana. Cell. 166(2). 481-491.

Tomura S, Wilkinson MJ, Cooper M, Powell O. 2025. Improved genomic prediction performance with ensembles of diverse models. G3: Genes, Genomes, Genetics. p. jkaf048. 

Wisser RJ, Fang Z, Holland JB, Teixeira JE, Dougherty J, Weldekidan T, de Leon N, Flint-Garcia S, Lauter N, 583 Murray SC et al. 2019. The genomic basis for short-term evolution of environmental adaptation in 584 maize. Genetics. 213:1479–1494.
