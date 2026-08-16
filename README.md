# Ensemble AnalySis with Interpretable Genomic Prediction (EasiGP): Computational Tool for Interpreting Ensembles of Genomic Prediction Models
This code is used for "Ensemble AnalySis with Interpretable Genomic Prediction (EasiGP): Computational Tool for Interpreting Ensembles of Genomic Prediction Models". (https://doi.org/10.1002/tpg2.70138)

EasiGP analyses the ensemble of multiple diverse genomic prediction models at the genome level in crop breeding programs.
Circos plots are then constructed using the effect of each genomic marker region and the interactions of genomic markers for a target trait.
With a constructed circos plot, we can visually compare the inferred trait genetic architecture of genomic prediction models to deepen our understanding of their predictive behaviour at the genome level.
The comparison of the inferred genomic marker effects with known key genome regions also enables the discovery of potential new genome regions that have not been well-investigated in previous studies.

## Description
EasiGP is a web-based application (no coding required to use it day-to-day) that runs and compares multiple genomic prediction models, then visualises what each model has learned as a circos plot. It also includes several optional add-ons for preparing your data and for building on top of your results.

- **Models**: twelve individual genomic prediction models, plus an ensemble that combines them, are available to select and run from the GUI.
   - ridge regression best linear unbiased prediction (rrBLUP), genomic best linear unbiased prediction (GBLUP), BayesB and reproducing kernel Hilbert Space (RKHS): BGLR (Pérez and de Los Campos, 2014) in R
   - Random forest (RF), support vector regression (SVR) and K-nearest neighbours (KNN): Scikit-learn (Pedregosa et al., 2012) in Python
   - Multilayer perceptron (MLP): PyTorch (Paszke et al., 2019) in Python
   - Four graph attention network (GAT) variants - GAT infinitesimal, GAT fully-connected, GAT prior-knowledge, and GAT biological prior-knowledge (which learns over genes rather than individual markers, using a curated or FLASH-P-generated gene-interaction network - see "Biological prior network" below) - all built with PyTorch Geometric (Fey et al., 2019) in Python
   - **ensemble**: combines the predictions of every other selected model, optionally with a chosen weighting method rather than a plain average
   - Any model's hyperparameters can either be set manually, or automatically tuned (a choice of search algorithms, e.g. grid, random, or Bayesian search) against a validation set

- **Data preprocessing** (optional, all configured from the GUI before running the models):
   - **LD pruning**: removes markers in high linkage disequilibrium, with an optional diagnostic LD-decay plot
   - **RF marker importance filtering**: narrows the marker set further, down to the top markers by Random Forest importance (by percentage or fixed count) - typically run after LD pruning
   - **Biological prior network**: builds the gene-level interaction network the GAT biological prior-knowledge model needs, either from a gene-interaction network file you already have, or generated automatically by [FLASH-P](https://flash-p.com/) (Mitsanis et al., 2026) (which needs access to Claude - see "Setup procedure" below); gene genomic coordinates can likewise be looked up automatically instead of hand-curated
   - **PLINK support**: genotype data can be supplied either as a CSV file or as a PLINK1 binary fileset (`.bed`/`.bim`/`.fam`), converted automatically

- **Circos plots** compare the inferred trait genetic architecture across models: each model's marker effects are shown as a ring, alongside marker-pair interactions and (optionally) known key gene regions for comparison. Most of a circos plot's own display settings (label size, tick spacing, seam gap, how much to widen a marker/gene region so it's visible, etc.) are suggested automatically based on your data, and can be overridden manually if needed.

- **Two ways to run a pipeline**:
   - **Sequential**: runs every task (population × phenotype × model, etc.) one after another, either directly in the GUI or as a single submitted HPC job
   - **Parallel**: splits the same work into a batch of independent HPC jobs that run at the same time, then assembles their results together afterwards - much faster for a large number of tasks on a cluster. If any individual task fails partway through (e.g. a bad hyperparameter combination), completed tasks are automatically saved as they finish and a resubmitted job picks up only the unfinished ones, rather than starting over.

- **Data**: example data files based on the TeoNAM dataset (Chen et al., 2019), MaizeNAM dataset (Buckler et al., 2009) and Arabidopsis dataset (the 1001 Genomes Consortium, 2016) to run this tool
  - Details are explained in "README.md" in the Data folder

- **Result**: folder used as storage for output files from this tool

- Key files, for anyone working with the code directly (this isn't needed to use the GUI):
   - `environment_windows.yml` / `environment_linux.yml`: the packages needed to build EasiGP's own environment on Windows / Linux
   - `main_app.py`: the GUI application itself (this is what `streamlit run` launches - see "Setup procedure" below)
   - `genomic_prediction.py`: the genomic prediction models themselves
   - `circos_plot.py`: circos plot generation
   - `run_sequential.py` / `run_step1_batch.py` / `run_step2_assemble.py`: the headless scripts an HPC job actually runs, generated for you by the GUI's own "Generate and save job files" option - you shouldn't need to edit these by hand
   - `Preprocess/`: the data preprocessing add-ons described above
   - `models/`: the prediction models and the hyperparameter-tuning engine

## EasiGP setup procedure

EasiGP comes in two forms, and there are two different ways to get either one running. Read the next two sections first to work out which combination is right for you - then jump straight to the matching set of steps.

### Full version or Light version - which do I need?

- **Full version** includes everything: the complete EasiGP toolkit, plus its Claude Code integration used by the "Biological prior network" preprocessing step to auto-generate a gene-interaction network with FLASH-P, instead of requiring one to be supplied manually. To use this you'll need access to Claude (an Anthropic account with API or subscription billing) - see "Getting a Claude Code access token" below.
- **Light version** includes the complete EasiGP toolkit only, with no Claude Code/FLASH-P integration and nothing extra to set up for it. You can still supply your own gene-interaction network file by hand with this version - you just can't have EasiGP generate one for you automatically. Choose this if you don't have (or don't need) Claude access, or want the simpler setup.

Both versions give you the same genomic prediction models, hyperparameter tuning, data preprocessing, and circos plot generation - the only difference is the automatic gene-network generation.

### Ready-made image, or set up from source code - which method do I need?

- **Option A - ready-made image (recommended for most people).** Someone (a colleague, a lab administrator, or you, following the "Building the image files yourself" note below) has already packaged EasiGP and everything it depends on into a single file, ready to run. You don't install Python, conda, or any packages yourself - you just load this file and run it. This is by far the easiest path if you don't have a coding background.
- **Option B - set up manually from the source code.** You clone the EasiGP code from GitHub yourself and build the environment it needs using conda. This gives you direct access to every file so you can inspect or modify the code, but involves more setup steps and some familiarity with the command line.

If you're not sure, use Option A.

---

### Option A: Using a ready-made image (recommended)

A few concepts worth knowing before you start, if you haven't used these before:

- **Docker** and **Apptainer** are both tools that run a pre-packaged, self-contained copy of an application - all its software dependencies already included - without you needing to install any of those dependencies yourself. Docker is normally used on your own laptop/PC; Apptainer (sometimes still called by its older name, Singularity) is the equivalent commonly available on HPC/university computing clusters, since Docker itself usually isn't allowed there for security reasons.
- An **image** is that self-contained, ready-to-run package. A **tar file** (e.g. `EasiGP.tar`) is simply how that image is saved to a single file so it can be copied between computers - similar in spirit to a zip file.
- A **container** is a running copy of an image - i.e. the image is the "template", and the container is EasiGP actually running from it.

#### Getting the image file

EasiGP's ready-made images are published on [Docker Hub](https://hub.docker.com/r/shunichirot/easigp/tags) rather than distributed as downloadable files directly from GitHub - the images are several gigabytes each (over 2GB), too large for GitHub to host directly. That Docker Hub page shows the exact tags available for the Full and Light versions, and the exact `docker pull` command to use for each - you'll need this for the steps below, wherever you see `<v1>` (Full version) or `<v1_light>` (Light version).

- On a **local PC**, using Docker (below), you pull the image directly from that page - no separate download or `.tar` file needed at all.
- On **HPC**, using Apptainer (below), you can often build directly from Docker Hub too, the same way - a `.tar` file is only needed as a fallback, for an HPC whose login/build environment doesn't have internet access, or if you'd simply rather transfer the image over yourself. See that section for exactly how to create one, if you need it.

#### Getting a Claude Code access token (Full version only)

The Full version's Claude Code integration needs to authenticate with Anthropic, using an access token you generate once yourself. Since the computer running EasiGP (especially on HPC) often can't open a web browser to log in, generate this token on a separate machine where you *can* log in with a browser:

1. Install Claude Code there if you haven't already (see [Anthropic's Claude Code documentation](https://docs.claude.com) for installation instructions).
2. Run `claude setup-token` and follow the prompts to log in.
3. Copy the token it prints out - you'll paste this in below, wherever you see `<YOUR TOKEN>`. Keep it private; treat it like a password.

This token works with either an API-billed Anthropic account or a Claude subscription (Pro/Max/Team/Enterprise).

#### For a local PC/laptop, using Docker

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) if you don't already have it, and make sure it's running.
2. Open a terminal (**Command Prompt** or **PowerShell** on Windows; **Terminal** on Mac/Linux).
3. Pull the image directly from Docker Hub - see "Getting the image file" above for the exact tag to use in place of `<TAG>`/`<LIGHT TAG>` below:
   - Full version: `docker pull shunichirot/easigp:<TAG>`
   - Light version: `docker pull shunichirot/easigp:<LIGHT TAG>`

   This downloads several gigabytes, so it can take a while the first time - you only need to repeat it when a new version is released.
4. Run it:
   - Full version:
     ```
     docker run -p 8501:8501 -it -e CLAUDE_CODE_OAUTH_TOKEN=<YOUR TOKEN> shunichirot/easigp:<TAG> streamlit run main_app.py --server.address=0.0.0.0 --server.port=8501
     ```
   - Light version:
     ```
     docker run -p 8501:8501 -it shunichirot/easigp:<LIGHT TAG> streamlit run main_app.py --server.address=0.0.0.0 --server.port=8501
     ```
   (If port 8501 is already in use on your machine, change *both* `8501`s before the colon-separated pair, e.g. `-p 8502:8501`, and use that new port number in the next step instead.)
5. Once you see EasiGP's own startup messages in the terminal, open your browser and go to `http://localhost:8501` (or whichever port you chose in step 4).
6. When you're done, press `Ctrl+C` in the terminal to stop the container.

Any files EasiGP creates (results, generated networks, etc.) live inside the container and are lost when it stops, unless you mount a folder from your own computer into it - add `-v "<PATH ON YOUR COMPUTER>":/workspace/data` to the `docker run` command above (before `shunichirot/easigp:<TAG>`) to keep a folder in sync, e.g. `-v "$(pwd)/data":/workspace/data` on Mac/Linux or `-v "${PWD}\data":/workspace/data` in PowerShell.

#### For HPC, using Apptainer

Apptainer builds a **sandbox** - an extracted, folder-based copy of the image - then runs EasiGP from that sandbox. Do this once; you can reuse the same sandbox for every future session.

1. Log in to your HPC and make sure Apptainer is available (try `apptainer --version`; if that fails, check your HPC's documentation for how to load it, e.g. `module load apptainer`).
2. Build the sandbox - see "Getting the image file" above for the exact tag to use in place of `<TAG>`/`<LIGHT TAG>` below. There are two ways to do this, depending on whether your HPC's login node has outbound internet access (many do; some, especially compute nodes, don't):

   **a. Directly from Docker Hub (simplest - try this first):**
   ```
   mkdir -p /scratch/user/$USER/EasiGP
   apptainer build --sandbox /scratch/user/$USER/EasiGP docker://shunichirot/easigp:<TAG>
   ```
   (Light version: replace both `EasiGP` in the paths with `EasiGP_light`, and `<TAG>` with `<LIGHT TAG>`.) If this works, skip straight to step 3 below.

   **b. Via a `.tar` file (if step a fails, or your login node has no internet access):**
   - On a *different* computer that has both internet access and Docker installed - your own laptop is usually easiest - download and re-package the image as a single file:
     ```
     docker pull shunichirot/easigp:<TAG>
     docker save -o EasiGP.tar shunichirot/easigp:<TAG>
     ```
     (Light version: use `<LIGHT TAG>` and name the file `EasiGP_light.tar` instead.) This file will be several gigabytes - the same size as the image itself.
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
     --partition=<YOUR PARTITION> --account=<YOUR GROUP> \
     srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l
   ```
4. Once your interactive job is running, run the container:
   - Full version - generate a Claude Code access token first (see "Getting a Claude Code access token" above), then:
     ```
     export APPTAINERENV_CLAUDE_CODE_OAUTH_TOKEN="<YOUR TOKEN>"
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
   ssh -N -L 8501:<COMPUTE_NODE_NAME>:8501 <YOUR_ACCOUNT>@<YOUR_HPC_LOGIN_ADDRESS>
   ```
   For example, on UQ's Bunya cluster: `ssh -N -L 8501:bun128:8501 USERNAME@bunya.rcc.uq.edu.au`. Leave this terminal open for as long as you're using EasiGP.
6. Open `http://localhost:8501` in your browser.
7. When you're done, close the browser tab, stop the SSH tunnel (`Ctrl+C` in that terminal), and end your interactive job (`exit`, or however your scheduler expects you to release it).

##### Bunya (UQ) or Gadi (NCI) users: an easier option using a virtual desktop

If you're on UQ's Bunya or NCI's Gadi, there's a simpler alternative to steps 3-7 above: both offer a **virtual desktop** - a full graphical Linux desktop, running on a compute node, that opens directly in your own browser. Since your browser and the compute node running EasiGP end up in the same place, this skips the separate interactive-job and SSH-tunnel steps entirely - you just open a terminal *inside* the virtual desktop for step 2 and 4 above, then open a second browser tab (e.g. Firefox) *inside that same desktop* for step 6, instead of tunnelling back to your own machine.

- **Bunya**: go to [onBunya](https://bunya-ondemand.rcc.uq.edu.au/pun/sys/dashboard) and launch a desktop session from **Interactive Apps** (see UQ RCC's [OnDemand Guide](https://github.com/UQ-RCC/hpc-docs/blob/main/guides/OnDemand-Guide.md) for the full walkthrough). This already *is* your interactive job, so step 3 above isn't needed - open a terminal from within the desktop and continue from step 2.
  - If you've separately set up EasiGP's conda environment on Bunya (Option B below) and added conda's own initialisation to your `.bashrc`, onBunya's own desktop sessions won't start until that's removed (`conda init --reverse`) - see UQ RCC's guide for details. This doesn't apply to the Apptainer route described here on its own.
- **Gadi**: go to the [Australian Research Environment (ARE)](https://are.nci.org.au/) and launch a **VDI** (Virtual Desktop Infrastructure) session (see this [Gadi/ARE platform guide](https://acdguide.github.io/BigData/platforms/platforms-nci-gadi.html) for more background). Open a terminal from within the desktop and continue from step 2.
  - Gadi's compute nodes (including VDI sessions) typically don't have general internet access - this doesn't affect the Light version at all, but may prevent the Full version's Claude Code/FLASH-P (Mitsanis et al., 2026) features from reaching Anthropic's API from inside a VDI session, and may also mean option 2a above (building directly from Docker Hub) doesn't work from a VDI session either, even though it might from the login node - option 2b (via a `.tar` file) always works regardless. Check with NCI if you specifically need internet access from a VDI/compute-node session for this.

#### Building the image files yourself

If you're the one preparing/publishing these images (e.g. you have the EasiGP Dockerfile and want to build and push fresh ones to Docker Hub), see the Dockerfile's own comments for the exact `docker build` commands, including the `--build-arg INSTALL_FLASHP=false` option used to produce the Light version.

---

### Option B: Setting up manually from source code

#### For laptop

1. Clone the EasiGP folder from GitHub (https://github.com/ShunichiroT/EasiGP) and store it locally.
2. Convert the format of your genotype/phenotype data into the format EasiGP requires (details: https://github.com/ShunichiroT/EasiGP/tree/main/Data).
3. Start Anaconda and create an Anaconda environment using the provided yml file:
   - Windows (use Anaconda Prompt): `conda env create -f environment_windows.yml`
   - Mac / Linux (use Terminal): `conda activate` and `conda env create -f environment_linux.yml`
5. Activate the environment you just created: `conda activate EasiGP`
6. Change directory to the main EasiGP folder: `cd <YOUR PATH TO EasiGP>/EasiGP`
7. Run: `streamlit run main_app.py --server.port 8501`
   a. If port 8501 is already in use, change it to another free port.
8. Your default browser should open automatically to the GUI.
   - If it doesn't open automatically, copy the local URL printed in the terminal (e.g. `http://localhost:8501`) into your browser.
9. Complete the configuration in the GUI and run the pipeline.
10. Before using LD pruning: this feature requires `plink2` to be installed and available on your PATH (or point the GUI at its executable path directly).

#### For HPC

1. Clone the EasiGP folder from GitHub (https://github.com/ShunichiroT/EasiGP), log in to your HPC, and store it there.
2. Convert the format of your genotype/phenotype data into the format EasiGP requires (details: https://github.com/ShunichiroT/EasiGP/tree/main/Data).
3. Submit a job for interactive mode, e.g. for Slurm:
   ```
   salloc --nodes=<NODE_NUMBER; e.g. 1> --ntasks-per-node=<TASK_NUMBER; e.g. 1> \
     --cpus-per-task=<CPU_NUMBER; e.g. 1> --mem=<MEMORY; e.g. 10G> \
     --job-name=<YOUR JOB NAME> --time=<ALLOCATED TIME; e.g. 01:00:00> \
     --partition=<YOUR PARTITION; e.g. general> --account=<YOUR GROUP> \
     srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l
   ```
4. Once the interactive job starts running, activate Anaconda (e.g. `module load anaconda3`).
5. Create an Anaconda environment using the provided yml file: `conda env create -f environment_linux.yml`.
6. Activate the environment you just created if it is not: `conda activate EasiGP`
7. Change directory to the main EasiGP folder: `cd <YOUR PATH TO EasiGP>/EasiGP`
8. Run: `streamlit run main_app.py --server.port 8501 --server.headless true`
   a. If port 8501 is already in use, change it to another free port.
   b. Note the compute node name shown by the scheduler (e.g. in your shell prompt, or via `squeue --me`) - you'll need it in the next step.
9. From your **local machine**, open a new terminal and set up an SSH tunnel to that compute node:
   ```
   ssh -N -L 8501:<COMPUTE_NODE_NAME>:8501 <YOUR_ACCOUNT>@<YOUR_HPC_LOGIN_ADDRESS>
   ```
   For example, on UQ's Bunya cluster: `ssh -N -L 8501:bun128:8501 USERNAME@bunya.rcc.uq.edu.au`
10. Open the local URL in your browser (e.g. `http://localhost:8501`).
11. Complete the configuration in the GUI, then use the "Generate and save job files" option to create the submission script and its config JSON file.
12. Log in to the HPC using another terminal (separate from the tunnel in step 9, which must stay open only as long as you're using the GUI).
13. Change directory to the main EasiGP folder: `cd <YOUR PATH TO EasiGP>/EasiGP`
14. Submit the job file created in step 11 (e.g. `sbatch <script name>.sh` for Slurm).
15. Before using LD pruning: this feature requires `plink2` to be installed and available on your PATH (or point the GUI at its executable path directly).
    - Yo can also activate your module on HPC if available (e.g. `module load plink`)

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
