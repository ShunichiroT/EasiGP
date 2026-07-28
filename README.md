# Ensemble AnalySis with Interpretable Genomic Prediction (EasiGP): Computational Tool for Interpreting Ensembles of Genomic Prediction Models
This code is used for "Ensemble AnalySis with Interpretable Genomic Prediction (EasiGP): Computational Tool for Interpreting Ensembles of Genomic Prediction Models". (https://doi.org/10.1002/tpg2.70138)

EasiGP analyses the ensemble of multiple diverse genomic prediction models at the genome level in crop breeding programs.
Circos plots are then constructed using the effect of each genomic marker region and the interactions of genomic markers for a target trait.
With a constructed circos plot, we can visually compare the inferred trait genetic architecture of genomic prediction models to deepen our understanding of their predictive behaviour at the genome level.
The comparison of the inferred genomic marker effects with known key genome regions also enables the discovery of potential new genome regions that have not been well-investigated in previous studies.

## Description
- Model: the code for seven individual genomic prediction models (rrBLUP, BayesB, RKHS, RF, SVR, MLP and GAT) and the naive ensemble-average models is stored. These genomic prediction models are implemented through the "main" function.
   - ridge regression best linear unbiased prediction (rrBLUP), genomic best linear unbiased prediction (GBLUP), BayesB and reproducing kernel Hilbert Space (RKHS): BGLR (Pérez and de Los Campos, 2014) in R
   - Random forest (RF) and support vector regression (SVR): Sklearn (Pedregosa et al., 2012) in Python
   - Multilayer perceptron (MLP): PyTorch (Paszke et al., 2019) in Python
   - Graph attention network (GAT), GAT infinitesimal, GAT fully-connected & GAT prior-knowledge: PyTorch Geometric (Fey et al., 2019) in Python 
  
- Data: example data files based on the TeoNAM dataset (Chen et al., 2019), MaizeNAM dataset (Buckler et al., 2009) and Arabidopsis dataset (the 1001 Genomes Consortium, 2016) to run this tool
  - Details are explained in "README.md" in the Data folder

- Result: folder used as storage for output files from this tool

- environment_windows.yml: a list of packages needed to implement this tool in Windows

- environment_linux.yml: a list of packages needed to implement this tool in Linux

- genomic_prediction.py: code that bundles the genomic prediction models. This function is implemented through the "main" function.

- circos_plot.py: code that generates a circos plot. This function is implemented through the "main" function.

- main.py: the top function that manages the implementation of this tool. Users can modify the settings and hyperparameters to customise this tool based on their requirements.

- main_parallel.py: the top function that manages the implementation of this tool. Users can modify the settings and hyperparameters to customise this tool based on their requirements. Multiple prediction tasks are processed in parallel using HPC or a distributed computing system


## EasiGP setup procedure

### For laptop

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

### For HPC

1. Clone the EasiGP folder from GitHub (https://github.com/ShunichiroT/EasiGP), log in to your HPC, and store it there.
2. Convert the format of your genotype/phenotype data into the format EasiGP requires (details: https://github.com/ShunichiroT/EasiGP/tree/main/Data).
3. Activate Anaconda (e.g. `module load anaconda3`).
4. Create an Anaconda environment using the provided yml file: `conda env create -f environment_linux.yml`
5. Activate the environment you just created: `conda activate EasiGP`
6. Change directory to the main EasiGP folder: `cd <YOUR PATH TO EasiGP>/EasiGP`
7. Submit a job for interactive mode, e.g. for Slurm:
   ```
   salloc --nodes=<NODE_NUMBER; e.g. 1> --ntasks-per-node=<TASK_NUMBER; e.g. 1> \
     --cpus-per-task=<CPU_NUMBER; e.g. 1> --mem=<MEMORY; e.g. 10G> \
     --job-name=<YOUR JOB NAME> --time=<ALLOCATED TIME; e.g. 01:00:00> \
     --partition=<YOUR PARTITION; e.g. general> --account=<YOUR GROUP> \
     srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l
   ```
8. Once the interactive job starts running, run: `streamlit run main_app.py --server.port 8501 --server.headless true`
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

Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chillemi G, Antiga L, Desmaison A, Tejani A, Chilamkurthy S et al . 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems. 32.

Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, Blondel M, Prettenhofer P, Weiss R, Dubourg V, Vanderplas J, Passos A, Cournapeau D, Brucher M, Perrot M, Duchesnay E. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830.

Pérez P, de Los Campos G. 2014. Genome-wide regression and prediction with the bglr statistical package. Genetics. 198:483–495. 

The 1001 Genomes Consortium. 2016. 1,135 Genomes Reveal the Global Pattern of Polymorphism in Arabidopsis thaliana. Cell. 166(2). 481-491.

Tomura S, Wilkinson MJ, Cooper M, Powell O. 2025. Improved genomic prediction performance with ensembles of diverse models. G3: Genes, Genomes, Genetics. p. jkaf048. 

Wisser RJ, Fang Z, Holland JB, Teixeira JE, Dougherty J, Weldekidan T, de Leon N, Flint-Garcia S, Lauter N, 583 Murray SC et al. 2019. The genomic basis for short-term evolution of environmental adaptation in 584 maize. Genetics. 213:1479–1494.
