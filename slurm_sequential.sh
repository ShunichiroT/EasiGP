#!/bin/bash --login
# EasiGP - sequential job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --job-name=EasiGP_MaizeNAM_sequential
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH -o C:\Users\uqstomur\OneDrive - The University of Queensland\Documents\Scripts\EasiGP_ver1\Result\MaizeNAM\logs\EasiGP_MaizeNAM_sequential.output
#SBATCH -e C:\Users\uqstomur\OneDrive - The University of Queensland\Documents\Scripts\EasiGP_ver1\Result\MaizeNAM\logs\EasiGP_MaizeNAM_sequential.error

# Submit with:  sbatch EasiGP_MaizeNAM_sequential.sh
# Runs this single job via run_sequential.py - no GUI and no manual
# configuration is needed beyond this one-time export.

AAAA
BBBB
python run_sequential.py --config "C:\Users\uqstomur\OneDrive - The University of Queensland\Documents\Scripts\EasiGP_ver1\Result\MaizeNAM\sequential_config.json"
