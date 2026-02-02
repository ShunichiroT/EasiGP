import os
from genomic_prediction import *
  

# Change below
### ======================================================================= ###
# 1. Genomic prediction configuration

# Assign path for R if causing an error
# e.g.'C:\Program Files\R\R-4.4.0'
R_PATH = None

# Your target phenotypes
# Write 'all' to select all phenotypes in your phenotype file  
PHENOTYPE = ['days2anthesis', 'asi'] 

# Name of genomic prediction models to run
# Available models
  # ['rrBLUP', 'GBLUP', 'BayesB', 'RKHS', 'RF', 'SVR', 'MLP', 'ensemble']
  # ['GAT_infinitesimal', 'GAT_fully_connected', 'GAT_prior_knowledge']
MODEL = ['rrBLUP', 'BayesB', 'RF', 'ensemble'] 

# Data splitting ratio
# If elements are in float values, they are used as a training set ratio when 
# splitting the data into training and test sets
    # e.g. [0.8, 0.65, 0.5]

# If elements are in tuple formats, they are used to split the data into 
# training, validation and test sets

# The data needs to be split into training, validation and test sets for 
# weight optimisation
    # e.g. [(0.5,0.25,0.25),(0.8,0.1,0.1)] 
# The element of each tuple shows the ratio of the training, 
# validation and test set, respectively

RATIO = [0.8]  #[(0.5,0.25,0.25)]     

# Number of iterations with random sampling to create training & test sets in each population-phenotype-ratio prediction combination
ITER_NUM = 2

# Folder name that stores prediction results (inside the Result folder) 
RESULT_NAME = 'MaizeNAM'

if not os.path.exists('./Result/'+RESULT_NAME):
    os.makedirs('./Result/'+RESULT_NAME)

# File paths for your genotype & phenotype files
GENOTYPE_FILE_NAME = './Data/MaizeNAM/MaizeNAM_dataset_genotype_population_1.csv' 
PHENOTYPE_FILE_NAME = './Data/MaizeNAM/MaizeNAM_dataset_phenotype_population_1.csv' 

# Configuration for the parallel run of prediction scenarios
# Prediction scenarios (population*trait*ratio*training-test set iteration numbers) are split into k batches in parallel
# Batch_id: assign the id of the batch you want to run in parallel
    # It is possible to automate id assignment using a job array in HPC if any
    # e.g. int(os.environ["SLURM_ARRAY_TASK_ID"])
# Batch size: the total number of prediction scenarios in each parallel batch
PARALLEL = {'batch_id': CHANGE HERE,
            'batch_size': 2}

# Hyperparameters
# rrBLUP:                          
    # iteration number, burin-in
# GBLUP:                            
    # iteration number, buin-in, number of samples for Shapley scores, 
    # return marker effect?
# BayesB:                          
    # iteration number, burin-in
# RKHS:                            
    # iteration number, buin-in, number of samples for Shapley scores, 
    # return marker effect?
# RF:                              
    # tree number, maximum features per tree, maximum samples per tree, 
    # number of samples for Shapley scores, return marker effect interactions?
# SVR:                             
    # kernel type, epsilon, regularisation, dimension for poly kernel, gamma, 
    # number of samples for Shapley scores, return marker effect?
# MLP:                             
    # neuron numbers, dropout, learning rate, decay, epoch, training batch size, 
    # number of samples for Shapley scores
# GAT_infinitesimal_node_level:    
    # neuron numbers, dropout, learning rate, decay, epoch, training batch size, 
    # number of heads, number of samples for Shapley scores, return marker effect?
# GAT_infinitesimal:               
    # neuron numbers, dropout, learning rate, decay, epoch, training batch size, 
    # number of heads, number of samples for Shapley scores, return marker effect?
# GAT_fully_connected:             
    # neuron numbers, dropout, learning rate, decay, epoch, training batch size, 
    # number of heads, number of samples for Shapley scores, return marker effect?
# GAT_prior_knowledge:             
    # neuron numbers, dropout, learning rate, decay, epoch, training batch size, 
    # number of heads, number of samples for Shapley scores, 
    # selection rate for edges from RF (e.g. 0.1 = selection the top 10% of the 
    # most important edges), return marker effect?
HPARAMETERS = {'rrBLUP': [10000, 2000],
               'GBLUP': [12000, 2000, 30, False], 
               'BayesB': [12000, 2000],    
               'RKHS': [12000, 2000, 30, False],   
               'RF': [1000, 1.0, None, 30, False],           
               'SVR':['rbf', 0.5, 1.0, 3, 'scale', 30, False],      
               'MLP':[30, 0, 0.0001, 5e-4, 200, 8, 10],
               'GAT_infinitesimal_node_level':[20, 0.9, 0.005, 5e-4, 1, 8, 1, 3, False],
               'GAT_infinitesimal':[20, 0, 0.01, 5e-4, 40, 8, 1, 1, True],
               'GAT_fully_connected':[20, 0, 0.01, 5e-4, 40, 8, 1, 1, True],
               'GAT_prior_knowledge':[20, 0, 0.01, 5e-4, 1, 8, 1, 1, 0.2, True]} 

# Method names for weight optimisation in ensembles 
# The current available methods 
# ['Nelder_Mead', 'Linear transformation', 'Bayesian_optimisation']

# Write "None" if you implement naive ensemble approach 
W_OPT = ['Nelder Mead', 'Bayesian optimisation', 'Linear transformation'] 

# Hyperparameters for weight optimisation
# Linear transformation:
    # learning rate, epochs, decay, batch size, patience value, number of MLP models  
# Nelder_Mead:
    # initial value, minimum boundary, maximum boundary, fatol, xatol, adaptive
# Bayesian:
    # minimum boundary, maximum boundary, iterations, point numbers, allow duplicate points
HYPERPARAMETERS_OPT = {'Linear transformation':[0.005, 150, 0.01, 2, 10, 30],
                       'Nelder Mead': [5, 0.001, 10, 1e-8, 1e-8, False],
                       'Bayesian optimisation': [0.0001, 10.0, 200, 1, True]}

### ======================================================================= ###

# Run genomic prediction models
metrics, predicted_result_train, predicted_result_test, effect, interactions, \
    POPULATION, PHENOTYPE = GP(GENOTYPE_FILE_NAME, PHENOTYPE_FILE_NAME, MODEL, 
                               PHENOTYPE, RATIO, ITER_NUM, HPARAMETERS, R_PATH, W_OPT, RESULT_NAME, HYPERPARAMETERS_OPT, PARALLEL)
