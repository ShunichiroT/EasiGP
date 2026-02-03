import os

from genomic_prediction import *
from metric_plot import *
from scatter_plot import *
from circos_plot import *

# Change below
### ======================================================================= ###
# 1. Genomic prediction configuration

# Assign path for R if causing an error
# e.g.'C:\Program Files\R\R-4.4.0'
R_PATH = None

# Your target phenotypes
# Write 'all' to select all phenotypes in your phenotype file  
PHENOTYPE = ['days2anthesis'] 

# Name of genomic prediction models to run
# Available models
  # ['rrBLUP', 'GBLUP', 'BayesB', 'RKHS', 'RF', 'SVR', 'MLP', 'ensemble']
  # ['GAT_infinitesimal', 'GAT_fully_connected', 'GAT_prior_knowledge']
MODEL = ['rrBLUP', 'BayesB', 'RF', 'ensemble'] 

# Data splitting ratio
# If elements are in float values, they are used as the training set ratio when 
# splitting the data into training and test sets
    # e.g. [0.8, 0.65, 0.5]

# If elements are in tuple formats, they are used to split the data into 
# training, validation and test sets

# The data needs to be split into training, validation and test sets for 
# weight optimisation
    # e.g. [(0.5,0.25,0.25),(0.8,0.1,0.1)] 
# The element of each tuple shows the ratio of the training, 
# validation and test set, respectively

RATIO = [0.8]#[(0.5,0.25,0.25)]  #[(0.5,0.25,0.25)]     #[0.8]     

# Number of iterations with random sampling for training & test sets
ITER_NUM = 2

# Folder name that stores prediction results (inside the Result folder) 
RESULT_NAME = 'MaizeNAM'

if not os.path.exists('./Result/'+RESULT_NAME):
    os.makedirs('./Result/'+RESULT_NAME)

# File paths for your genotype & phenotype files
GENOTYPE_FILE_NAME = './Data/MaizeNAM/MaizeNAM_dataset_genotype_population_1.csv' 
PHENOTYPE_FILE_NAME = './Data/MaizeNAM/MaizeNAM_dataset_phenotype_population_1.csv' 

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
               'RF': [1000, 1.0, None, 30, True],           
               'SVR':['rbf', 0.5, 1.0, 3, 'scale', 30, False],      
               'MLP':[30, 0, 0.0001, 5e-4, 200, 8, 10],
               'GAT_infinitesimal_node_level':[20, 0.9, 0.005, 5e-4, 1, 8, 1, 3, False],
               'GAT_infinitesimal':[20, 0, 0.01, 5e-4, 40, 8, 1, 1, True],
               'GAT_fully_connected':[20, 0, 0.01, 5e-4, 40, 8, 1, 1, True],
               'GAT_prior_knowledge':[20, 0, 0.01, 5e-4, 1, 8, 1, 1, 0.2, True]} 

# Method names for weight optimisation in ensembles 
# The current available methods 
# ['Nelder_Mead', 'Linear transformation', 'Bayesian_optimisation']

# Write "None" if you implement the naive ensemble approach 
W_OPT = None #['Nelder Mead', 'Bayesian optimisation', 'Linear transformation'] 

# Hyperparameters for weight optimisation
# Linear transformation:
    # learning rate, epochs, decay, batch size, patience value, number of MLP models  
# Nelder_Mead:
    # initial value, minimum boundary, maximum boundary, fatol, xatol, adaptive
# Bayesian:
    # minimum boundary, maximum boundary, iterations, point numbers, allow duplicate points
HYPERPARAMETERS_OPT = {'Linear transformation':[0.005, 150, 0.01, 2, 10, 30],
                       'Nelder Mead': [5, 0, 10, 1e-8, 1e-8, False],
                       'Bayesian optimisation': [0.0001, 10.0, 200, 1, True]}

# ---------------------------------------------------------------------------- #
# 2.Scatter plot matrix configuration

# True if scatter plot is needed. False otherwise
SCATTER_CREATE = True 

# Write a file path if you want to add QTL information. None otherwise
QTL = None 

# Parameters
# font_size:  font size for scatter plot matrix
# fig_size:   size of scatter plot matrix
SCATTER_CONFIG = {'font_size':2,
                  'fig_size':30}

# ---------------------------------------------------------------------------- #
# 3. Circos plot configuration

# File path for the information of chromosomes
CHROMOSOME_INFO = './Data/MaizeNAM/chrom.csv'

# File path for the information of all markers
MARKER_INFO = './Data/MaizeNAM/marker_info.csv'

# File path for key gene region information for comparison
# Write 'None' if no key gene region information needs to be included
GENE_INFO = './Data/MaizeNAM/gene_info.csv'

# Configuration for circos plots
# space:           space size between rings
# start:           start angle of a ring
# end:             end angle of a ring
# link_width:      the thickness of links
# interaction_top: select only the top N% of strongest links in the ratio form
# label_size:      size of font
# scale:           scale of the circos plot  
CIRCOS_CONFIG = {'space':3,
                 'start':15,
                 'end':345,
                 'link_width':1,
                 'interaction_top':0.01,
                 'label_size':6,
                 'scale':100}

# Adjust the edge location of each marker for visualisation
END_ADJUST = 0

# Choose a method for aggregating genomic marker effects
# Assign 0 if you do not wish to introduce a window to average the effects in each window interval
# Otherwise, assign a window size here
# If the circos plot does not show with WINDOWS > 0, you can increase the size of the window
WINDOW = 30

# colour palette for circos plot
CYTOBAND_COLORMAP = {   
       "gpos100": "#000000", # 0,0,0
       "gpos": "#000000",    # 0,0,0
       "gpos75": "#828282",  # 130,130,130
       "gpos66": "#A0A0A0",  # 160,160,160
       "gpos50": "#C8C8C8",  # 200,200,200
       "gpos33": "#D2D2D2",  # 210,210,210
       "gpos25": "#C8C8C8",  # 200,200,200
       "gvar": "#DCDCDC",    # 220,220,220
       "gneg": "#FFFFFF",    # 255,255,255
       "acen": "#D92F27",    # 217,47,39
       "stalk": "#647FA4",   # 100,127,164
       "green": "#47c462",
       "brown": "#e0a22f",
       "purple": "#a62bcc",
       "blue1": "#def2ff",
       "blue2": "#c2e5fc",
       "blue3": "#addeff",
       "blue4": "#99d6ff",
       "blue5": "#83ccfc",
       "blue6": "#68c1fc",
       "blue7": "#45b5ff",
       "blue8": "#14a0fc",
       "blue9": "#027ac9",
       "blue10": "#014f82",
       "red1": "#fce1a7",
       "red2": "#ffd780",
       "red3": "#ffc954",
       "red4": "#fcba2b",
       "red5": "#ffaf03",
       "red6": "#d99502",
       "red7": "#b57c02",
       "red8": "#8a5e01",
       "red9": "#874001",
       "red10": "#610901"
   }

### ======================================================================= ###

# Run genomic prediction models
metrics, predicted_result_train, predicted_result_test, effect, interactions, \
    POPULATION, PHENOTYPE = GP(GENOTYPE_FILE_NAME, PHENOTYPE_FILE_NAME, MODEL, 
                               PHENOTYPE, RATIO, ITER_NUM, HPARAMETERS, R_PATH, W_OPT, RESULT_NAME, HYPERPARAMETERS_OPT)

# Store violin plots
if W_OPT is not None:
    metric_plot(metrics.copy(), MODEL+W_OPT, RESULT_NAME)
else:
    metric_plot(metrics.copy(), MODEL, RESULT_NAME)

# Generate scatter plot matrices if needed
if SCATTER_CREATE:
    scatter_plot(MODEL, PHENOTYPE, predicted_result_test, effect, QTL, SCATTER_CONFIG, RESULT_NAME)

# Generate circos plots
circos_plot(effect, interactions, MARKER_INFO, CHROMOSOME_INFO, GENE_INFO, 
            POPULATION, PHENOTYPE, CIRCOS_CONFIG, END_ADJUST, WINDOW, CYTOBAND_COLORMAP, RESULT_NAME) 
