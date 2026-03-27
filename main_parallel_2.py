from assemble import *
from metric_plot import *
from scatter_plot import *
from circos_plot import *
  

# Change below
### ======================================================================= ###

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

# Folder name that stores prediction results (inside Result folder) 
RESULT_NAME = 'MaizeNAM'

# ---------------------------------------------------------------------------- #
# 3. Circos plot configuration

# Prediction scenario (within population prediction or between population scenario)
# Specify with either 'within' or 'between'
SCENARIO = 'within'

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
# interaction_top: select only the top N% of strongest links
# label_size:      size of font
# scale:           scale of the circos plot  
CIRCOS_CONFIG = {'space':3,
                 'start':15,
                 'end':345,
                 'link_width':10,
                 'interaction_top':0.001,
                 'label_size':6,
                 'scale':100}

# adjust the edge location of each marker for visualisation
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
       "red10": "#610901",
       "transduction":"#02b01c",
       "transduction_clock":"#0afcd0",
       "clock": "#0389ad",
       "photoperiod":"#947b01",
       "autonomous":"#ffbc03",
       "integrator":"#91029c",
       "integrator_clock":"#f990fc",
       "GA":"#a990fc",
       "aging":"#90a7fc",
       "centromere": "#333333"
   }

### ======================================================================= ###

# Assemble files
metrics, predicted_result_train, predicted_result_test, effect, interactions, \
    POPULATION, PHENOTYPE, MODEL = assemble(RESULT_NAME)

# Store violin plots
metric_plot(metrics.copy(), MODEL, RESULT_NAME)

if 'GAT_fully_connected' in MODEL or 'GAT_prior_knowledge' in metrics['model'].unique().tolist():
    attention_distribution(attention, RESULT_NAME, 10)

# Generate scatter plot matrices if needed
if SCATTER_CREATE:
    scatter_plot(MODEL, PHENOTYPE, predicted_result_test, effect, QTL, SCATTER_CONFIG, RESULT_NAME)

# Generate circos plots
circos_plot(effect, interactions, MARKER_INFO, CHROMOSOME_INFO, GENE_INFO, 
            POPULATION, PHENOTYPE, CIRCOS_CONFIG, END_ADJUST, WINDOW, CYTOBAND_COLORMAP, RESULT_NAME, SCENARIO) 
