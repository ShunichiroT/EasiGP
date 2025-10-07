## Description

This folder contains two folders containing example files to implement both genomic prediction and circos plot generation functions.

* TeoNAM folder contains adjusted example data from the TeoNAM dataset (Chen et al.,2019), accompanied by key genomic marker information for flowering (Dong et al., 2012; Wisser et al., 2019)
* Arabidopsis folder contains adjusted example data from the 1001 Genomes Consortium (2016), Grimm et al. (2017) and Gibbs et al. (2025), accompanied by key genomic marker information for flowering and branching (Arabidopsis Information Resource (TAIR); https://www.arabidopsis.org/)
* MaizeNAM folder contains adjusted example data from the MaizeNAM dataset (Buckler et al.,2009), accompanied by key genomic marker information for flowering (Dong et al., 2012; Wisser et al., 2019)



## Data structure

1. Genotype data

* Columns:

  * ID: identification code for each individual
  * population: the name of the affiliated population
  * genomic markers: SNP information in numerical format (0,1 or 2)

* Rows: records of each individual



2\. Phenotype data

* Columns:

  * ID: identification code for each individual
  * population: the name of the affiliated population
  * phenotype: recorded phenotypes of each individual

* Rows: records of each individual



3\. Chromosome data (chrom.csv)

* Columns:

  * chromosome: chromosome number ("chr"+NUMBER)
  * start: the beginning location of the chromosome (the value should be 0 for the standard use)
  * end: the end location of the chromosome
  * population: target population (write "all" for across all populations)

* Rows:

  * Each chromosome



4\. Key gene region data (gene\_info.csv)

* Columns:

  * chromosome: belonging chromosome number ('chr'+number)
  * start: the beginning location of the marker
  * end: the end location of the marker
  * name: name of each gene
  * colour: allocated colour of each gene
  * source: information source of each gene. Used as the name of each ring section for a circos plot
  * phenotype: target phenotype of each gene
  * population: target population of each gene (write "all" for across all populations)

* Rows: markers



5\. Marker data (marker\_info.csv)

* Columns:
  * chromosome: belonging chromosome number (NUMBER)
  * name: name of each marker
  * start: the beginning location of the marker
  * end: the end location of the marker
* Rows: markers



6\. QTL data (optional. Create this file when QTL information needs to be added on a scatter plot matrix)

* Columns:

  * phenotype: phenotype that QTL affects
  * marker: name of QTL

* Rows: QTL
