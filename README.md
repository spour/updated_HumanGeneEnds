# humangene_ends

The scripts and files in this repository allow you to locally train the baseline and cryptic models described in : Shkurin, A, Pour, S.E., and Hughes, TR, "Known sequence features can explain half of all human gene ends". 
NOTE: for issues/questions, please reach out to sara.eslampour@mail.utoronto.ca, or t.hughes@utoronto.ca. Or open an issue on github.

## Prerequisites
Bio==1.5.3
joblib==1.2.0+computecanada
matplotlib==3.4.2+computecanada
numba==0.56.4
numpy==1.20.2+computecanada
pandas==0.25.3+computecanada
regex==2022.10.31
scikit_learn==1.2.1
scipy==1.4.1+computecanada
skimage==0.0


## Baseline Model
This script is used to generate the baseline feature matrices for a given fasta file using the position weight matrices (PWMs) located in "https://github.com/spour/updated_HumanGeneEnds/tree/main/BaselineModelPWMs".

