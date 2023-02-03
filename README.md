# Human Gene Ends

The scripts and files in this repository allow you to locally train the baseline and cryptic models described in : Shkurin, A, Pour, S.E., and Hughes, TR, "Known sequence features can explain half of all human gene ends". 
NOTE: for issues/questions, please reach out to sara.eslampour@mail.utoronto.ca, or t.hughes@utoronto.ca. Or open an issue on github. It is also highly recommended you submit to a computing cluster or parallelize the calculations

## Prerequisites
```
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
Python 3 (any version)
```

## Cloning
You can get access to all the files, directories, scripts, and results like this
```
git clone https://github.com/spour/updated_HumanGeneEnds.git
```


## Baseline Model
This script is used to generate the baseline feature matrices for a given fasta file using the position weight matrices (PWMs) located in "BaselineModelPWMs". The baseline model was trained on the following data:
 ```
 negatives_testing.bed as the test set for non-CPA sites
 negatives_training.bed as the train set for non-CPA sites
 Dominant_CPA_baseline_testing.bed as the test set for CPA sites
 Dominant_CPA_baseline_training.bed as the train set for CPA sites
 ```
To make the feature matrices, run the following on the stranded fasta files e.g.
```
bedtools getfasta -fi <path to hg38 fa> -s -bed <file> -fo <out file -name 
```
and then for each file
```
python generate_baseline_features.py -f <file> -pwms <BaselineModelPWMs> -out <out_path>
```
##### Make the model (either LR or RF) from the files above by running:

```
train_models.py -p <positive_training_csv> <negative_training_csv> <positive_testing_csv> <negative_testing_csv> -f [baseline_rf|baseline_lr] -out <out_path> -outp <outfiles_prefix>
```


