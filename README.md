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


## Train and predict using Models
This script is used to generate the baseline feature matrices for a given fasta file using the position weight matrices (PWMs) located in "BaselineModelPWMs". The baseline model was trained on the following data:
 ```
 negatives_testing.bed as the test set for non-CPA sites
 negatives_training.bed as the train set for non-CPA sites
 Dominant_CPA_baseline_testing.bed as the test set for CPA sites
 Dominant_CPA_baseline_training.bed as the train set for CPA sites
 ```
To make the feature matrices, run the following on the stranded fasta files e.g. you need the name column
```
bedtools getfasta -fi <path to hg38 fa> -s -bed <file> -fo <out file -name 
```
and then for each file
```
python generate_baseline_features.py -f <file> -pwms <BaselineModelPWMs> -out <out_path>
```
##### Make the model (either LR or RF) from the files above by running:

N.b. for creating/predicting with cryptic model you will need the libary of RNAhyb 7-mer scores ".../files/7mer_RHYB.npy"

```
train_models.py -p <positive_training_csv> <negative_training_csv> <positive_testing_csv> <negative_testing_csv> -f [baseline_rf|baseline_lr] -out <out_path> -outp <outfiles_prefix>
```

##### Score and predict on genes with the model by running, where the last flag is if you're scoring baseline for windows of 500 or cryptic with windows of 140 as described in paper:
```
python generate_score_cryptic_features.py -f <fasta with seqs of interest> -pwms <path to pwms> -out <outdir and filename> -m <path to model> -ft [cryptic|baseline]
```
Output should be like:
```
0	1
ZNF445::chr3:44441140-44477670(-)_0	0.9135617554490442
ZNF445::chr3:44441140-44477670(-)_1	0.9134519756102718
ZNF445::chr3:44441140-44477670(-)_2	0.9139810592975619
ZNF445::chr3:44441140-44477670(-)_3	0.9139818218498337
ZNF445::chr3:44441140-44477670(-)_4	0.9161833625169807
``` 

## Convert model output to bed
To turn the above output to proper bed files where each window has the model prediction, there are two steps in "misc"
Get all paths for the files that were scored by the model, raw outputs e.g. if the files had suffix ".scored"
```
find "$(pwd -P)" -name  "*scored" > files_to_initial.bed
```
2. Run step 1 
```
python model_scored_to_bed_step1.py -i files_to_initial.bed -o <out_path>
```
3. Run step 2
```
sh model_scored_to_bed_step2.sh <path from step 1> <size of original window scored -1, e.g. 500-1=499> <out_suffix>
```

### Additional files desc.
The maximum scores for longest gene isoforms used in this study are "maximum_score_window_of_baseline.bed"; the cryptic sites found by the baseline RF are "Cryptic_CPA_sites.hg38.bed"; and APA sites from polyADB liftovered to hg38 with +/-250 nt flank are "human.PAS.txt.raw.hg38.bed.250region".

### Additional script descriptions in misc, utils mostly
| File | #Description   |   
| :---:   | :---: | 
| average_score_bed.sh | remove duplicated in bed file and get average score  | 
| :---:   | :---: | 
| filter.sh | go through bed file and write those with score higher than <x> to a collated file  |
| :---:   | :---: | 
| filter_out_apa.sh | remove regions in one file that intersect with another  | 
| :---:   | :---: | 
| merge_adj.sh | sort bed file and collapse regions within 30 nt of each other, retaining information about what was collapsed  | 
| :---:   | :---: | 
| get_max_adjacent_bed.py  | from what was collapsed from merge.sh, pick the window with the maximum score  | 
| :---:   | :---: | 
| intersect.sh  | bedtools intersect with perfect overlap between windows only  | 
| :---:   | :---: | 
| get_max_of_gene.sh  | select maximum scoring window for each gene  | 
 




