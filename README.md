# HumanGeneEnds
The scripts and files in this repository allow you to locally train the baseline and cryptic models described in the __*correction*__ to : Shkurin, A and Hughes, TR, "Known sequence features can explain half of all human gene ends". I also provide some of the underlying data behind the results in the paper. If you have any issues or questions please reach out at sara.eslampour@mail.utoronto.ca

Below are the necessary steps to train each of the models


# Baseline Model
## Data Preparation

In the scripts directory, run create_baseline_features_23112021.py. There are 4 bed files to be processed: Dominant_CPA_baseline_testing.bed, Dominant_CPA_baseline_training.bed, Negatives_CPA_baseline_testing.bed, Negatives_CPA_baseline_training.bed. These should be turned into FASTA files using bedtools and then you can create the features based on the PWM scores in sliding windows. To do this, you can use the PWMs shown in PWMs/BaselineModelPWMs. The script takes in an argument that is a file of the list of PWMs (and their paths) that you are interested in scoring. BaselineModelPWMs contains one such file "pwms_all.txt", please update with the correct paths to the PWM files.

Given that you may want to submit this to a computing cluster, a cognate wrapper script is provided in the subdirectory "wrapper scripts". 

An example script is shown below:

```
bedtools getfasta -fi ../../utils/hg38.fa -bed bed -s -name -fo fas
python /scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/create_baseline_features_23112021.py -f fas -pwms /scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/CNN_baseline/disease/pwms_all.txt -out ./xx
```
