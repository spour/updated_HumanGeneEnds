#!/bin/bash
#SBATCH --account=def-thughes
##SBATCH --mem-per-cpu=150G
##SBATCH --time=25:00:00
#SBATCH --time=8:00:00
#SBATCH --mem=45G
#SBATCH --job-name=leftoverbaseline_model_wholegene_14122021_100G_36h
#SBATCH --output=%x-%j.out
##SBATCH --mem-per-cpu=6G
##SBATCH --time=00:30:00
##SBATCH --job-name=leftoverbaseline_model_wholegene_08122021_aleksei_fig3

module load gcc/9.3.0 arrow python scipy-stack
#source /scratch/spour98/scoring_aleksei_15112021/bin/activate
source /scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/scoring_venv/bin/activate
python /scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/rbp_u1_models/ensemble/create_baseline_features_23112021_140_rhybshort.py -f $1 -pwms $2 -out $3
