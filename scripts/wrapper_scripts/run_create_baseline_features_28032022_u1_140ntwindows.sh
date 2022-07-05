#!/bin/bash
#SBATCH --account=def-thughes
#SBATCH --mem-per-cpu=4G
#SBATCH --time=8:00:00
#SBATCH --job-name=U1_cryptic_27012022
#SBATCH --output=%x-%j.out
##SBATCH --mem-per-cpu=6G
##SBATCH --time=00:30:00
##SBATCH --job-name=leftoverbaseline_model_wholegene_08122021_aleksei_fig3

#module load gcc/9.3.0 arrow python scipy-stack
#source /scratch/spour98/scoring_aleksei_15112021/bin/activate
source /scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/scoring_venv/bin/activate
#python /scratch/spour98/scoring_aleksei_15112021/create_baseline_features_29122021_cryptic.py -f $1 -out $2
python /scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/fscore_cryptic/create_baseline_features_28032022_u1_140ntwindows.py -f $1 -pwms $2 -out $3
