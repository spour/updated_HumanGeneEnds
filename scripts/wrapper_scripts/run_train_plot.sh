#!/bin/bash
#SBATCH --account=def-thughes
#SBATCH --mem-per-cpu=40G
#SBATCH --time=55:00:00
#SBATCH --cpus-per-task=5
#SBATCH --job-name=rf_04012022
#SBATCH --output=%x-%j.out

source /scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/scoring_venv/bin/activate
#module load gcc/9.3.0 arrow python scipy-stack
python /scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/train_plot_new.py -ptr positives_training -ntr negatives_training -pte positives_testing -nte negatives_testing

