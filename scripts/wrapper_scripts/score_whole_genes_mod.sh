#!/bin/bash
#SBATCH --account=def-thughes
##SBATCH --mem-per-cpu=100G
#SBATCH --mem-per-cpu=50G
##USE SBATCH --cpus-per-task=4
## USE SBATCH --time=04:00:00
#SBATCH --time=04:00:00
##SBATCH --job-name=baseline_model_wholegene_06122021_60G
##SBATCH --mem-per-cpu=30G
##SBATCH --time=6:00:00
#SBATCH --job-name=scoring_12012022_out_imbalanced
#SBATCH --output=%x-%j.out

#module load gcc/9.3.0 arrow python scipy-stack


source /scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/scoring_venv/bin/activate
#python /scratch/spour98/scoring_aleksei_15112021/score_whole_genes_mod.py -m $1 -i $2 -o $3
#python /scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/score_whole_genes_mod.py -m $1 -i $2 -o $3
python /scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/rbp_u1_models/ensemble/score_whole_genes_mod.py -m $1 -i $2 -o $3
