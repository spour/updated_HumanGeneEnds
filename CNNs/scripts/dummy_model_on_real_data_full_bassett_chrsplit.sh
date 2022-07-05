#!/bin/bash
#SBATCH --mem-per-cpu=5G
#SBATCH --time=4:00:00
#SBATCH --job-name=bassett_testing_48h_5g
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:1       # request GPU "generic resource"
#SBATCH --cpus-per-task=2
#SBATCH --account=def-thughes

source /scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/scoredd_allrbps/cnn_032022/bin/activate

python /scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/dummy_model_on_real_data_full_bassett_chrsplit.py -pte $1 -ptr $2 -nte $3 -ntr $4 -outdir $5 -outname $6
