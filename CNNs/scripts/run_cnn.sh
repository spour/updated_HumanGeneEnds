#!/bin/bash
#SBATCH --account=def-thughes
#SBATCH --time=36:00:00
#SBATCH --job-name=bassett_testing_48h_5g
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10G        # memory per node



#module load gcc/9.3.0 arrow python scipy-stack
#source /scratch/spour98/scoring_aleksei_15112021/cnns/cnn_venv/bin/activate
source /scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/CNN_baseline/cnn_27052022/bin/activate
python /scratch/spour98/scoring_aleksei_15112021/redo_04012022/run_cnn_trash.py -f $1 -m $2
#(cnn_venv) [spour98@cedar1 whole_gene_fa]$  for file in /scratch/spour98/scoring_aleksei_15112021/fastas/whole_gene_fa/*noends.wholegenes*.fa; do sbatch /scratch/spour98/scoring_aleksei_15112021/run_cnn.sh $file /scratch/spour98/scoring_aleksei_15112021/cnns/posPAS_100000NEG_resnet_20211116142636.tf; done
