#!/bin/bash

#SBATCH --account=ucb510_asc1
#SBATCH --nodes=1
#SBATCH --time=07:00:00
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --job-name=test-octo
#SBATH --mem-per-gpu=45G
#SBATCH --output=test-octo.%j.out

module purge
module load anaconda
module load cuda/12.1.1
module load cudnn
module load aocc
module load gcc
module load intel
module load openblas
conda deactivate
conda activate octo

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/stca9620/software/anaconda/envs/octo/lib/python3.10/site-packages/nvidia/cublas/lib

cd octo
python scripts/finetune.py --config.pretrained_path=hf://rail-berkeley/octo-small-1.5 --debug
#python examples/02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small-1.5 --data_dir=./aloha_sim_dataset --batch_size=32 --save_dir=/home/stca9620/octo/output
