#!/bin/bash
#SBATCH -c 8
#SBATCH -t 8:00:00
#SBATCH --mem=100G
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1

module load gcc/6.2.0 python/3.7.4
cd /n/scratch3/users/k/kaw308
source jupytervenv/bin/activate
module load gcc/6.2.0
module load cuda/11.2
cd /n/scratch3/users/k/kaw308/viewmaker
source init_env.sh

# srun python scripts/run_ecg_1d.py config/ecg/pretrain_viewmaker_ptb_xl_1d_simclr_e_2.json --gpu-device 0
srun python scripts/run_ecg.py config/ecg/transfer_viewmaker_ptb_xl_simclr_e_2.0.json --gpu-device 0
