#!/bin/bash
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH --mem=100G
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:3

module load gcc/6.2.0 python/3.7.4
cd ~
source jupytervenv/bin/activate
module load gcc/6.2.0
module load cuda/11.2
cd ~/viewmaker
source init_env.sh

srun python scripts/run_ecg.py config/ecg/pretrain_viewmaker_ptb_xl_simclr_e_0.2.json --gpu-device 0

srun python scripts/run_ecg.py config/ecg/transfer_viewmaker_ptb_xl_simclr_e_0.2.json --gpu-device 0
