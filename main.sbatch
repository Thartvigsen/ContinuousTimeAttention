#!/bin/bash
#SBATCH -p short
#SBATCH -N 1                      # number of nodes
#SBATCH -n 4                      # number of cores
#SBATCH --mem=20GB                # memory pool for all cores
#SBATCH -t 0-24:00                # time (D-HH:MM)
##SBATCH --exclude=compute-0-01
#SBATCH --gres=gpu:0              # number of GPU
#SBATCH --job-name=main
#SBATCH -o slurm-main-output%a    # STDOUT
#SBATCH -e slurm-main-error%a     # STDERR

##export PYTHONPATH=../pytorchExp/env3/bin/python
##source ../pytorchExp/env3/bin/activate
python main.py --taskid=${SLURM_ARRAY_TASK_ID} #--num_gpu=1
