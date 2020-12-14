#!/bin/bash
################################
# Script for running multiple experiments by submitting multiple slurm jobs
# How to use this script?
#  ./main.sh [taskid range]
# For example, if you want to run task id 1 to id 5 (5 tasks in parallel), you could type:
# ./main.sh 1-5
################################
# sbatch --array=$1 main.sbatch

#srun --array=$1 main.sbatch

# TURING
export PYTHONPATH=/home/twhartvigsen/work/env3/bin/python3
source /home/twhartvigsen/work/env3/bin/activate
sbatch --array=$1 main.sbatch

# LOCAL
#source /home/tom/Documents/env/bin/activate
#python main.py --taskid=$1

# DMKD
#source /home/tom/env/bin/activate
#for i in $(seq $1 $2);
#  do
#    echo "Starting task $i"
#    python main.py --taskid=$i
#  done

################################
# How to use more GPUs?
# change in file: main.sbatch
################################

