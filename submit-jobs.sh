#!/bin/bash
#SBATCH --job-name=sqr
#SBATCH --time=72:00:00
##SBATCH --time=00:15:00
##SBATCH --begin=20:00
#SBATCH --array=0-122
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq

## in the list above, the partition name depends on where you are running your job. 
## On DAS5 the default would be `defq` on Lisa the default would be `gpu` or `gpu_shared`
## Typing `sinfo` on the server command line gives a column called PARTITION.  There, one can find the name of a specific node, the state (down, alloc, idle etc), the availability and how long is the time limit . Ask your supervisor before running jobs on queues you do not know.

# Load GPU drivers
module load julia/1.9.3

# This loads the anaconda virtual environment with our packages
source $HOME/.bashrc
conda activate sqr-noversion

# Simple trick to create a unique directory for each run of the script
# echo $$
# mkdir o`echo $$`
# cd o`echo $$`


# Set TAU and DS_ID based on SLURM_ARRAY_TASK_ID
if [ "$SLURM_ARRAY_TASK_ID" -le 122 ]; then
  TAU=0.5
  DS_ID=$SLURM_ARRAY_TASK_ID
else
  TAU=0.5
  DS_ID=$((SLURM_ARRAY_TASK_ID - 123))  # Map 123–244 to 0–121
fi

# Run the actual experiment.
python ~/SQR/sqr.py $DS_ID $TAU
mv *.json /var/scratch/fht800/sqr_results2/
#mv *.json /var/scratch/fht800/sqr_test_results/

#jupyter nbconvert --execute ~/hierarchical-conformal-prediction/models/dbpedia14/dbpedia14.ipynb

