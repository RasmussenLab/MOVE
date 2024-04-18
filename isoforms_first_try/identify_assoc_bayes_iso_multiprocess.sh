#!/bin/bash

# The following are commonly used options for running jobs. Remove one
# "#" from the "##SBATCH" lines (changing them to "#SBATCH") to enable
# a given option.

#SBATCH --job-name=500multi

# The number of CPUs (cores) used by your task. Defaults to 1.
#SBATCH --cpus-per-task=50
# The amount of RAM used by your task. Tasks are automatically assigned 15G
# per CPU (set above) if this option is not set.
#SBATCH --mem=1000G
# Request a GPU on the GPU code. Use `--gres=gpu:a100:2` to request both GPUs.
##SBATCH --partition=gpuqueue --gres=gpu:a100:1
# Send notifications when job ends. Remember to update the email address!
##SBATCH --mail-user=qgh533@ku.dk --mail-type=END,FAIL
# Set an error file
#SBATCH --error=multi_bayes_iso.err


module load python/3.11.3
python identify_assoc_bayes_iso_multiprocess.py
