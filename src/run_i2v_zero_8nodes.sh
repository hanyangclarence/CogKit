#!/bin/bash 
#SBATCH --job-name=yh
#SBATCH -o slurm_output/i2v_zero_8nodes_%j.out
#SBATCH -e slurm_output/i2v_zero_8nodes_%j.err
#SBATCH --mem=400G
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8  # total number of tasks across all nodes
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:8

source /gpfs/u/home/LMCG/LMCGhazh/scratch/miniconda3x86/etc/profile.d/conda.sh
conda activate cogvideo

bash launch_i2v_zero_8nodes.sh