#!/bin/bash 
#SBATCH --job-name=yh
#SBATCH -o slurm_output/i2v_lora_%j.out
#SBATCH -e slurm_output/i2v_lora_%j.err
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # total number of tasks across all nodes
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1

source /gpfs/u/home/LMCG/LMCGhazh/scratch/miniconda3x86/etc/profile.d/conda.sh
conda activate cogvideo

bash launch_i2v_lora.sh