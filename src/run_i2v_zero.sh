#!/bin/bash 
#SBATCH --job-name=yh
#SBATCH -o slurm_output/i2v_zero_%j.out
#SBATCH -e slurm_output/i2v_zero_%j.err
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8  # total number of tasks across all nodes
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:8


conda activate cogvideo

bash launch_i2v_zero.sh