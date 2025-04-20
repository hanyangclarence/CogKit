#!/bin/bash

#SBATCH --job-name=yh
#SBATCH -o slurm_output/i2v_zero_8nodes_%j.out
#SBATCH -e slurm_output/i2v_zero_8nodes_%j.err
#SBATCH --mem=400G
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1  # total number of tasks across all nodes
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:8

source /gpfs/u/home/LMCG/LMCGhazh/scratch/miniconda3x86/etc/profile.d/conda.sh
conda activate cogvideo

export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export GPUS_PER_NODE=8

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Initialize argument variables
DEBUG_ARG=()
CONFIG_FILE=""
OUTPUT_DIR=""
RESUME_CHECKPOINT=""

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG_ARG=(--debug)
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --resume_from_checkpoint)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check for required arguments
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: --config argument is required."
    exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output_dir argument is required."
    exit 1
fi

# Set config argument
CONFIG_ARG=(--config "$CONFIG_FILE")

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX-5B-I2V"
    --model_name "cogvideox-i2v"  # candidate: ["cogvideox-i2v", "cogvideox1.5-i2v"]
    --model_type "i2v"
    --training_type "sft"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "$OUTPUT_DIR"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "no_use"

    # Note:
    #  for CogVideoX series models, number of training frames should be **8N+1**
    #  for CogVideoX1.5 series models, number of training frames should be **16N+1**
    --train_resolution "17x480x720"  # (frames x height x width)
)

# Training Configuration
TRAIN_ARGS=(
    --seed 42  # random seed
    --train_epochs 1000  # number of training epochs

    --learning_rate 2e-5

    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "fp16"  # ["no", "fp16"]  Note: CogVideoX-2B only supports fp16 training
    ########################################################################
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 100 # save checkpoint every x steps
    --checkpointing_limit 1 # maximum number of checkpoints to keep, after which the oldest one is deleted
)
# Add resume_from_checkpoint if provided
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    CHECKPOINT_ARGS+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
fi

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation false   # ["true", "false"]
    --validation_steps 500  # should be multiple of checkpointing_steps
    --gen_fps 16
    --validation_num 1  # number of validation samples
)

export LOCAL_RANK=${SLURM_LOCALID}
export GLOBAL_RANK=${SLURM_PROCID}

# Print the environment variables into a txt file
echo "Master Address: ${MASTER_ADDR}" > "slurm_output/env_variables_${GLOBAL_RANK}.txt"
echo "Master Port: ${MASTER_PORT}" >> "slurm_output/env_variables_${GLOBAL_RANK}.txt"
echo "SLURM Node List: ${SLURM_NODELIST}" >> "slurm_output/env_variables_${GLOBAL_RANK}.txt"
echo "SLURM Job ID: ${SLURM_JOB_ID}" >> "slurm_output/env_variables_${GLOBAL_RANK}.txt"
echo "SLURM Proc ID: ${SLURM_PROCID}" >> "slurm_output/env_variables_${GLOBAL_RANK}.txt" # Should be set by SLURM for each process
echo "SLURM NTASKS: ${SLURM_NTASKS}" >> "slurm_output/env_variables_${GLOBAL_RANK}.txt"   # Should be 64
echo "SLURM NNODES: ${SLURM_NNODES}" >> "slurm_output/env_variables_${GLOBAL_RANK}.txt"
echo "SLURM GPU_PER_NODE: ${GPUS_PER_NODE}" >> "slurm_output/env_variables_${GLOBAL_RANK}.txt"
echo "LOCAL_RANK: ${LOCAL_RANK}" >> "slurm_output/env_variables_${GLOBAL_RANK}.txt"
echo "GLOBAL_RANK: ${GLOBAL_RANK}" >> "slurm_output/env_variables_${GLOBAL_RANK}.txt"
echo "------------------------------------" >> "slurm_output/env_variables_${GLOBAL_RANK}.txt"
echo "CONFIG_FILE: ${CONFIG_FILE}" >> "slurm_output/env_variables_${GLOBAL_RANK}.txt"
echo "OUTPUT_DIR: ${OUTPUT_DIR}" >> "slurm_output/env_variables_${GLOBAL_RANK}.txt"
echo "RESUME_CHECKPOINT: ${RESUME_CHECKPOINT}" >> "slurm_output/env_variables_${GLOBAL_RANK}.txt"
echo "------------------------------------" >> "slurm_output/env_variables_${GLOBAL_RANK}.txt"

export LAUNCHER="accelerate launch --config_file ../quickstart/configs/accelerate_config_8nodes.yaml --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
"

export SCRIPT="train.py"

export ARGS=" \
    ${MODEL_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAIN_ARGS[@]} \
    ${SYSTEM_ARGS[@]} \
    ${CHECKPOINT_ARGS[@]} \
    ${VALIDATION_ARGS[@]} \
    ${DEBUG_ARG[@]} \
    ${CONFIG_ARG[@]} \
"

export CMD="$LAUNCHER $SCRIPT $ARGS"

echo $CMD >> "slurm_output/env_variables_${GLOBAL_RANK}.txt"

$CMD