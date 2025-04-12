#!/usr/bin/env bash
# Run by `bash scripts/train_ddp_i2v.sh`

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Initialize argument arrays
DEBUG_ARG=()
CONFIG_ARG=(--config "config/v2.yaml")

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG_ARG=(--debug)
            shift
            ;;
        *)
            echo "Error: Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX-5B-I2V"
    --model_name "cogvideox-i2v"  # candidate: ["cogvideox-i2v", "cogvideox1.5-i2v"]
    --model_type "i2v"
    --training_type "lora"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "training_logs_lora"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/gpfs/u/scratch/LMCG/LMCGhazh/enroot/rlbench_data/root/RACER-DataGen/racer_datagen/rlbench_videos"

    # Note:
    #  for CogVideoX series models, number of training frames should be **8N+1**
    #  for CogVideoX1.5 series models, number of training frames should be **16N+1**
    --train_resolution "17x480x720"  # (frames x height x width)
)

# Training Configuration
TRAIN_ARGS=(
    --seed 42  # random seed
    --train_epochs 1000  # number of training epochs
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "fp16"  # ["no", "fp16"]
    --learning_rate 2e-5
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 300  # save checkpoint every x steps
    --checkpointing_limit 1   # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation false   # ["true", "false"]
    --validation_steps 10  # should be multiple of checkpointing_steps
    --gen_fps 16
    --validation_num 1  # number of validation samples
)

# Combine all arguments and launch training
accelerate launch train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${DEBUG_ARG[@]}" \
    "${CONFIG_ARG[@]}"