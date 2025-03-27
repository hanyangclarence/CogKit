#!/usr/bin/env bash
# Run by `bash scripts/train_ddp_i2v.sh`

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Check for debug mode
DEBUG_ARG=()
if [[ $# -ge 1 && "$1" == "--debug" ]]; then
    DEBUG_ARG=(--debug)
    shift
fi

# Ensure no unknown arguments are passed
if [[ $# -gt 0 ]]; then
    echo "Error: Unknown arguments: $@"
    exit 1
fi

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX-5B-I2V"
    --model_name "cogvideox-i2v"  # candidate: ["cogvideox-i2v", "cogvideox1.5-i2v"]
    --model_type "i2v"
    --training_type "lora"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "training_logs"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/gpfs/u/scratch/LMCG/LMCGhazh/yanghan/embodied_o1/CogKit/quickstart/data/i2v"

    # Note:
    #  for CogVideoX series models, number of training frames should be **8N+1**
    #  for CogVideoX1.5 series models, number of training frames should be **16N+1**
    --train_resolution "17x480x720"  # (frames x height x width)
)

# Training Configuration
TRAIN_ARGS=(
    --seed 42  # random seed
    --train_epochs 1  # number of training epochs
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
    --checkpointing_steps 10  # save checkpoint every x steps
    --checkpointing_limit 2   # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true   # ["true", "false"]
    --validation_steps 10  # should be multiple of checkpointing_steps
    --gen_fps 16
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
    "${DEBUG_ARG[@]}"
