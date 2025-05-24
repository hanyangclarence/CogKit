#!/bin/bash

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
    --model_path "THUDM/CogVideoX1.5-5B-I2V"
    --model_name "cogvideox1.5-i2v"  # candidate: ["cogvideox-i2v", "cogvideox1.5-i2v"]
    --model_type "i2v"
    --training_type "sft"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "$OUTPUT_DIR"
    --report_to "wandb"
)

# Data Configuration
DATA_ARGS=(
    --data_root "no_use"

    # Note:
    #  for CogVideoX series models, number of training frames should be **8N+1**
    #  for CogVideoX1.5 series models, number of training frames should be **16N+1**
    --train_resolution "17x768x768"  # (frames x height x width)
)

# Training Configuration
TRAIN_ARGS=(
    --seed 42  # random seed
    --train_epochs 1000  # number of training epochs

    --learning_rate 2e-5

    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 15
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"]  Note: CogVideoX-2B only supports fp16 training
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
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
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


# Combine all arguments and launch training
accelerate launch --config_file ../quickstart/configs/accelerate_config_vela.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${DEBUG_ARG[@]}" \
    "${CONFIG_ARG[@]}"