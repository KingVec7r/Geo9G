#!/bin/bash

GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6012
 
MODEL="$WORKSPACE_DIR/Geo9G/FM9G4B-V"
DATA="$DATA_BASE_DIR/geo_mix_vrsb_train_fm9g_format.json"
EVAL_DATA="$DATA_BASE_DIR/geo_mix_vrsb_val_fm9g_format.json"

MODEL_MAX_Length=2048 # if conduct multi-images sft, please set MODEL_MAX_Length=4096


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm true \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --output_dir output/ \
    --logging_dir output/ \
    --logging_strategy "steps" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero2.json \
    --report_to "tensorboard" 
