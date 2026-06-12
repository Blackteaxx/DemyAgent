#!/bin/bash
set -x


export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export GRADIO_SERVER_PORT=7860
export NNODES=1 # nodes
export NODE_RANK=0
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')


MODEL_PATH=Qwen/Qwen3-8B

lr=1e-4
epoch=2
output_dir=saves/qwen3_8b/sft_ep${epoch}_lr${lr}  # Changed output_dir based on lr and epoch


FORCE_TORCHRUN=1
llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --template qwen3 \
    --output_dir $output_dir \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --dataset tool \
    --cutoff_len 20000 \
    --max_samples 10000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --logging_steps 10 \
    --save_steps 500 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model true \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000 \
    --deepspeed examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]
