# sudo apt-get update
# sudo apt-get install ffmpeg libsm6 libxext6 libaio-dev -y

export PYTHONPATH=$(pwd)

model_name_or_path="LanguageBind/Video-LLaVA-7B-hf"
max_img_seq_len=20000
max_txt_seq_len=4096

lora_enabled=false
qlora_enabled=false
global_batch_size=128
trainable_modules="all"

DATA_CONFIG_FILE="train/data_configs/videollava/data_vista_videollava.yaml"
OUTPUT_DIR="output/instruction_tuning"
if [ -z "$RUN_NAME" ]; then
    RUN_NAME="videollava_vista_data_sft_lr1e-6_f_8"
fi

export WANDB_API_KEY="<your wandb key>"
export WANDB_PROJECT="video_llm_videollava_instruction_tuning"
export WANDB_NAME=$RUN_NAME

# set default TRAINING_STEPS and LEARNING_RATE values
if [ -z "$TRAINING_EPOCHS" ]; then
    TRAINING_EPOCHS=1
fi

if [ -z "$LEARNING_RATE" ]; then
    LEARNING_RATE=1e-6
fi

if [ $lora_enabled = true ]; then
    echo "lora is enabled"
    if [ $qlora_enabled = true ]; then
        echo "qlora & dora is enabled"
        RUN_NAME="${RUN_NAME}_img${max_img_seq_len}_txt${max_txt_seq_len}_qlora"
    else
        RUN_NAME="${RUN_NAME}_img${max_img_seq_len}_txt${max_txt_seq_len}_lora"
    fi
else
    echo "lora is disabled"
    RUN_NAME="${RUN_NAME}_img${max_img_seq_len}_txt${max_txt_seq_len}"
fi
echo "RUN_NAME = $RUN_NAME"


# resume from checkpoint
resume_from_checkpoint=""
if [ -d $resume_from_checkpoint ]; then
    echo "resume_from_checkpoint = $resume_from_checkpoint"
    export WANDB_LAST_RUN_ID="your_last_run_id"
else
    echo "No checkpoint found, training from scratch"
fi

export NCCL_DEBUG=INFO;
export CXX=g++;

MAIN_PROCESS_IP=${MASTER_ADDR}
MAIN_PROCESS_PORT=${MASTER_PORT}
NUM_MACHINES=${WORLD_SIZE}
MACHINE_RANK=${RANK}


NGPU_PER_NODE=${NPROC_PER_NODE}
NUM_PROCESSES=$((${NUM_MACHINES} * ${NGPU_PER_NODE}))
NUM_WORKERS=16

if [ $NUM_WORKERS -gt 112 ]; then
    NUM_WORKERS=112
fi


echo MAIN_PROCESS_IP=${MAIN_PROCESS_IP}
echo MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}
echo NUM_MACHINES=${NUM_MACHINES}
echo MACHINE_RANK=${MACHINE_RANK}
echo NUM_PROCESSES=${NUM_PROCESSES}
echo NUM_WORKERS=${NUM_WORKERS}
echo "Running ${RUN_NAME}"


if [ $lora_enabled = true ]; then
    echo "lora is enabled"
    config_file="train/accelerate_configs/accelerate_config_zero2.yaml"
    echo $config_file
else
    echo "lora is disabled"
    config_file="train/accelerate_configs/accelerate_config_zero3.yaml"
    echo $config_file
fi

per_device_train_batch_size=1
gradient_accumulation_steps=$(($global_batch_size / ($per_device_train_batch_size * $NUM_PROCESSES)))
echo gradient_accumulation_steps=$global_batch_size / \($per_device_train_batch_size \* $NUM_PROCESSES\) = $gradient_accumulation_steps


accelerate launch --config_file=$config_file \
    --machine_rank ${MACHINE_RANK} --main_process_ip ${MAIN_PROCESS_IP} --main_process_port ${MAIN_PROCESS_PORT} \
    --num_machines=${NUM_MACHINES} --num_processes=${NUM_PROCESSES} \
    train/train_videollava.py \
    --model_name_or_path $model_name_or_path \
    --data_config_file $DATA_CONFIG_FILE \
    --run_name $RUN_NAME \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $TRAINING_EPOCHS \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --save_steps 200 \
    --eval_steps 200 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers $NUM_WORKERS \
    --report_to "wandb" \
    --do_train True \
    --trainable_modules "$trainable_modules" \
    --lora_enabled $lora_enabled \
    --qlora_enabled $qlora_enabled \
    --max_img_seq_len $max_img_seq_len \
    --max_txt_seq_len $max_txt_seq_len \
    --resume_from_checkpoint "$resume_from_checkpoint"