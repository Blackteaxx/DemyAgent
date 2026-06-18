#!/bin/bash
set -x

# ============================================================
#  统一评测脚本：一个脚本跑 aime / gpqa / lcb（或全部）
#
#  通过环境变量传参（不传则用下面的默认值）：
#    MODEL_PATH       要评测的模型路径
#    EXPERIMENT_NAME  实验名（wandb 名 + 输出目录名，前后模型记得用不同名）
#    BENCH            评测集: all(默认) | aime | gpqa | lcb | math(=aime+gpqa)
#    N_GPUS           使用的 GPU 数（默认 8）。需 >= tp(4)，且能被 tp 整除
#
#  注意：aime/gpqa 的 ground_truth 是字符串，lcb 是结构体(测试用例)，
#       两者 schema 不兼容，不能拼进同一个 val 数据集。
#       所以 BENCH=all 会自动跑两个独立 job：先 math 再 lcb。
#
#  例子：
#    MODEL_PATH=/workspace/DemyAgent/model/Qwen3-4B-RA-SFT \
#    EXPERIMENT_NAME=sft  BENCH=all \
#    bash recipe/demystify/eval/eval_qwen3_4b.sh
# ============================================================

# ===================== 环境变量传参（不传用默认值）=====================
MODEL_PATH="${MODEL_PATH:-/workspace/DemyAgent/model/Qwen3-4B-RA-SFT}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval-qwen3-4b-ra-sft}"
BENCH="${BENCH:-all}"
N_GPUS="${N_GPUS:-8}"
# ======================================================================

export VLLM_USE_V1=1

# ================= data =================
DATA_ROOT=/workspace/DemyAgent/data
open_agent_rl=$DATA_ROOT/Open-AgentRL-30K/Open-AgentRL-30K.parquet
gpqa_diamond=$DATA_ROOT/eval/gpqa-diamond/gpqa_diamond.parquet
aime_2024=$DATA_ROOT/eval/aime2024/aime_2024_problems.parquet
aime_2025=$DATA_ROOT/eval/aime2025/aime_2025_problems.parquet
livecodebench=$DATA_ROOT/eval/livecodebench-v6/lcb_v6_2502_2505.parquet

train_files="['$open_agent_rl']"

# 数学类(string ground_truth) 与 代码类(struct ground_truth) 必须分开跑
# math_files="['$gpqa_diamond','$aime_2024','$aime_2025']"
math_files="['$aime_2024','$aime_2025']"
aime_files="['$aime_2024','$aime_2025']"
gpqa_files="['$gpqa_diamond']"
lcb_files="['$livecodebench']"

# 根据 BENCH 决定要跑哪些 job，每个 job = "后缀|test_files"
# all 拆成两个独立 job（math + lcb），其余各一个 job
declare -a JOBS
case "$BENCH" in
  all)  JOBS=("math|$math_files" "lcb|$lcb_files") ;;
  aime) JOBS=("aime|$aime_files") ;;
  gpqa) JOBS=("gpqa|$gpqa_files") ;;
  lcb)  JOBS=("lcb|$lcb_files") ;;
  math) JOBS=("math|$math_files") ;;
  *) echo "[ERROR] 未知 bench: '$BENCH' (可选: all|aime|gpqa|lcb|math)"; exit 1 ;;
esac

# ================= 固定超参 =================
tool_config_path=recipe/demystify/sandbox_fusion_tool_config.yaml
project_name=demystify-agentic-rl

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
reward_manager=dapo
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0
max_turns=16
max_prompt_length=4096
max_response_length=16384
actor_lr=1e-6
train_batch_size=64
ppo_mini_batch_size=16
n_resp_per_prompt=8
n_resp_per_prompt_val=8
infer_tp=1
train_sp=2
offload=True
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

# 校验 N_GPUS 与 tp 兼容（vllm 要求 GPU 数能被 tensor_parallel 整除）
if [ "$N_GPUS" -lt "$infer_tp" ] || [ $(( N_GPUS % infer_tp )) -ne 0 ]; then
    echo "[ERROR] N_GPUS=$N_GPUS 必须 >= $infer_tp 且能被 $infer_tp 整除(tensor_parallel_size)"
    exit 1
fi

# ================= 单个评测 job =================
run_eval() {
    local exp_name="$1"
    local test_files="$2"

    local default_local_dir=/workspace/DemyAgent/eval/checkpoint/$exp_name
    local VAL_SAVE_PATH="${default_local_dir}/validation"
    mkdir -p "${default_local_dir}/rollout" "$VAL_SAVE_PATH"

    echo "=================================================="
    echo " >>> RUN EVAL"
    echo " MODEL_PATH      = $MODEL_PATH"
    echo " EXPERIMENT_NAME = $exp_name"
    echo " test_files      = $test_files"
    echo "=================================================="

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=$adv_estimator \
        algorithm.use_kl_in_reward=$use_kl_in_reward \
        algorithm.kl_ctrl.kl_coef=$kl_coef \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.return_raw_chat=True \
        data.train_batch_size=$train_batch_size \
        data.max_prompt_length=$max_prompt_length \
        data.max_response_length=$max_response_length \
        data.prompt_key=prompt \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.custom_cls.path=recipe/demystify/reward.py \
        data.custom_cls.name=CustomRLHFDataset \
        custom_reward_function.path=recipe/demystify/reward.py \
        custom_reward_function.name=compute_score \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
        actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
        actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
        actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
        actor_rollout_ref.actor.optim.lr=$actor_lr \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
        actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
        actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
        actor_rollout_ref.rollout.multi_turn.format=hermes \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
        actor_rollout_ref.rollout.n=$n_resp_per_prompt \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
        reward_model.reward_manager=${reward_manager} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
        +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
        trainer.logger=['console','swanlab'] \
        trainer.project_name=$project_name \
        trainer.experiment_name=$exp_name \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.val_before_train=True \
        trainer.validation_data_dir=${VAL_SAVE_PATH} \
        trainer.log_val_generations=20 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.default_local_dir=$default_local_dir \
        trainer.val_only=True \
        trainer.test_freq=10 \
        trainer.total_epochs=1
}

# ================= 依次跑所有 job =================
for job in "${JOBS[@]}"; do
    suffix="${job%%|*}"
    files="${job#*|}"
    # 单一基准时不加后缀，all 时加 -math / -lcb 区分
    if [ "${#JOBS[@]}" -gt 1 ]; then
        exp_name="${EXPERIMENT_NAME}-${suffix}"
    else
        exp_name="${EXPERIMENT_NAME}"
    fi
    run_eval "$exp_name" "$files"
done
