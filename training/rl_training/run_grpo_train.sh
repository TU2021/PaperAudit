set -euxo pipefail

# ========== 环境变量 & 目录配置 ==========
export NCCL_IB_DISABLE=0
export NCCL_TRANSPORT=IB
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${PYTHONPATH:-}:/mnt/parallel_ssd/home/zdhs0124/AI4S_review/verl"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_DISABLE_DASHBOARD=1
export VLLM_USE_V1=1
export WANDB_DISABLED=true


TIME_STAMP=$(date +"%Y%m%d_%H%M%S")
ROOT_EXP_DIR="/mnt/parallel_ssd/home/zdhs0124/AI4S_review/verl/grpo_qwen3/train"
EXPERIMENT_DIR="${ROOT_EXP_DIR}/${TIME_STAMP}"
ROLLOUT_DIR="${EXPERIMENT_DIR}/rollout_output"
VAL_ROLLOUT_DIR="${EXPERIMENT_DIR}/val_rollout_output"


mkdir -p "${EXPERIMENT_DIR}" "${ROLLOUT_DIR}" "${VAL_ROLLOUT_DIR}"


TRAIN_DATA="~/train.parquet"
VAL_DATA="~/test.parquet"

python -m verl.trainer.main_ppo \
  --config-path ~/verl/verl/trainer/config \
  --config-name grpo_config \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="[$VAL_DATA]" \
  data.prompt_key="prompt" \
  data.max_prompt_length=4096 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.train_batch_size=64 \
  data.micro_batch_size_per_gpu=8 \
  actor_rollout_ref.hybrid_engine=true \
  actor_rollout_ref.model.path="~/model/Llama-3.2-3B-Instruct-merged" \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.max_model_len=12288 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.trace.token2text=true \
  actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=100 \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=true \
  algorithm.use_kl_in_reward=true \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','scheduler','extra'] \
  trainer.total_epochs=3 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8 \
  trainer.experiment_name="'${TIME_STAMP}'" \
  trainer.default_local_dir="${EXPERIMENT_DIR}" \
  trainer.logger=['console','swanlab'] \
  trainer.save_freq=100 \
  trainer.test_freq=50 \
  trainer.val_before_train=false \
  trainer.rollout_data_dir="${ROLLOUT_DIR}" \
  trainer.validation_data_dir="${VAL_ROLLOUT_DIR}" \
  trainer.log_val_generations=100 \
  reward_model.enable=false \
  critic.enable=false \
  custom_reward_function.path="~/reward_function.py" \
  custom_reward_function.name="compute_score" \
  2>&1 | tee "${EXPERIMENT_DIR}/training.log" 

