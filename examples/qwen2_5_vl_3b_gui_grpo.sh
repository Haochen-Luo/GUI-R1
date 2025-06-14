set -x
# export CUDA_HOME=/usr
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

SYSTEM_PROMPT=""""""

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=datasets/train.parquet \
    data.val_files=datasets/test.parquet \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.compute_score=r1gui \
    trainer.experiment_name=qwen2_5_vl_3b_guir1_grpo \
    trainer.n_gpus_per_node=6 \
    data.max_pixels=1258291 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.val_batch_size=256
