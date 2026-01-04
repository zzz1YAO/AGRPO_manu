#!/bin/bash
export RAY_TMPDIR=/home/rliuay/runtao/proj_videoqa
export VLLM_USE_V1=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export WAND_PROJECT="longvideoagent_train"

# 视觉框架正式训练脚本 - TVQA Plus数据集
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR='/home/rliuay/runtao/proj_videoqa/Vision-R1_bbox/data/tvqa_plus'
export TRAIN_DATA_DIR='/home/rliuay/runtao/proj_videoqa/Vision-R1_bbox/data/tvqa_plus/train'
export TEST_DATA_DIR='/home/rliuay/runtao/proj_videoqa/Vision-R1_bbox/data/tvqa_plus/test'

# 模型选择
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME="dapo-$(date +%Y%m%d-%H%M%S)"

# 设置环境变量
export VLLM_ATTENTION_BACKEND=XFORMERS

echo "=== 开始正式训练 ==="
echo "使用GPU: $CUDA_VISIBLE_DEVICES"
echo "训练数据: $TRAIN_DATA_DIR/train_newaction.parquet"
echo "测试数据: $TEST_DATA_DIR/test.parquet"
echo "实验名称: $EXPERIMENT_NAME"

# 正式训练配置
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_dapo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_DIR/train_newaction.parquet \
    data.val_files=$TEST_DATA_DIR/test.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=8100 \
    data.max_response_length=1500 \
    +data.max_obs_length=25600 \
    +data.max_start_length=8192 \
    data.truncation='left' \
    +data.shuffle_train_dataloader=True \
    +data.video_id_key="vid_name" \
    data.gen_batch_size=4 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=128 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9999 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=9999 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=9999 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    algorithm.use_kl_in_reward=false \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    algorithm.filter_groups.enable=true \
    algorithm.filter_groups.metric=seq_final_reward \
    algorithm.filter_groups.max_num_gen_batches=10 \
    reward_model.overlong_buffer.enable=true \
    reward_model.overlong_buffer.len=4096 \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    trainer.logger=['wandb'] \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5000 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=3001 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=../$EXPERIMENT_NAME \
    trainer.val_only=false \
    trainer.val_before_train=false \
    trainer.resume_mode=auto \
    +max_turns=5 \
    +do_vision=true \
    +vision.base_frame_dir="../Tvqa_data/bbt_frames" \
    +vision.bbox_json_path="../Tvqa_data/clip_bbox_mapping.json" \
    +vision.model="grok-4-fast-non-reasoning" \
    2>&1 | tee $EXPERIMENT_NAME.log

echo "=== 训练完成 ==="
echo "日志文件: $EXPERIMENT_NAME.log"
echo "检查点保存位置: verl_checkpoints/$EXPERIMENT_NAME"
