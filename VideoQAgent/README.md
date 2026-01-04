# LongVideoAgent

**A multi-agent framework for reasoning over long videos, with a master LLM coordinating grounding and vision agents for efficient, fine-grained video QA.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)  
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)  
[![GitHub stars](https://img.shields.io/github/stars/Visualignment/VideoQAgent?style=social)](https://github.com/Visualignment/VideoQAgent)

---

## Overview

This repository provides a **complete implementation** of **LongVideoAgent** — a multi-agent system for reasoning over hour-long videos. It addresses the limitations of traditional MLLMs that rely on lossy compression or limited tools, by enabling a **master LLM** to iteratively coordinate:

- A **grounding agent** to localize question-relevant video segments.
- A **vision agent** to extract fine-grained visual observations (objects, actions, faces, etc.).

The framework supports:
- **Training**: Reinforcement learning via **[VERL](https://github.com/volcengine/verl)** to optimize the master agent's planning and multi-agent cooperation.
- **Evaluation**: Full agent pipelines on episode-level benchmarks.
- **Baseline**: API-based evaluation (e.g., Gemini) on TVQA and TVQA+ for rapid prototyping.

Key datasets: **LongTVQA** and **LongTVQA+** (aggregated from TVQA/TVQA+). The system achieves state-of-the-art results with interpretable, multi-round decision traces.

## Installation

### Preferred Method

First, create a virtual environment named `lvagent`, clone our repository, and perform the installation in the project directory.

```bash
conda create -n lvagent python=3.11
conda activate lvagent
cd ./VideoQAgent
# Install verl
pip install -e .
# Install flash-attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

If you encounter any unresolved conflicts during installation, try the following alternative approach:

```bash
# Install verl without dependencies
pip install -e . --no-deps
# Install other dependencies
pip install -r requirements.txt
```

## Quick Start

We provide a quick start script that uses Qwen2.5-3B-Instruct as the base model. This script is recommended for hardware with at least 2x H800 GPUs.

To begin training immediately, run:

```bash
bash scripts/quickstart_qwen_2_5_3B_grpo.sh
```



Using the provided quick start script (`scripts/quickstart_qwen_2_5_3B_grpo.sh`) as an example, below is a detailed breakdown of key parameters and other important considerations.

```bash
#!/bin/bash
export VLLM_USE_V1=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export WAND_PROJECT="longvideoagent_train"
# Visual framework training script - TVQA+ dataset
export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR='/home/rliuay/runtao/proj_videoqa/Vision-R1_bbox/data/tvqa_plus'
export TRAIN_DATA_DIR='/home/rliuay/runtao/proj_videoqa/Vision-R1_bbox/data/tvqa_plus/train'
export TEST_DATA_DIR='/home/rliuay/runtao/proj_videoqa/Vision-R1_bbox/data/tvqa_plus/test'
# Model selection
export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
export EXPERIMENT_NAME=$WAND_PROJECT$(date +%Y%m%d_%H%M%S)
# Environment settings
export VLLM_ATTENTION_BACKEND=XFORMERS
# Training configuration
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_DIR/train_newaction.parquet \
    data.val_files=$TEST_DATA_DIR/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=4 \
    data.max_prompt_length=12800 \
    data.max_response_length=1500 \
    +data.max_obs_length=-1 \
    +data.max_start_length=9999 \
    data.truncation='error' \
    +data.shuffle_train_dataloader=True \
    +data.video_id_key="vid_name" \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=128 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=15001 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    reward_model.reward_manager="tvqa_plus" \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5000 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=401 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=../$EXPERIMENT_NAME \
    trainer.val_only=false \
    trainer.val_before_train=false \
    +max_turns=5 \
    +do_vision=true \
    +vision.base_frame_dir="../Tvqa_data/bbt_frames" \
    +vision.bbox_json_path="../Tvqa_data/clip_bbox_mapping.json" \
    +vision.model="grok-4-fast-non-reasoning" \
    2>&1 | tee $EXPERIMENT_NAME.log
```

For most parameters, refer to the **Config Explanation** section in the official VERL documentation.

### Key Parameters to Note

- `data.max_prompt_length`: Maximum length of the input prompt in multi-turn interactions.
- `data.max_response_length`: Maximum length of each response per turn.
- `data.max_obs_length`: Maximum length that triggers vision-based reasoning in multi-turn dialogue; set to `-1` to disable truncation.
- `data.video_id_key`: The key name representing the video ID in datasets like LongTVQA (e.g., `"vid_name"`).
- `max_turns`: Maximum number of interaction turns (K); defaults to `5`.



## Baseline Evaluation

This section outlines the baseline evaluation script for the VideoQAgent project, located in `src/evaluation/baseline`. The script evaluates a model's performance on the TVQA dataset (or variants like TVQA+) using an API-based model for question-answering tasks.

### Overview

The `evaluate_api_tvqa.py` script processes video frames, subtitles, and questions to generate answers via an API-based model (e.g., Gemini). It supports input validation, multi-threaded processing, and result output. Dataset variants are handled by providing appropriate input files.

### Directory Structure

- `src/evaluation/baseline/`
  - `evaluate_api_tvqa.py`: Runs evaluation on TVQA or its variants using an API-based model.
  - `utils.py`: Contains utility functions, including `run_simple_qa`, for evaluation logic.

### Script

#### `evaluate_api_tvqa.py`

Evaluates a model on the TVQA dataset (or variants like TVQA+) by calling an API. The model name corresponds to the API provider's model (e.g., `gemini-2.5-pro-exp-03-25`).

**Usage**:

```bash
python src/evaluation/baseline/evaluate_api_tvqa.py \
  --questions-path ./tvqa_question.json \
  --subs-path ./substitle.json \
  --output-filename ./eval_gemini_tvqa.json \
  --base-frame-dir ../bbt_frames \
  --model gemini-2.5-pro-exp-03-25 \
  --threads 10
```

**Arguments**:

- `--questions-path`: Path to the TVQA questions JSON file (required).
- `--subs-path`: Path to the subtitles JSON file (required).
- `--output-filename`: Path for saving results (required).
- `--base-frame-dir`: Directory with video frame images (required).
- `--model`: API provider's model name (required).
- `--threads`: Number of threads for parallel processing (required; adjust based on API rate limits).

**Behavior**:

- Validates input paths (questions, subtitles, frames).
- Calls `run_simple_qa` from `utils.py` to perform evaluation via API.
- Saves results to the specified output file.
- Supports TVQA variants by adjusting the `--questions-path` input.

#### `utils.py`

Includes `run_simple_qa`, which handles API calls for question answering. Configure the API base URL (e.g., `base_url`) in `utils.py` to match the API provider's endpoint.

### Prerequisites

- Ensure input files (questions JSON, subtitles JSON, frame directory) exist.
- Configure the API base URL in `utils.py` for the chosen model.
- The specified model must be supported by the API provider.
- Install Python dependencies (e.g., `argparse`, `os`).
- Set `--threads` based on the API provider's rate limits to avoid overloading.

### Notes

- Update the API base URL in `utils.py` to match your provider's endpoint.
- Use a thread count suitable for the API's maximum load to optimize performance.
- Ensure the output directory is writable for result storage.

For TVQA+ or other variants, provide the appropriate questions JSON file via `--questions-path`.
