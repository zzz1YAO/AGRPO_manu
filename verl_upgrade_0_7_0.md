# Search-R1 升级 VERL 到 0.7.0（基于 VideoQAgent 对照）

本文基于当前仓库里的 **Search-R1（S）** 与 **VideoQAgent（V）** 的实现对照，整理：

1. 两者独特的多轮逻辑如何与 VERL 的 PPO Ray Trainer 配合。
2. 选择 GRPO 时，多轮逻辑对 batch 的影响。
3. 在 Search-R1 中将 VERL 从 0.1.0 升级到 0.7.0 的操作步骤与注意事项。

> 说明：本教程只给出阅读后的升级指导，不直接修改代码。

---

## 1. 多轮逻辑与 VERL PPO Ray Trainer 的联动方式

### Search-R1：`search_r1/llm_agent/generation.py` + 旧版 `verl/trainer/ppo/ray_trainer.py`

**多轮逻辑核心**
- `LLMGenerationManager.run_llm_loop(...)` 负责完整的多轮交互：
  - `execute_predictions()` 解析 `<search>` / `<answer>` 动作并调用检索工具。
  - 每轮更新 `rollings`（当前 prompt + 已生成 response + observation），并把 response/observation 写入 `responses` 与 `responses_with_info_mask`。
  - 使用 `TensorHelper` 构造 `attention_mask`、`position_ids`，并在 `_compose_final_output` 中输出 `DataProto`。
- 输出的 `DataProto` 会带上：
  - `input_ids`, `attention_mask`, `position_ids`（拼接了 prompt 与 multi-turn responses）。
  - `responses`（仅响应部分）以及 `responses_with_info_mask`（附带 <information> mask）。
  - `info_mask`（用于 info block 相关 mask 计算）。

**与 RayPPOTrainer 的联动**
- 当 `do_search=true` 时，`RayPPOTrainer.fit()` 使用 `LLMGenerationManager` 进行多轮生成：
  - 先构造 `GenerationConfig`（包含 `max_turns`、`max_obs_length`、`search_url`、`topk`）。
  - `generation_manager.run_llm_loop()` 返回 `DataProto`，随后 `actor_rollout_wg.compute_log_prob()` 计算旧策略的 log-prob。
  - `batch` 会在 rollout.n 维度上重复，然后与生成输出合并。

**关键代码位置**
- 多轮逻辑：`Search-R1/search_r1/llm_agent/generation.py`（`LLMGenerationManager.run_llm_loop`）
- Ray trainer 入口：`Search-R1/verl/trainer/ppo/ray_trainer.py`（`fit()` + `_validate()`）

---

### VideoQAgent：`videoagent/action_generation.py` + 新版 `verl/trainer/ppo/ray_trainer.py`

**多轮逻辑核心**
- `VisionLLMGenerationManager.run_llm_loop(...)`：
  - 与 Search-R1 类似，也是「多轮推理 → 解析动作 → 更新 observation」。
  - 支持 `<search>`、`<request_grounding>`、`<answer>` 动作。
  - 通过 video IDs 和字幕/视觉 API 来生成观察信息。
- 输出也封装成 `DataProto`（包含 `responses`、`input_ids`、`attention_mask`、`position_ids`）。

**与 RayPPOTrainer 的联动**
- 在新版 `RayPPOTrainer.fit()` 中，使用 `VisionGenerationConfig` + `VisionLLMGenerationManager.run_llm_loop()` 做 multi-turn rollout。
- 生成后执行：
  - `batch.repeat(rollout.n)` 对齐多次采样。
  - `compute_response_mask(...)` 生成 `response_mask`（新版用于统一 KL / loss / advantage 计算）。

**关键代码位置**
- 多轮逻辑：`VideoQAgent/videoagent/action_generation.py`（`VisionLLMGenerationManager.run_llm_loop`）
- Ray trainer 入口：`VideoQAgent/verl/trainer/ppo/ray_trainer.py`（`fit()` + `compute_advantage()`）

---

### 两者共性总结

| 维度 | Search-R1 | VideoQAgent | 共性 |
|------|-----------|-------------|------|
| 多轮管理类 | `LLMGenerationManager` | `VisionLLMGenerationManager` | 都负责 multi-turn loop + 工具调用 + observation 拼接 |
| 训练入口 | `RayPPOTrainer.fit()` | `RayPPOTrainer.fit()` | 都在 PPO 训练前执行 `run_llm_loop()` |
| 输出数据 | `DataProto` | `DataProto` | 输出都包含 `input_ids / responses / attention_mask / position_ids` |
| batch repeat | `batch.repeat(rollout.n)` | `batch.repeat(rollout.n)` | 为多采样对齐 PPO 训练 |
| 多 GPU padding | `_generate_with_gpu_padding()` | `_generate_with_gpu_padding()` | 支持批次补齐 |

---

## 2. 使用 GRPO 时，多轮逻辑对 batch 的影响

### Search-R1（旧版 VERL）

**GRPO 关键点**
- `compute_advantage()` 在 GRPO 模式下调用 `core_algos.compute_grpo_outcome_advantage()`。
- GRPO 分组依据：`data.non_tensor_batch['uid']`。

**批次影响点**
1. 在 `do_search=true` 路径中：
   - `uid` 不是随机 UUID，而是 `batch.non_tensor_batch['index']`（原始样本 ID）。
   - 这样 GRPO 的分组按 **原始样本 ID** 进行，保证同一 prompt 的多次采样进入同一组。
2. `batch.repeat(rollout.n)` 之后，样本被扩展为多条 response。
3. `_balance_batch()` 会对 batch 进行 token-level 负载均衡重排。
   - 代码里明确提醒：这种重排会改变顺序，需要依赖 `uid` 才能保证 GRPO 的分组正确。

**结论**
- 多轮逻辑本身不直接改变 GRPO batch 结构，但它会增加序列长度，进而影响：
  - `attention_mask` / `response_mask` 的形状
  - `_balance_batch()` 的重排
- 只要 `uid` 保持一致，GRPO 分组依旧正确。

---

### VideoQAgent（新版 VERL）

**GRPO 关键点**
- `compute_advantage()` 在 GRPO 路径中使用 `data.non_tensor_batch['uid']` 分组。
- `core_algos.compute_grpo_outcome_advantage()` 直接使用 `response_mask` 聚合 token-level reward。

**批次影响点**
1. `uid` 在生成前就分配 (`uuid.uuid4()`)，随后 `batch.repeat(rollout.n)` 会复制 uid。这样同一个 prompt 的多采样进入同组。
2. 新版 trainer 会显式生成 `response_mask`，GRPO 用它来 mask outcome reward。
3. `balance_batch` 只改变顺序，不改变 uid，GRPO 分组仍有效。

**结论**
- GRPO 的 batch 分组主要由 `uid` 决定，多轮逻辑不会破坏该机制。
- 真正要注意的是：必须确保 `uid` 在 **repeat 前分配**，否则多采样无法正确归组。

---

## 3. Search-R1 升级 VERL 至 0.7.0 的步骤（教程）

> 目标：用 VideoQAgent 中的 VERL 0.7.0 替换 Search-R1 的 VERL 0.1.0，并让 Search-R1 的多轮逻辑继续正常工作。

### Step 0：准备对照基线

- **旧版（S）**：`Search-R1/verl`（0.1.0）
- **新版（V）**：`VideoQAgent/verl`（0.7.0dev1）

建议先对比以下文件结构差异：
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/main_ppo.py`
- `verl/trainer/config/ppo_trainer.yaml`

---

### Step 1：替换 VERL 子目录

```bash
# 用 VideoQAgent 的 VERL 覆盖 Search-R1
rm -rf Search-R1/verl
cp -R VideoQAgent/verl Search-R1/verl
```

> 这一步会把 Search-R1 的 VERL 换成 0.7.0 的版本。

---

### Step 2：恢复 Search-R1 的多轮逻辑联动

新版 `ray_trainer.py` 默认导入 `videoagent.action_generation`，需要改成 Search-R1 的多轮生成器：

- 修改 `Search-R1/verl/trainer/ppo/ray_trainer.py`：
  - 把 `from videoagent.action_generation import VisionGenerationConfig, VisionLLMGenerationManager`
  - 改为 `from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager`
- 将 `VisionGenerationConfig` 的字段调整为 Search-R1 的生成参数：
  - 需要 `max_turns / max_start_length / max_prompt_length / max_response_length / max_obs_length / search_url / topk`。
- 在训练循环中调用 `LLMGenerationManager.run_llm_loop()`。

**注意**：新版 trainer 内部会计算 `response_mask`，因此保留 `GenerationManager` 输出的 `attention_mask` 即可。

---

### Step 3：更新 `main_ppo.py` 入口结构

新版 VERL 0.7.0 的 `main_ppo.py` 结构与旧版不同（引入 TaskRunner + config 校验）。

操作建议：
1. 参考 `VideoQAgent/verl/trainer/main_ppo.py` 重新组织 Search-R1 的 `main_ppo.py`。
2. 保留 Search-R1 的 `RewardManager`（或按新版 `custom_reward_function` 机制注册）。
3. 让 Ray 初始化逻辑对齐新版：
   - 使用 `get_ppo_ray_runtime_env()`
   - 调用 `validate_config()`

**核心目标**：新版入口要能正确实例化 `RayPPOTrainer`，并加载 Search-R1 自定义 reward/多轮逻辑。

---

### Step 4：迁移配置文件结构

VERL 0.7.0 把 config 拆成多个子模块（`actor/critic/rollout/data/...`）。
因此需要将 Search-R1 的旧版 `ppo_trainer.yaml` 迁移到新版配置结构：

**关键迁移字段**
- `data.*`：训练/验证文件、batch size、max_prompt_length、max_response_length、max_start_length、max_obs_length
- `actor_rollout_ref.rollout.n` / `n_agent`：与 GRPO/多轮采样相关
- `algorithm.adv_estimator`：支持 `gae` / `grpo`
- `retriever.url` / `retriever.topk`：Search-R1 的检索地址
- `max_turns` / `do_search`：多轮交互开关

**注意**：新版 `ppo_trainer.yaml` 默认采用 `defaults` 机制引用子配置，需要在 Search-R1 中补齐对应模块，或直接复制 VideoQAgent 的配置结构进行改造。

---

### Step 5：更新训练脚本参数

Search-R1 的 `train_ppo.sh` / `train_grpo.sh` 仍基于旧版配置键。
升级后需要：

- 对齐新版 config 名称（例如 `trainer.balance_batch`, `actor_rollout_ref.rollout.n`, `algorithm.norm_adv_by_std_in_grpo` 等）。
- 确保 `algorithm.adv_estimator=grpo` 时 `actor_rollout_ref.actor.use_kl_loss=true` 或 `algorithm.use_kl_in_reward=true`。

---

### Step 6：检查 Reward 逻辑与 DataProto 字段

Search-R1 的 `RewardManager` 直接读取 `prompts` / `responses` / `attention_mask`，这在新版仍成立，但需要注意：

- 新版 trainer 可能在 `reward_model` / `reward_fn` 的调用路径上增加额外字段（如 `response_mask`）。
- 如果 `RewardManager` 依赖 `data.non_tensor_batch['reward_model']` / `data_source` 等字段，需确保这些字段仍在数据集加工流程中被保留。

---

## 常见升级坑位汇总

1. **Ray trainer import 未替换**
   - 默认会调用 `videoagent.action_generation`。
2. **配置结构不兼容**
   - 0.7.0 使用 `defaults` 分模块配置，旧版 YAML 无法直接复用。
3. **GRPO 分组失效**
   - 若 `uid` 在 repeat 之后生成，会导致分组错误。
4. **Reward 管理逻辑断裂**
   - 需要检查 `reward_model` 与 `RewardManager` 的字段来源。

---

## 结论

升级 Search-R1 到 VERL 0.7.0 的核心步骤是：

1. 直接替换 `verl` 目录。
2. 把 `RayPPOTrainer` 的多轮逻辑联动切回 `search_r1.llm_agent.generation`。
3. 迁移 `main_ppo.py` 入口与 config 架构。
4. 检查 GRPO 的 batch grouping（`uid`）以及 reward 字段是否完整。

这样可以在保持 Search-R1 multi-turn search agent 逻辑的同时，享受新版 VERL 的特性支持。
