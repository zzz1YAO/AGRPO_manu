# Search-R1 ↔ VideoQAgent：GRPO/多轮逻辑与 vLLM rollout 的新理解

本文补充说明 **S(Search-R1, verl 0.1.0)** 与 **V(VideoQAgent, verl 0.7.0dev1)** 在 GRPO、多轮逻辑以及 vLLM rollout 的实现差异，重点回答：

- **repeat 发生在哪里？用的是 `n` 还是 `n_agent`？**
- **run_llm_loop 与 Ray trainer 在 GRPO 场景下如何配合？**
- **vLLM rollout 中是否仍使用 `n=1`，以及多样本如何实现？**
- **多轮 information mask 是否仍保留？**

> 注意：本文为理解记录，不修改代码。

---

## 1) Search-R1（旧版 VERL 0.1.0）

### 1.1 GRPO/多轮逻辑配合方式（repeat 发生在 trainer）

**关键路径**：`Search-R1/verl/trainer/ppo/ray_trainer.py`

- 训练循环中，在生成之前就做了一次 **`n_agent` 级别的 repeat**：
  ```python
  batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
  ```
  这一步把一个 prompt 复制成多条“agent 轨迹”。

- 生成完成后，会再 **repeat `rollout.n`**：
  ```python
  batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
  ```
  这一步是 PPO/GRPO 里常规的“多采样数”对齐。

**结论（S 中的 repeat 位置）**
- **`n_agent` 的 repeat 在 trainer 内，而不是 `run_llm_loop` 内。**
- **`rollout.n` 的 repeat 也在 trainer 内。**

### 1.2 run_llm_loop 的职责（只生成，不 repeat）

**关键文件**：`Search-R1/search_r1/llm_agent/generation.py`

- `LLMGenerationManager.run_llm_loop()` 负责：
  - 多轮执行（解析 `<search>` / `<answer>`）
  - 生成 `responses` / `responses_with_info_mask`
  - 拼接 `input_ids`, `attention_mask`, `position_ids`
  - 记录 `info_mask`，供后续 state masking 使用

**没有发现任何 repeat 逻辑在 `run_llm_loop` 内部。**

### 1.3 vLLM rollout 的采样数

**关键文件**：`Search-R1/verl/workers/rollout/vllm_rollout/vllm_rollout.py`

- vLLM 初始化时 `SamplingParams(n=1)`：
  ```python
  kwargs = dict(n=1, logprobs=1, max_tokens=config.response_length)
  ```
- 只有当 `config.n > 1` 时，`vLLMRollout.generate_sequences()` 会做内部 repeat：
  ```python
  if self.config.n > 1 and do_sample:
      idx = idx.repeat_interleave(self.config.n, dim=0)
      ...
  ```

**在 S 的实际用法中**：
- 作者更倾向于 **外部 repeat（trainer）+ vLLM 内 n=1**。
- 也就是“手动 repeat prompt → 每条 prompt 只生成 1 个 response”。

### 1.4 GRPO 分组与 `uid`

- 在 `do_search=true` 路径，`uid` 设置为 `batch.non_tensor_batch['index']`：
  ```python
  batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
  ```
- 这样 GRPO 能以 **原始样本 id** 为组，保证多次采样属于同一组。

---

## 2) VideoQAgent（新版 VERL 0.7.0dev1）

### 2.1 GRPO/多轮逻辑配合方式（repeat 发生在 trainer）

**关键路径**：`VideoQAgent/verl/trainer/ppo/ray_trainer.py`

- 生成前并没有 `n_agent`（新版本不再使用 `n_agent`）。
- 但生成前会构造 `gen_batch_output = gen_batch.repeat(rollout.n)`，再送入 multi-turn：
  ```python
  gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
  gen_batch_output = generation_manager.run_llm_loop(...)
  ```
- 生成后再次 repeat 同样是 `rollout.n`：
  ```python
  batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
  ```

**结论（V 中的 repeat 位置）**
- **统一使用 `rollout.n`**，没有 `n_agent`。
- repeat 发生在 trainer 内，`run_llm_loop` 仍是纯生成逻辑。

### 2.2 run_llm_loop 的职责（只生成，不 repeat）

**关键文件**：`VideoQAgent/videoagent/action_generation.py`

- `VisionLLMGenerationManager.run_llm_loop()` 负责多轮逻辑与工具调用，输出 `DataProto`。
- 与 Search-R1 类似，**没有 repeat 逻辑**。

### 2.3 vLLM rollout 的采样数

**关键文件**：`VideoQAgent/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

- `SamplingParams` 显式写死：
  ```python
  kwargs["n"] = 1  # already repeat in ray_trainer
  ```
- 说明 **新版本默认认为重复由 trainer 负责，而不是 vLLM 自己采样 n>1**。

**结论**
- **V 中 vLLM rollout 固定 n=1。**
- 多采样完全由 trainer 的 `rollout.n` repeat 实现。

---

## 3) 关于 `n_agent` vs `rollout.n` 的定位

| 维度 | Search-R1 (S) | VideoQAgent (V) |
|------|---------------|-----------------|
| 采样数量 | `rollout.n` | `rollout.n` |
| agent 数量 | `rollout.n_agent` | **无** |
| repeat 位置 | trainer 内 (两次 repeat) | trainer 内 (统一 repeat) |
| vLLM 内部 n | 可变，但通常 n=1 | 固定 n=1 |

**理解要点**
- Search-R1 的作者更像是 **“手动 repeat + vLLM n=1”** 来模拟 GRPO 多样本。
- 新版 VERL 明确把 GRPO 的采样控制放在 **`rollout.n`**，完全不靠 vLLM 的 `n`。

---

## 4) 多轮 information mask 的保留

**Search-R1**
- `LLMGenerationManager._compose_final_output()` 会生成：
  - `info_mask`（基于 `responses_with_info_mask`）
  - `attention_mask`

**VideoQAgent**
- 也保留 `responses_with_info_mask` 和 info mask 的逻辑（用于 state masking / loss masking）。

**结论**
- 升级到新 VERL 时，必须保证：
  - 多轮 observation（`<information>` / grounding 信息）仍通过 `info_mask` 记录。
  - trainer 在 `state_masking` 逻辑里使用该 mask 不被破坏。

---

## 5) 升级时必须注意的点（新增理解）

1. **`n_agent` 的含义只存在于旧版 Search-R1**
   - 新版应统一改为 `rollout.n`。
   - 如果保留 `n_agent`，需要明确它是“多轨迹模拟”还是“多样本采样”。

2. **S 中的“多轮 + GRPO”是 trainer 级别实现**
   - `run_llm_loop()` 不 repeat。
   - 旧版通过 repeat prompt 来实现多次采样。

3. **V 中的 GRPO 采样完全依赖 `rollout.n`**
   - vLLM rollout 内部固定 `n=1`。
   - `rollout.n` 负责多采样。

4. **info_mask 必须保留**
   - Search-R1 与 VideoQAgent 都使用 `responses_with_info_mask`。
   - 升级时需确保 `info_mask` 逻辑与 state masking 继续一致。

---

## 6) 简短结论

- **Search-R1 旧版是“trainer repeat + vLLM n=1”模式**，其中 `n_agent` 额外引入多轨迹逻辑。
- **VideoQAgent 新版是“trainer repeat (rollout.n) + vLLM n=1”模式**，完全弃用 `n_agent`。
- **run_llm_loop 只负责多轮生成，不做 repeat。**
- **升级到新 VERL 时，优先统一成 `rollout.n`，并保留 info mask。**
