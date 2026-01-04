# Copyright 2025 [Your Name or Organization]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import torch
from typing import Any, Tuple, Dict
from omegaconf import DictConfig
import ray
from verl import DataProto


def compute_tvqa_score(solution_str: str, ground_truth: str, format_score: float = 0.0, one_turn: bool = True) -> Tuple[float, float]:
    """
    计算 TVQA+ 任务的分数。

    Args:
        solution_str: 模型生成的完整回答（包含 prompt）
        ground_truth: 正确答案（如 "a1"）
        format_score: 格式分数（当模型没有按格式回答时给予的分数，默认为 0.0）
        one_turn: 是否为一轮问题（True 为 T=1 类，False 为 T>1 类）

    Returns:
        Tuple[float, float]: 主分数和第二个分数，基于以下规则：
            - <reasoning>...</reasoning> 标签存在：+0.2
            - <answer>...</answer> 标签存在：+0.2
            - 答案正确：+1.0
            - T>1 时，如果答案错误且使用了工具（<search> 或 <request_grounding>）：+0.5
    """

    response = solution_str
    total_score = 0.0
    second_score = 0.0  # 初始化第二个分数
    reasoning_match = re.search(r"<reasoning>.*?</reasoning>", response, re.DOTALL)
    if reasoning_match:
        total_score += 0.2

    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not answer_match:
        return total_score, second_score

    total_score += 0.2
    predicted_answer = answer_match.group(1).strip()

    predicted_num_match = re.search(r'(\d+)', predicted_answer)
    ground_truth_num_match = re.search(r'(\d+)', ground_truth)

    predicted_num = predicted_num_match.group(1) if predicted_num_match else None
    ground_truth_num = ground_truth_num_match.group(1) if ground_truth_num_match else None

    answer_correct = False
    if predicted_num is not None and ground_truth_num is not None:
        if predicted_num == ground_truth_num:
            total_score += 1.0
            answer_correct = True
    else:
        print(f"Warning: Unable to extract numbers for comparison - predicted: {predicted_answer}, ground_truth: {ground_truth}")

    if not one_turn:
        used_tool = re.search(r"<search>", response) or re.search(r"<request_grounding>", response)
        if used_tool:
            second_score += 1  # 修复并添加 +1 到第二个分数

    return total_score, second_score


def _select_rm_score_fn(data_source: str) -> callable:
    """
    根据数据源选择合适的奖励函数。

    Args:
        data_source: 数据源名称。

    Returns:
        callable: 对应的奖励函数，返回 Tuple[float, float]。
    """
    if data_source == 'tvqa_plus_vision':
        return compute_tvqa_score
    elif data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        from verl.utils.reward_score import qa_em
        def wrapped_fn(solution_str, ground_truth, format_score=0.0, one_turn=True):
            main_score = qa_em.compute_score_em(solution_str, ground_truth, format_score, one_turn)
            second_score = 1.0 if solution_str.strip() == ground_truth.strip() else 0.0  # 示例：简单 exact match 作为第二个分数
            return main_score, second_score
        return wrapped_fn
    else:
        print(f"Warning: Unknown data source '{data_source}', using default score 0.0")
        return lambda solution_str, ground_truth, format_score=0.0, one_turn=True: (0.0, 0.0)


class CustomRewardManager:
    """自定义奖励管理器，完全复制旧版本逻辑。"""

    def __init__(self, tokenizer, num_examine, format_score=0.0, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score

    def __call__(self, data: DataProto, return_dict=False, **kwargs):
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {
                    'reward_tensor': data.batch['rm_scores'],
                    'reward_extra_info': {}
                }
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        all_main_scores = []
        all_second_scores = []
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[:valid_prompt_length]

            response_ids = data_item.batch['responses']
            response_mask = data_item.batch['attention_mask'][prompt_length:]
            valid_response_ids = response_ids[response_mask.bool()]  # ← 关键修复
            valid_response_length = valid_response_ids.shape[0]       # 仍等于 sum

            # sequences_str = self.tokenizer.decode(torch.cat((prompt_ids, response_ids)), skip_special_tokens=True)  # full decode for compare 
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=True)



            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            one_turn = data_item.non_tensor_batch['reward_model'].get('one_turn', True)

            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score_tuple = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                format_score=self.format_score,
                one_turn=one_turn
            )
            main_score, second_score = score_tuple if isinstance(score_tuple, tuple) else (score_tuple, 0.0)

            reward_tensor[i, valid_response_length - 1] = main_score
            all_main_scores.append(main_score)
            all_second_scores.append(second_score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                if data_source == 'tvqa_plus_vision':
                    print(f"\n🔍 Sample {i} (Data source: {data_source}):")
                    print(f"Ground truth: {ground_truth}")
                    print(f"Response: {sequences_str}")
                    print(f"Main Score: {main_score}, Second Score: {second_score}")
                else:
                    print(f"Sample {i} (Data source: {data_source}): {sequences_str}")

        reward_extra_info = {'all_main_scores': all_main_scores, 'all_second_scores': all_second_scores}

        if return_dict:
            return {
                'reward_tensor': reward_tensor,
                'reward_extra_info': reward_extra_info
            }
        return reward_tensor


def load_reward_manager(config: DictConfig, tokenizer: Any, num_examine: int, **reward_kwargs: Any) -> Any:
    """
    加载自定义奖励管理器。

    Args:
        config: PPO 训练器配置对象。
        tokenizer: 分词器对象。
        num_examine: 打印调试信息的样本数量。
        **reward_kwargs: 奖励管理器的额外参数。

    Returns:
        CustomRewardManager: 自定义奖励管理器实例。
    """
    print("✅ Loading CustomRewardManager for TVQA+ support")
    return CustomRewardManager(
        tokenizer=tokenizer,
        num_examine=num_examine,
        format_score=reward_kwargs.get('format_score', 0.0),
        **reward_kwargs
    )


def compute_reward(data: DataProto, reward_fn: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    为一批数据计算奖励。

    Args:
        data: DataProto 对象。
        reward_fn: 奖励函数（CustomRewardManager 实例）。

    Returns:
        Tuple[torch.Tensor, Dict[str, Any]]: 奖励张量和额外信息字典。
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result['reward_tensor']
        reward_extra_info = reward_result.get('reward_extra_info', {})
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_info = {}

    return reward_tensor, reward_extra_info


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None):
    """
    异步计算一批数据的奖励。

    Args:
        data: DataProto 对象。
        config: 配置对象（可选）。
        tokenizer: 分词器对象（可选）。
        reward_fn: 奖励函数（CustomRewardManager 实例）。

    Returns:
        Tuple[torch.Tensor, Dict[str, Any]]: 奖励张量和额外信息字典。
    """
    if reward_fn is None:
        assert config is not None and tokenizer is not None
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )

    return compute_reward(data, reward_fn)