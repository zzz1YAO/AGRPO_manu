# Copyright 2025 Search-R1 Contributors
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

from typing import Any, Dict

import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.utils.reward_score import search_r1_like_qa_em


def _select_rm_score_fn(data_source: str) -> callable:
    if data_source in ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]:
        return search_r1_like_qa_em.compute_score
    raise NotImplementedError(f"Unsupported data_source: {data_source}")


class SearchR1RewardManager:
    """Reward manager for Search-R1 data sources."""

    def __init__(self, tokenizer, num_examine: int, format_score: float = 0.0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score

    def __call__(self, data: DataProto, return_dict: bool = False, **kwargs: Any):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": {}}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print_data_sources: Dict[str, int] = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            prompt_mask = data_item.batch["attention_mask"][:prompt_length].bool()
            valid_prompt_ids = prompt_ids[prompt_mask]

            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["attention_mask"][prompt_length:].bool()
            valid_response_ids = response_ids[response_mask]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch["data_source"]
            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                format_score=self.format_score,
            )

            valid_response_length = valid_response_ids.shape[0]
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
        return reward_tensor


def load_reward_manager(config: DictConfig, tokenizer: Any, num_examine: int, **reward_kwargs: Any) -> Any:
    format_score = reward_kwargs.get("format_score", 0.0)
    return SearchR1RewardManager(tokenizer=tokenizer, num_examine=num_examine, format_score=format_score)
