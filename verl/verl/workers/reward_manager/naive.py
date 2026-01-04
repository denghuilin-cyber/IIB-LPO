# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # 🔧 修复：正确提取 reward_model.ground_truth
            # reward_model 在dataset中是dict，collate后变成numpy array of dicts
            reward_model_entry = data_item.non_tensor_batch.get("reward_model", None)
            ground_truth = None
            
            if reward_model_entry is not None:
                # DEBUG: 打印 reward_model 的类型（仅打印前几个样本）
                if i < 3:
                    print(f"[DEBUG] Sample {i}: reward_model_entry type = {type(reward_model_entry)}")
                    print(f"[DEBUG] Sample {i}: reward_model_entry value = {reward_model_entry}")
                
                # 如果是 dict 类型（dataset直接返回的）
                if isinstance(reward_model_entry, dict):
                    ground_truth = reward_model_entry.get("ground_truth", None)
                # 否则，可能是经过 collate 后的某种格式，直接当作 ground_truth 使用
                else:
                    ground_truth = reward_model_entry
            
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"\n{'='*80}")
                print(f"[Reward Verification] Sample {already_print_data_sources[data_source]} of {data_source}")
                print(f"{'='*80}")
                print("[prompt]", prompt_str[:200] + "..." if len(prompt_str) > 200 else prompt_str)
                print("[response]", response_str[:300] + "..." if len(response_str) > 300 else response_str)
                print(f"[ground_truth] type: {type(ground_truth)}, value: {ground_truth}")
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
                print(f"{'='*80}\n")
                
                # 🎯 同时保存到文件（不依赖配置）
                try:
                    import json
                    import os
                    from datetime import datetime
                    
                    # 保存到 /tmp 目录，确保一定能写入
                    log_dir = "/nas/dhl/temp/verl_reward_verification"
                    os.makedirs(log_dir, exist_ok=True)
                    log_file = os.path.join(log_dir, f"rewards_{datetime.now().strftime('%Y%m%d')}.jsonl")
                    
                    # 🔧 简化并强制处理 ground_truth 的类型转换
                    gt_for_save = None
                    if ground_truth is not None:
                        try:
                            if hasattr(ground_truth, 'tolist'):  # tensor or numpy array
                                gt_for_save = ground_truth.tolist()
                            elif hasattr(ground_truth, 'item'):  # numpy scalar
                                gt_for_save = ground_truth.item()
                            elif isinstance(ground_truth, bytes):
                                gt_for_save = ground_truth.decode('utf-8', errors='ignore')
                            else:
                                gt_for_save = str(ground_truth)
                        except Exception as e:
                            print(f"⚠️ Error converting ground_truth: {e}")
                            gt_for_save = str(ground_truth)  # Fallback to string representation
                    
                    record = {
                        "timestamp": datetime.now().isoformat(),
                        "data_source": data_source,
                        "prompt": prompt_str,
                        "response": response_str,
                        "ground_truth": gt_for_save,
                        "score": score if not isinstance(score, dict) else score.get("score", score),
                    }
                    
                    with open(log_file, "a") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
                    if already_print_data_sources[data_source] == 1:
                        print(f"💾 Reward logs saved to: {log_file}")
                except Exception as e:
                    print(f"⚠️ Failed to save reward log: {e}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
