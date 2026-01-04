# # Copyright 2024 Bytedance Ltd. and/or its affiliates
# # Copyright 2023-2024 SGLang Team
# # Copyright 2025 ModelBest Inc. and/or its affiliates
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """
# PPO Trainer with Ray-based single controller.
# This trainer supports model-agonistic model initialization with huggingface
# """

# import json
# import os
# import uuid
# from collections import defaultdict
# from copy import deepcopy
# from dataclasses import dataclass, field
# from pprint import pprint
# from typing import Optional
# import logging

# import numpy as np
# import ray
# import torch
# from omegaconf import OmegaConf, open_dict
# from torch.utils.data import Dataset, Sampler
# from torchdata.stateful_dataloader import StatefulDataLoader
# from tqdm import tqdm

# from verl import DataProto
# from verl.experimental.dataset.sampler import AbstractCurriculumSampler
# from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
# from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
# from verl.single_controller.ray.base import create_colocated_worker_cls
# from verl.trainer.config import AlgoConfig
# from verl.trainer.ppo import core_algos
# from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
# from verl.trainer.ppo.metric_utils import (
#     compute_data_metrics,
#     compute_throughout_metrics,
#     compute_timing_metrics,
#     process_validation_metrics,
# )
# from verl.trainer.ppo.reward import compute_reward, compute_reward_async
# from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
# from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
# from verl.utils.config import omega_conf_to_dataclass
# from verl.utils.debug import marked_timer
# from verl.utils.metric import reduce_metrics
# from verl.utils.rollout_skip import RolloutSkip
# from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
# from verl.utils.torch_functional import masked_mean
# from verl.utils.tracking import ValidationGenerationsLogger


# logger = logging.getLogger(__file__)
# logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# @dataclass
# class ResourcePoolManager:
#     """
#     Define a resource pool specification. Resource pool will be initialized first.
#     """

#     resource_pool_spec: dict[str, list[int]]
#     mapping: dict[Role, str]
#     resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

#     def create_resource_pool(self):
#         """Create Ray resource pools for distributed training.

#         Initializes resource pools based on the resource pool specification,
#         with each pool managing GPU resources across multiple nodes.
#         For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
#         For Megatron backend, uses max_colocate_count>1 for different models.
#         """
#         for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
#             # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
#             # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
#             # For Megatron backend, we recommend using max_colocate_count>1
#             # that can utilize different WorkerGroup for differnt models
#             resource_pool = RayResourcePool(
#                 process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
#             )
#             self.resource_pool_dict[resource_pool_name] = resource_pool

#         self._check_resource_available()

#     def get_resource_pool(self, role: Role) -> RayResourcePool:
#         """Get the resource pool of the worker_cls"""
#         return self.resource_pool_dict[self.mapping[role]]

#     def get_n_gpus(self) -> int:
#         """Get the number of gpus in this cluster."""
#         return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

#     def _check_resource_available(self):
#         """Check if the resource pool can be satisfied in this ray cluster."""
#         node_available_resources = ray._private.state.available_resources_per_node()
#         node_available_gpus = {
#             node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
#             for node, node_info in node_available_resources.items()
#         }

#         # check total required gpus can be satisfied
#         total_available_gpus = sum(node_available_gpus.values())
#         total_required_gpus = sum(
#             [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
#         )
#         if total_available_gpus < total_required_gpus:
#             raise ValueError(
#                 f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
#             )

#         # check each resource pool can be satisfied, O(#resource_pools * #nodes)
#         for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
#             num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
#             for node, available_gpus in node_available_gpus.items():
#                 if available_gpus >= num_gpus:
#                     node_available_gpus[node] -= num_gpus
#                     num_nodes -= 1
#                     if num_nodes == 0:
#                         break
#             if num_nodes > 0:
#                 raise ValueError(
#                     f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
#                     + "cannot be satisfied in this ray cluster"
#                 )


# def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
#     """Apply KL penalty to the token-level rewards.

#     This function computes the KL divergence between the reference policy and current policy,
#     then applies a penalty to the token-level rewards based on this divergence.

#     Args:
#         data (DataProto): The data containing batched model outputs and inputs.
#         kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
#         kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

#     Returns:
#         tuple: A tuple containing:
#             - The updated data with token-level rewards adjusted by KL penalty
#             - A dictionary of metrics related to the KL penalty
#     """
#     response_mask = data.batch["response_mask"]
#     token_level_scores = data.batch["token_level_scores"]
#     batch_size = data.batch.batch_size[0]

#     # compute kl between ref_policy and current policy
#     # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
#     kld = core_algos.kl_penalty(
#         data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
#     )  # (batch_size, response_length)
#     kld = kld * response_mask
#     beta = kl_ctrl.value

#     token_level_rewards = token_level_scores - beta * kld

#     current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
#     current_kl = torch.mean(current_kl, dim=0).item()

#     # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
#     kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
#     data.batch["token_level_rewards"] = token_level_rewards

#     metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

#     return data, metrics


# def compute_response_mask(data: DataProto):
#     """Compute the attention mask for the response part of the sequence.

#     This function extracts the portion of the attention mask that corresponds to the model's response,
#     which is used for masking computations that should only apply to response tokens.

#     Args:
#         data (DataProto): The data containing batched model outputs and inputs.

#     Returns:
#         torch.Tensor: The attention mask for the response tokens.
#     """
#     responses = data.batch["responses"]
#     response_length = responses.size(1)
#     attention_mask = data.batch["attention_mask"]
#     return attention_mask[:, -response_length:]


# def compute_advantage(
#     data: DataProto,
#     adv_estimator: AdvantageEstimator,
#     gamma: float = 1.0,
#     lam: float = 1.0,
#     num_repeat: int = 1,
#     norm_adv_by_std_in_grpo: bool = True,
#     config: Optional[AlgoConfig] = None,
# ) -> DataProto:
#     """Compute advantage estimates for policy optimization.

#     This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
#     The advantage estimates are used to guide policy optimization in RL algorithms.

#     Args:
#         data (DataProto): The data containing batched model outputs and inputs.
#         adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
#         gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
#         lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
#         num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
#         norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
#             GRPO. Defaults to True.
#         config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

#     Returns:
#         DataProto: The updated data with computed advantages and returns.
#     """
#     # Back-compatible with trainers that do not compute response mask in fit
#     if "response_mask" not in data.batch.keys():
#         data.batch["response_mask"] = compute_response_mask(data)
#     # prepare response group
#     if adv_estimator == AdvantageEstimator.GAE:
#         # Compute advantages and returns using Generalized Advantage Estimation (GAE)
#         advantages, returns = core_algos.compute_gae_advantage_return(
#             token_level_rewards=data.batch["token_level_rewards"],
#             values=data.batch["values"],
#             response_mask=data.batch["response_mask"],
#             gamma=gamma,
#             lam=lam,
#         )
#         data.batch["advantages"] = advantages
#         data.batch["returns"] = returns
#         if config.get("use_pf_ppo", False):
#             data = core_algos.compute_pf_ppo_reweight_data(
#                 data,
#                 config.pf_ppo.get("reweight_method"),
#                 config.pf_ppo.get("weight_pow"),
#             )
#     elif adv_estimator == AdvantageEstimator.GRPO:
#         # Initialize the mask for GRPO calculation
#         grpo_calculation_mask = data.batch["response_mask"]
#         # Call compute_grpo_outcome_advantage with parameters matching its definition
#         advantages, returns = core_algos.compute_grpo_outcome_advantage(
#             token_level_rewards=data.batch["token_level_rewards"],
#             response_mask=grpo_calculation_mask,
#             index=data.non_tensor_batch["uid"],
#             norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
#         )
#         data.batch["advantages"] = advantages
#         data.batch["returns"] = returns
#     else:
#         # handle all other adv estimator type other than GAE and GRPO
#         adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
#         adv_kwargs = {
#             "token_level_rewards": data.batch["token_level_rewards"],
#             "response_mask": data.batch["response_mask"],
#             "config": config,
#         }
#         if "uid" in data.non_tensor_batch:  # optional
#             adv_kwargs["index"] = data.non_tensor_batch["uid"]
#         if "reward_baselines" in data.batch:  # optional
#             adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

#         # calculate advantage estimator
#         advantages, returns = adv_estimator_fn(**adv_kwargs)
#         data.batch["advantages"] = advantages
#         data.batch["returns"] = returns
#     return data


# class RayPPOTrainer:
#     """Distributed PPO trainer using Ray for scalable reinforcement learning.

#     This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
#     managing actor rollouts, critic training, and reward computation with Ray backend.
#     Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
#     """

#     # TODO: support each role have individual ray_worker_group_cls,
#     # i.e., support different backend of different role
#     def __init__(
#         self,
#         config,
#         tokenizer,
#         role_worker_mapping: dict[Role, WorkerType],
#         resource_pool_manager: ResourcePoolManager,
#         ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
#         processor=None,
#         reward_fn=None,
#         val_reward_fn=None,
#         train_dataset: Optional[Dataset] = None,
#         val_dataset: Optional[Dataset] = None,
#         collate_fn=None,
#         train_sampler: Optional[Sampler] = None,
#         device_name=None,
#     ):
#         """
#         Initialize distributed PPO trainer with Ray backend.
#         Note that this trainer runs on the driver process on a single CPU/GPU node.

#         Args:
#             config: Configuration object containing training parameters.
#             tokenizer: Tokenizer used for encoding and decoding text.
#             role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
#             resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
#             ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
#             processor: Optional data processor, used for multimodal data
#             reward_fn: Function for computing rewards during training.
#             val_reward_fn: Function for computing rewards during validation.
#             train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
#             val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
#             collate_fn: Function to collate data samples into batches.
#             train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
#             device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
#         """

#         # Store the tokenizer for text processing
#         self.tokenizer = tokenizer
#         self.processor = processor
#         self.config = config
#         self.reward_fn = reward_fn
#         self.val_reward_fn = val_reward_fn

#         self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
#         assert self.hybrid_engine, "Currently, only support hybrid engine"
        
#         # Initialize COT augmenter for GRPO if enabled
#         self._init_cot_augmenter()

#         if self.hybrid_engine:
#             assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

#         self.role_worker_mapping = role_worker_mapping
#         self.resource_pool_manager = resource_pool_manager
#         self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
#         self.use_rm = need_reward_model(self.role_worker_mapping)
#         self.use_critic = need_critic(self.config)
#         self.ray_worker_group_cls = ray_worker_group_cls
#         self.device_name = device_name if device_name else self.config.trainer.device
#         self.validation_generations_logger = ValidationGenerationsLogger(
#             project_name=self.config.trainer.project_name,
#             experiment_name=self.config.trainer.experiment_name,
#         )

#         # if ref_in_actor is True, the reference policy will be actor without lora applied
#         self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

#         # define in-reward KL control
#         # kl loss control currently not suppoorted
#         if self.config.algorithm.use_kl_in_reward:
#             self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

#         self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

#     def _init_cot_augmenter(self):
#         """Initialize the COT augmenter for GRPO if configured."""
#         from verl.utils.grpo_cot_augmentation import GRPOCOTAugmenter, load_cot_examples_from_file
        
#         # Check if COT augmentation is enabled in config
#         cot_config = self.config.actor_rollout_ref.rollout.get("cot_augmentation", None)
        
#         if cot_config is None or not cot_config.get("enable", False):
#             self.cot_augmenter = None
#             print("COT augmentation is disabled.")
#             return
        
#         # Load COT examples
#         cot_examples = None
#         cot_examples_getter = None
        
#         # ⭐ NEW: Multi-dataset COT support (simple matching)
#         if cot_config.get("use_multi_dataset", False) or cot_config.get("dataset_cot_mapping", None):
#             print("Initializing multi-dataset simple COT loader...")
            
#             try:
#                 import sys
#                 import os
#                 # Add examples directory to path
#                 # Go up 4 levels: ppo -> trainer -> verl -> project_root
#                 examples_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "examples", "grpo_trainer")
#                 if examples_path not in sys.path:
#                     sys.path.insert(0, examples_path)
                
#                 from multi_dataset_simple_loader import initialize_multi_dataset_simple_cot_loader, get_multi_dataset_simple_cot_examples
                
#                 # Get COT file mapping from config
#                 cot_file_mapping = cot_config.get("dataset_cot_mapping", cot_config.get("cot_file_mapping", {}))
#                 if isinstance(cot_file_mapping, str):
#                     # If it's a string, try to parse it as JSON
#                     import json
#                     cot_file_mapping = json.loads(cot_file_mapping)
                
#                 if not cot_file_mapping:
#                     raise ValueError("dataset_cot_mapping is required for multi-dataset COT but not provided")
                
#                 # Initialize multi-dataset COT loader
#                 cot_loader = initialize_multi_dataset_simple_cot_loader(
#                     cot_file_mapping=cot_file_mapping,
#                     cot_format_template=cot_config.get("cot_format_template",
#                         "Here is a reference example that demonstrates the problem-solving approach:\n\n<Example>\nQuestion: {question}\n\nStep-by-step Solution:\n{rationale}\n\nFinal Answer: {final_answer}\n</Example>\n\nNow, please solve the following problem using similar reasoning:"),
#                     use_full_cot=cot_config.get("use_full_cot", True),
#                     skip_on_mismatch=cot_config.get("skip_on_mismatch", True),
#                     verbose=cot_config.get("verbose", False),
#                 )
                
#                 # Wrap the getter function
#                 def multi_cot_getter_wrapper(batch, prompt_idx, num_repeats):
#                     return get_multi_dataset_simple_cot_examples(batch, prompt_idx, num_repeats, tokenizer=self.tokenizer)
                
#                 cot_examples_getter = multi_cot_getter_wrapper
#                 print(f"Multi-dataset simple COT loader initialized successfully")
#                 print(f"Loaded COT data for {len(cot_file_mapping)} datasets: {list(cot_file_mapping.keys())}")
                
#             except Exception as e:
#                 import traceback
#                 print(f"Error: Failed to initialize multi-dataset COT loader: {e}")
#                 traceback.print_exc()
#                 raise
        
#         # Special handling for GSM8K-style COT files with selected_cots (single dataset)
#         elif cot_config.get("cot_file_path", None):
#             # This is the GSM8K format with JSONL containing selected_cots
#             print(f"Initializing GSM8K COT loader from {cot_config.cot_file_path}")
            
#             # Import and initialize the GSM8K COT loader
#             try:
#                 from verl.utils.import_utils import load_extern_type
#                 # Try to load custom GSM8K COT loader if user provided it
#                 if cot_config.get("loader_path", None):
#                     loader_module = load_extern_type(
#                         cot_config.loader_path,
#                         "initialize_gsm8k_cot_loader"
#                     )
#                     cot_loader = loader_module(
#                         cot_file_path=cot_config.cot_file_path,
#                         cot_format_template=cot_config.get("cot_format_template", 
#                             "Here is a reference example that demonstrates the problem-solving approach:\n\n<Example>\nQuestion: {question}\n\nStep-by-step Solution:\n{rationale}\n\nFinal Answer: {final_answer}\n</Example>\n\nNow, please solve the following problem using similar reasoning:"),
#                         match_by=cot_config.get("match_by", "question"),
#                         use_full_cot=cot_config.get("use_full_cot", True),
#                     )
#                     getter_func = load_extern_type(cot_config.loader_path, "get_gsm8k_cot_examples")
#                 else:
#                     # Use the default GSM8K COT loader from examples
#                     import sys
#                     import os
#                     # Add examples directory to path
#                     # Go up 4 levels: ppo -> trainer -> verl -> project_root
#                     examples_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "examples", "grpo_trainer")
#                     if examples_path not in sys.path:
#                         sys.path.insert(0, examples_path)
                    
#                     from gsm8k_cot_loader import initialize_gsm8k_cot_loader, get_gsm8k_cot_examples
                    
#                     cot_loader = initialize_gsm8k_cot_loader(
#                         cot_file_path=cot_config.cot_file_path,
#                         cot_format_template=cot_config.get("cot_format_template",
#                             "Here is a reference example that demonstrates the problem-solving approach:\n\n<Example>\nQuestion: {question}\n\nStep-by-step Solution:\n{rationale}\n\nFinal Answer: {final_answer}\n</Example>\n\nNow, please solve the following problem using similar reasoning:"),
#                         match_by=cot_config.get("match_by", "question"),
#                         use_full_cot=cot_config.get("use_full_cot", True),
#                     )
#                     getter_func = get_gsm8k_cot_examples
                
#                 # Wrap the getter function to match expected signature
#                 def cot_getter_wrapper(batch, prompt_idx, num_repeats):
#                     return getter_func(batch, prompt_idx, num_repeats, tokenizer=self.tokenizer)
                
#                 cot_examples_getter = cot_getter_wrapper
#                 print(f"GSM8K COT loader initialized successfully")
                
#             except Exception as e:
#                 print(f"Warning: Failed to initialize GSM8K COT loader: {e}")
#                 print("Falling back to standard examples_file loading")
#                 # Fallback to standard file loading
#                 cot_examples = load_cot_examples_from_file(cot_config.cot_file_path)
#                 print(f"Loaded {len(cot_examples)} COT examples from {cot_config.cot_file_path}")
        
#         elif cot_config.get("examples_file", None):
#             # Load from file (simple text/json format)
#             cot_examples = load_cot_examples_from_file(cot_config.examples_file)
#             print(f"Loaded {len(cot_examples)} COT examples from {cot_config.examples_file}")
#         elif cot_config.get("examples", None):
#             # Use examples from config
#             cot_examples = list(cot_config.examples)
#             print(f"Using {len(cot_examples)} COT examples from config")
#         elif cot_config.get("examples_getter", None):
#             # Use custom getter function
#             from verl.utils.import_utils import load_extern_type
#             getter_path = cot_config.examples_getter.get("path")
#             getter_name = cot_config.examples_getter.get("name")
#             cot_examples_getter = load_extern_type(getter_path, getter_name)
#             print(f"Using custom COT examples getter: {getter_name} from {getter_path}")
#         else:
#             raise ValueError(
#                 "COT augmentation is enabled but no examples provided. "
#                 "Please specify 'cot_file_path', 'examples_file', 'examples', or 'examples_getter' in config."
#             )
        
#         # Initialize augmenter
#         self.cot_augmenter = GRPOCOTAugmenter(
#             cot_examples=cot_examples,
#             cot_examples_getter=cot_examples_getter,
#             tokenizer=self.tokenizer,
#             num_repeats=self.config.actor_rollout_ref.rollout.n,
#             sampling_strategy=cot_config.get("sampling_strategy", "sequential"),
#             add_separator=cot_config.get("add_separator", True),
#             separator=cot_config.get("separator", "\n\n"),
#             enable=cot_config.get("enable", False),
#             seed=cot_config.get("seed", None),
#             debug_print_augmented_prompts=cot_config.get("debug_print_augmented_prompts", True),
#             debug_num_samples=cot_config.get("debug_num_samples", 3),
#             debug_print_full_prompt=cot_config.get("debug_print_full_prompt", False),  # 🆕 传递新参数
#         )
#         print(f"COT augmenter initialized with strategy: {cot_config.get('sampling_strategy', 'sequential')}")

#     def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
#         """
#         Creates the train and validation dataloaders.
#         """
#         # TODO: we have to make sure the batch size is divisible by the dp size
#         from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

#         if train_dataset is None:
#             train_dataset = create_rl_dataset(
#                 self.config.data.train_files, self.config.data, self.tokenizer, self.processor, is_train=True
#             )
#         if val_dataset is None:
#             val_dataset = create_rl_dataset(
#                 self.config.data.val_files, self.config.data, self.tokenizer, self.processor, is_train=False
#             )
#         self.train_dataset, self.val_dataset = train_dataset, val_dataset

#         if train_sampler is None:
#             train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
#         if collate_fn is None:
#             from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

#             collate_fn = default_collate_fn

#         num_workers = self.config.data["dataloader_num_workers"]

#         self.train_dataloader = StatefulDataLoader(
#             dataset=self.train_dataset,
#             batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
#             num_workers=num_workers,
#             drop_last=True,
#             collate_fn=collate_fn,
#             sampler=train_sampler,
#         )

#         val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
#         if val_batch_size is None:
#             val_batch_size = len(self.val_dataset)

#         # 🔑 修复：StatefulDataLoader 不支持空数据集，需要特殊处理
#         if len(self.val_dataset) == 0:
#             # 验证集为空时，创建一个虚拟的空 DataLoader
#             print("⚠️ Warning: Validation dataset is empty. Creating a dummy validation dataloader.")
#             # 使用一个简单的空列表作为空 DataLoader 的替代
#             self.val_dataloader = []  # 空列表，len() 返回 0，迭代时不会产生任何 batch
#         else:
#             # 正常情况：创建真实的验证集 DataLoader
#             self.val_dataloader = StatefulDataLoader(
#                 dataset=self.val_dataset,
#                 batch_size=val_batch_size,
#                 num_workers=num_workers,
#                 shuffle=self.config.data.get("validation_shuffle", True),
#                 drop_last=False,
#                 collate_fn=collate_fn,
#             )

#         assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

#         # 打印 DataLoader 信息
#         if len(self.val_dataset) == 0:
#             print(f"Size of train dataloader: {len(self.train_dataloader)}")
#             print(f"⚠️ Validation dataloader is empty (validation will be skipped)")
#         else:
#             print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")
        

#         total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

#         if self.config.trainer.total_training_steps is not None:
#             total_training_steps = self.config.trainer.total_training_steps

#         self.total_training_steps = total_training_steps
#         print(f"Total training steps: {self.total_training_steps}")

#         try:
#             OmegaConf.set_struct(self.config, True)
#             with open_dict(self.config):
#                 if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
#                     self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
#                 if OmegaConf.select(self.config, "critic.optim"):
#                     self.config.critic.optim.total_training_steps = total_training_steps
#         except Exception as e:
#             print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

#     def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
#         """Dump rollout/validation samples as JSONL.
        
#         🎯 用于验证 reward 计算的正确性
#         输出文件: {dump_path}/{step}.jsonl
#         """
#         os.makedirs(dump_path, exist_ok=True)
#         filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

#         n = len(inputs)
#         base_data = {
#             "input": inputs,
#             "output": outputs,
#             "ground_truth": gts,  # 🔧 改名为 ground_truth，更清晰
#             "score": scores,
#             "step": [self.global_steps] * n,
#         }

#         for k, v in reward_extra_infos_dict.items():
#             if len(v) == n:
#                 base_data[k] = v

#         lines = []
#         for i in range(n):
#             entry = {k: v[i] for k, v in base_data.items()}
#             lines.append(json.dumps(entry, ensure_ascii=False))

#         with open(filename, "w") as f:
#             f.write("\n".join(lines) + "\n")

#         print(f"\n{'='*80}")
#         print(f"✅ Dumped {n} generations to: {filename}")
#         print(f"{'='*80}")
        
#         # 🎯 打印前3个样本到控制台，方便快速检查
#         print(f"\n📊 Sample Preview (first 3 of {n}):")
#         for i in range(min(3, n)):
#             print(f"\n--- Sample {i+1} ---")
#             print(f"[data_source]: {base_data.get('data_source', ['N/A'])[i] if 'data_source' in base_data else 'N/A'}")
#             print(f"[input]: {inputs[i][:100]}...")  # 只显示前100个字符
#             print(f"[output]: {outputs[i][:200]}...")  # 只显示前200个字符
#             print(f"[ground_truth]: {gts[i][:100] if gts[i] else 'None'}...")
#             print(f"[score]: {scores[i]}")
#         print(f"{'='*80}\n")

#     def _maybe_log_val_generations(self, inputs, outputs, scores):
#         """Log a table of validation samples to the configured logger (wandb or swanlab)"""

#         generations_to_log = self.config.trainer.log_val_generations

#         if generations_to_log == 0:
#             return

#         import numpy as np

#         # Create tuples of (input, output, score) and sort by input text
#         samples = list(zip(inputs, outputs, scores, strict=True))
#         samples.sort(key=lambda x: x[0])  # Sort by input text

#         # Use fixed random seed for deterministic shuffling
#         rng = np.random.RandomState(42)
#         rng.shuffle(samples)

#         # Take first N samples after shuffling
#         samples = samples[:generations_to_log]

#         # Log to each configured logger
#         self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

#     def _get_gen_batch(self, batch: DataProto) -> DataProto:
#         reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

#         # pop those keys for generation
#         # 🔑 修复：只pop存在的字段，避免position_ids不存在时的AssertionError
#         batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
#         existing_batch_keys = [k for k in batch_keys_to_pop if k in batch.batch.keys()]
        
#         non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
#         gen_batch = batch.pop(
#             batch_keys=existing_batch_keys,  # ← 只pop存在的字段
#             non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
#         )

#         # For agent loop, we need reward model keys to compute score.
#         if self.async_rollout_mode:
#             gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

#         return gen_batch

# # 文件: verl/trainer/ppo/ray_trainer.py
# # 替换函数: RayPPOTrainer._validate

#     def _validate(self):
#         # 🔑 FIX 1: 检查 dataloader 是否为空 (依赖于您的 MultiDatasetWithCOT 类的修复)
#         if len(self.val_dataloader) == 0:
#             print("⚠️ Validation dataloader is empty, skipping validation.")
#             return {}
            
#         data_source_lst = []
#         reward_extra_infos_dict: dict[str, list] = defaultdict(list)

#         # Lists to collect samples for the table
#         sample_inputs = []
#         sample_outputs = []
#         sample_gts = []
#         sample_scores = []
#         sample_turns = []
#         sample_uids = []

#         for test_data in self.val_dataloader:
#             test_batch = DataProto.from_single_dict(test_data)

#             if "uid" not in test_batch.non_tensor_batch:
#                 test_batch.non_tensor_batch["uid"] = np.array(
#                     [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
#                 )

#             # repeat test batch
#             test_batch = test_batch.repeat(
#                 repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
#             )

#             # we only do validation on rule-based rm
#             if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
#                 return {}

#             # Store original inputs
#             input_ids = test_batch.batch["input_ids"]
#             # TODO: Can we keep special tokens except for padding tokens?
#             input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
#             sample_inputs.extend(input_texts)
#             sample_uids.extend(test_batch.non_tensor_batch["uid"])

#             ground_truths = [
#                 item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
#             ]
#             sample_gts.extend(ground_truths)

#             test_gen_batch = self._get_gen_batch(test_batch)
#             test_gen_batch.meta_info = {
#                 "eos_token_id": self.tokenizer.eos_token_id,
#                 "pad_token_id": self.tokenizer.pad_token_id,
#                 "recompute_log_prob": False,
#                 "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
#                 "validate": True,
#                 "global_steps": self.global_steps,
#             }
#             print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

#             # pad to be divisible by dp_size
#             size_divisor = (
#                 self.actor_rollout_wg.world_size
#                 if not self.async_rollout_mode
#                 else self.config.actor_rollout_ref.rollout.agent.num_workers
#             )
#             test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
#             if not self.async_rollout_mode:
#                 test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
#             else:
#                 test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

#             # unpad
#             test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

#             print("validation generation end")

#             # Store generated outputs
#             output_ids = test_output_gen_batch.batch["responses"]
#             output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
#             sample_outputs.extend(output_texts)

#             test_batch = test_batch.union(test_output_gen_batch)
#             test_batch.meta_info["validate"] = True

#             # evaluate using reward_function
#             if self.val_reward_fn is None:
#                 raise ValueError("val_reward_fn must be provided for validation.")
#             result = self.val_reward_fn(test_batch, return_dict=True)
#             reward_tensor = result["reward_tensor"]
#             scores = reward_tensor.sum(-1).cpu().tolist()
#             sample_scores.extend(scores)

#             reward_extra_infos_dict["reward"].extend(scores)
#             print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
#             if "reward_extra_info" in result:
#                 for key, lst in result["reward_extra_info"].items():
#                     reward_extra_infos_dict[key].extend(lst)
#                     print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

#             # collect num_turns of each prompt
#             if "__num_turns__" in test_batch.non_tensor_batch:
#                 sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

#             data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

#         self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

#         # 🔑 FIX 2: 在连接之前检查列表是否为空
#         if not data_source_lst:
#             print("⚠️ Validation data list is empty after processing. Skipping metric computation.")
#             return {}

#         # dump generations
#         val_data_dir = self.config.trainer.get("validation_data_dir", None)
#         if val_data_dir:
#             self._dump_generations(
#                 inputs=sample_inputs,
#                 outputs=sample_outputs,
#                 gts=sample_gts,
#                 scores=sample_scores,
#                 reward_extra_infos_dict=reward_extra_infos_dict,
#                 dump_path=val_data_dir,
#             )

#         for key_info, lst in reward_extra_infos_dict.items():
#             assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

#         data_sources = np.concatenate(data_source_lst, axis=0)

#         data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
#         metric_dict = {}
#         for data_source, var2metric2val in data_src2var2metric2val.items():
#             core_var = "acc" if "acc" in var2metric2val else "reward"
#             for var_name, metric2val in var2metric2val.items():
#                 n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
#                 for metric_name, metric_val in metric2val.items():
#                     if (
#                         (var_name == core_var)
#                         and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
#                         and (f"@{n_max}" in metric_name)
#                     ):
#                         metric_sec = "val-core"
#                     else:
#                         metric_sec = "val-aux"
#                     pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
#                     metric_dict[pfx] = metric_val

#         if len(sample_turns) > 0:
#             sample_turns = np.concatenate(sample_turns)
#             metric_dict["val-aux/num_turns/min"] = sample_turns.min()
#             metric_dict["val-aux/num_turns/max"] = sample_turns.max()
#             metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

#         return metric_dict

#     def init_workers(self):
#         """Initialize distributed training workers using Ray backend.

#         Creates:
#         1. Ray resource pools from configuration
#         2. Worker groups for each role (actor, critic, etc.)
#         """
#         self.resource_pool_manager.create_resource_pool()

#         self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

#         # create actor and rollout
#         if self.hybrid_engine:
#             resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
#             actor_rollout_cls = RayClassWithInitArgs(
#                 cls=self.role_worker_mapping[Role.ActorRollout],
#                 config=self.config.actor_rollout_ref,
#                 role="actor_rollout",
#             )
#             self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
#         else:
#             raise NotImplementedError

#         # create critic
#         if self.use_critic:
#             resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
#             critic_cfg = omega_conf_to_dataclass(self.config.critic)
#             critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
#             self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

#         # create reference policy if needed
#         if self.use_reference_policy:
#             resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
#             ref_policy_cls = RayClassWithInitArgs(
#                 self.role_worker_mapping[Role.RefPolicy],
#                 config=self.config.actor_rollout_ref,
#                 role="ref",
#             )
#             self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

#         # create a reward model if reward_fn is None
#         if self.use_rm:
#             # we create a RM here
#             resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
#             rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
#             self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

#         # initialize WorkerGroup
#         # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
#         # you should not use `create_colocated_worker_cls`.
#         # Instead, directly pass different resource pool to different worker groups.
#         # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
#         all_wg = {}
#         wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
#         if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
#             wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
#         if OmegaConf.select(self.config.global_profiler, "steps") is not None:
#             wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
#             # Only require nsight worker options when tool is nsys
#             if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
#                 assert (
#                     OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
#                     is not None
#                 ), "worker_nsight_options must be set when using nsys with profile_steps"
#                 wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
#                     OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
#                 )
#         wg_kwargs["device_name"] = self.device_name

#         for resource_pool, class_dict in self.resource_pool_to_cls.items():
#             worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
#             wg_dict = self.ray_worker_group_cls(
#                 resource_pool=resource_pool,
#                 ray_cls_with_init=worker_dict_cls,
#                 **wg_kwargs,
#             )
#             spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
#             all_wg.update(spawn_wg)

#         if self.use_critic:
#             self.critic_wg = all_wg["critic"]
#             self.critic_wg.init_model()

#         if self.use_reference_policy and not self.ref_in_actor:
#             self.ref_policy_wg = all_wg["ref"]
#             self.ref_policy_wg.init_model()

#         self.rm_wg = None
#         if self.use_rm:
#             self.rm_wg = all_wg["rm"]
#             self.rm_wg.init_model()

#         # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
#         self.actor_rollout_wg = all_wg["actor_rollout"]
#         self.actor_rollout_wg.init_model()

#         # create async rollout manager and request scheduler
#         self.async_rollout_mode = False
#         if self.config.actor_rollout_ref.rollout.mode == "async":
#             from verl.experimental.agent_loop import AgentLoopManager

#             self.async_rollout_mode = True
#             self.async_rollout_manager = AgentLoopManager(
#                 config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
#             )

#     def _save_checkpoint(self):
#         from verl.utils.fs import local_mkdir_safe

#         # path: given_path + `/global_step_{global_steps}` + `/actor`
#         local_global_step_folder = os.path.join(
#             self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
#         )

#         print(f"local_global_step_folder: {local_global_step_folder}")
#         actor_local_path = os.path.join(local_global_step_folder, "actor")

#         actor_remote_path = (
#             None
#             if self.config.trainer.default_hdfs_dir is None
#             else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
#         )

#         remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
#         if remove_previous_ckpt_in_save:
#             print(
#                 "Warning: remove_previous_ckpt_in_save is deprecated,"
#                 + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
#             )
#         max_actor_ckpt_to_keep = (
#             self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
#         )
#         max_critic_ckpt_to_keep = (
#             self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
#         )

#         self.actor_rollout_wg.save_checkpoint(
#             actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
#         )

#         if self.use_critic:
#             critic_local_path = os.path.join(local_global_step_folder, "critic")
#             critic_remote_path = (
#                 None
#                 if self.config.trainer.default_hdfs_dir is None
#                 else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
#             )
#             self.critic_wg.save_checkpoint(
#                 critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
#             )

#         # save dataloader
#         local_mkdir_safe(local_global_step_folder)
#         dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
#         dataloader_state_dict = self.train_dataloader.state_dict()
#         torch.save(dataloader_state_dict, dataloader_local_path)

#         # latest checkpointed iteration tracker (for atomic usage)
#         local_latest_checkpointed_iteration = os.path.join(
#             self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
#         )
#         with open(local_latest_checkpointed_iteration, "w") as f:
#             f.write(str(self.global_steps))

#     def _load_checkpoint(self):
#         if self.config.trainer.resume_mode == "disable":
#             return 0

#         # load from hdfs
#         if self.config.trainer.default_hdfs_dir is not None:
#             raise NotImplementedError("load from hdfs is not implemented yet")
#         else:
#             checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
#             if not os.path.isabs(checkpoint_folder):
#                 working_dir = os.getcwd()
#                 checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
#             global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

#         # find global_step_folder
#         if self.config.trainer.resume_mode == "auto":
#             if global_step_folder is None:
#                 print("Training from scratch")
#                 return 0
#         else:
#             if self.config.trainer.resume_mode == "resume_path":
#                 assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
#                 assert "global_step_" in self.config.trainer.resume_from_path, (
#                     "resume ckpt must specify the global_steps"
#                 )
#                 global_step_folder = self.config.trainer.resume_from_path
#                 if not os.path.isabs(global_step_folder):
#                     working_dir = os.getcwd()
#                     global_step_folder = os.path.join(working_dir, global_step_folder)
#         print(f"Load from checkpoint folder: {global_step_folder}")
#         # set global step
#         self.global_steps = int(global_step_folder.split("global_step_")[-1])

#         print(f"Setting global step to {self.global_steps}")
#         print(f"Resuming from {global_step_folder}")

#         actor_path = os.path.join(global_step_folder, "actor")
#         critic_path = os.path.join(global_step_folder, "critic")
#         # load actor
#         self.actor_rollout_wg.load_checkpoint(
#             actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
#         )
#         # load critic
#         if self.use_critic:
#             self.critic_wg.load_checkpoint(
#                 critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
#             )

#         # load dataloader,
#         # TODO: from remote not implemented yet
#         dataloader_local_path = os.path.join(global_step_folder, "data.pt")
#         if os.path.exists(dataloader_local_path):
#             dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
#             self.train_dataloader.load_state_dict(dataloader_state_dict)
#         else:
#             print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

#     def _start_profiling(self, do_profile: bool) -> None:
#         """Start profiling for all worker groups if profiling is enabled."""
#         if do_profile:
#             self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
#             if self.use_reference_policy:
#                 self.ref_policy_wg.start_profile(profile_step=self.global_steps)
#             if self.use_critic:
#                 self.critic_wg.start_profile(profile_step=self.global_steps)
#             if self.use_rm:
#                 self.rm_wg.start_profile(profile_step=self.global_steps)

#     def _stop_profiling(self, do_profile: bool) -> None:
#         """Stop profiling for all worker groups if profiling is enabled."""
#         if do_profile:
#             self.actor_rollout_wg.stop_profile()
#             if self.use_reference_policy:
#                 self.ref_policy_wg.stop_profile()
#             if self.use_critic:
#                 self.critic_wg.stop_profile()
#             if self.use_rm:
#                 self.rm_wg.stop_profile()

#     def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
#         """Reorder the data on single controller such that each dp rank gets similar total tokens"""
#         attention_mask = batch.batch["attention_mask"]
#         batch_size = attention_mask.shape[0]
#         global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
#         world_size = self.actor_rollout_wg.world_size
#         global_partition_lst = get_seqlen_balanced_partitions(
#             global_seqlen_lst, k_partitions=world_size, equal_size=True
#         )
#         # reorder based on index. The data will be automatically equally partitioned by dispatch function
#         global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
#         batch.reorder(global_idx)
#         global_balance_stats = log_seqlen_unbalance(
#             seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
#         )
#         metrics.update(global_balance_stats)

#     # 文件: verl/trainer/ppo/ray_trainer.py
# # 替换函数: RayPPOTrainer.fit

#     def fit(self):
#         """
#         The training loop of PPO.
#         The driver process only need to call the compute functions of the worker group through RPC
#         to construct the PPO dataflow.
#         The light-weight advantage computation is done on the driver process.
#         """
#         from omegaconf import OmegaConf

#         from verl.utils.tracking import Tracking

#         logger = Tracking(
#             project_name=self.config.trainer.project_name,
#             experiment_name=self.config.trainer.experiment_name,
#             default_backend=self.config.trainer.logger,
#             config=OmegaConf.to_container(self.config, resolve=True),
#         )

#         self.global_steps = 0

#         # load checkpoint before doing anything
#         self._load_checkpoint()

#         # perform validation before training
#         # currently, we only support validation using the reward_function.
#         val_metrics = {}  # 确保 val_metrics 始终被初始化
#         if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
#             val_metrics = self._validate()
            
#             if val_metrics: # 只有当 metrics 非空时才打印和记录
#                 pprint(f"Initial validation metrics: {val_metrics}")
#                 logger.log(data=val_metrics, step=self.global_steps)
#             elif len(self.val_dataloader) == 0:
#                  print("⚠️ Initial validation skipped (Validation dataloader is empty).")
                 
#             if self.config.trainer.get("val_only", False):
#                 return

#         if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
#             rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
#             rollout_skip.wrap_generate_sequences()

#         # add tqdm
#         progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

#         # we start from step 1
#         self.global_steps += 1
#         last_val_metrics = None
#         self.max_steps_duration = 0

#         prev_step_profile = False
#         curr_step_profile = (
#             self.global_steps in self.config.global_profiler.steps
#             if self.config.global_profiler.steps is not None
#             else False
#         )
#         next_step_profile = False

#         for epoch in range(self.config.trainer.total_epochs):
#             for batch_dict in self.train_dataloader:
#                 metrics = {}
#                 timing_raw = {}

#                 with marked_timer("start_profile", timing_raw):
#                     self._start_profiling(
#                         not prev_step_profile and curr_step_profile
#                         if self.config.global_profiler.profile_continuous_steps
#                         else curr_step_profile
#                     )

#                 batch: DataProto = DataProto.from_single_dict(batch_dict)

#                 # 🔑 **关键修复**: 检查批次是否有效，防止因空批次导致的崩溃
#                 # 注意：batch.batch 是 TensorDict，不能直接用 if 判断
#                 if batch.batch is None or len(batch.batch) == 0:
#                     print("⚠️ Warning: Skipped an empty or invalid batch from the dataloader.")
#                     continue  # 跳过这个批次

#                 # add uid to batch
#                 batch.non_tensor_batch["uid"] = np.array(
#                     [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
#                 )

#                 gen_batch = self._get_gen_batch(batch)

#                 # pass global_steps to trace
#                 gen_batch.meta_info["global_steps"] = self.global_steps
                
#                 # 新增：传递 epoch 和 step 信息给 entropy writer
#                 gen_batch.meta_info["epoch"] = epoch
#                 gen_batch.meta_info["step"] = self.global_steps
                
                
#                 gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                
#                 # 🔍 DEBUG: 打印 repeat 后的 non_tensor_batch keys
#                 print(f"[DEBUG] After repeat, gen_batch.non_tensor_batch keys: {list(gen_batch.non_tensor_batch.keys())}")
#                 if 'pure_question' in gen_batch.non_tensor_batch:
#                     print(f"[DEBUG] pure_question length: {len(gen_batch.non_tensor_batch['pure_question'])}")
                
#                 # Apply COT augmentation if enabled (for GRPO with different examples per rollout)
#                 if self.cot_augmenter is not None:
#                     gen_batch = self.cot_augmenter.augment(gen_batch)
                    
#                     # 🔍 DEBUG: 打印 augment 后的 non_tensor_batch keys
#                     print(f"[DEBUG] After augment, gen_batch.non_tensor_batch keys: {list(gen_batch.non_tensor_batch.keys())}")
#                     if 'pure_question' in gen_batch.non_tensor_batch:
#                         print(f"[DEBUG] pure_question length: {len(gen_batch.non_tensor_batch['pure_question'])}")
#                     else:
#                         print(f"[DEBUG] ❌ pure_question is MISSING after augment!")

#                 is_last_step = self.global_steps >= self.total_training_steps

#                 with marked_timer("step", timing_raw):
#                     # generate a batch
#                     with marked_timer("gen", timing_raw, color="red"):
#                         if not self.async_rollout_mode:
#                             gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
#                         else:
#                             gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
#                         timing_raw.update(gen_batch_output.meta_info["timing"])
#                         gen_batch_output.meta_info.pop("timing", None)

#                     if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
#                         if self.reward_fn is None:
#                             raise ValueError("A reward_fn is required for REMAX advantage estimation.")

#                         with marked_timer("gen_max", timing_raw, color="purple"):
#                             gen_baseline_batch = deepcopy(gen_batch)
#                             gen_baseline_batch.meta_info["do_sample"] = False
#                             if not self.async_rollout_mode:
#                                 gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
#                             else:
#                                 gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
#                             batch = batch.union(gen_baseline_output)
#                             reward_baseline_tensor = self.reward_fn(batch)
#                             reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

#                             batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

#                             batch.batch["reward_baselines"] = reward_baseline_tensor

#                             del gen_baseline_batch, gen_baseline_output

#                     # repeat to align with repeated responses in rollout
#                     # 🆕 检测 CVAE 分叉，调整 repeat 次数
#                     if gen_batch_output.meta_info.get("cvae_branching_enabled", False):
#                         # CVAE 分叉启用：repeat 次数 = rollout.n × expansion_factor
#                         expansion_factor = gen_batch_output.meta_info.get("cvae_expansion_factor", 1)
#                         repeat_times = self.config.actor_rollout_ref.rollout.n * expansion_factor
#                         print(f"[CVAE Branching] 调整 repeat 次数: {self.config.actor_rollout_ref.rollout.n} × {expansion_factor} = {repeat_times}")
#                     else:
#                         # 正常情况：repeat 次数 = rollout.n
#                         repeat_times = self.config.actor_rollout_ref.rollout.n
                    
#                     batch = batch.repeat(repeat_times=repeat_times, interleave=True)
#                     batch = batch.union(gen_batch_output)

#                     ''' 新增：提取 rollout 阶段的熵值'''
#                     if "rollout_entropies" in batch.batch.keys():
#                         rollout_entropies = batch.batch["rollout_entropies"]
#                         # 提取出来了 tensor shape [batch_size, seq_len]
#                         # 每个位置存储该 token 的熵值
#                         response_masks = compute_response_mask(batch)
#                         loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
#                         rollout_entropy_agg = agg_loss(
#                             loss_mat=rollout_entropies, 
#                             loss_mask=response_masks, 
#                             loss_agg_mode=loss_agg_mode
#                         )
#                         rollout_entropy_metrics = {"rollout/entropy": rollout_entropy_agg.detach().item()}
#                         metrics.update(rollout_entropy_metrics)
#                         print(f"[Rollout Entropy] Step {self.global_steps}: {rollout_entropy_agg.detach().item():.4f}")

#                     if "response_mask" not in batch.batch.keys():
#                         batch.batch["response_mask"] = compute_response_mask(batch)
#                     # Balance the number of valid tokens across DP ranks.
#                     if self.config.trainer.balance_batch:
#                         self._balance_batch(batch, metrics=metrics)

#                     # compute global_valid tokens
#                     batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

#                     with marked_timer("reward", timing_raw, color="yellow"):
#                         # compute reward model score
#                         if self.use_rm and "rm_scores" not in batch.batch.keys():
#                             reward_tensor = self.rm_wg.compute_rm_score(batch)
#                             batch = batch.union(reward_tensor)

#                         if self.config.reward_model.launch_reward_fn_async:
#                             future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
#                         else:
#                             reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

#                     # recompute old_log_probs
#                     with marked_timer("old_log_prob", timing_raw, color="blue"):
#                         old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
#                         entropys = old_log_prob.batch["entropys"]
#                         response_masks = batch.batch["response_mask"]
#                         loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
#                         entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
#                         old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
#                         metrics.update(old_log_prob_metrics)
#                         old_log_prob.batch.pop("entropys")
#                         batch = batch.union(old_log_prob)

#                         if "rollout_log_probs" in batch.batch.keys():
#                             from verl.utils.debug.metrics import calculate_debug_metrics
#                             metrics.update(calculate_debug_metrics(batch))

#                     if self.use_reference_policy:
#                         with marked_timer("ref", timing_raw, color="olive"):
#                             if not self.ref_in_actor:
#                                 ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
#                             else:
#                                 ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
#                             batch = batch.union(ref_log_prob)

#                     # compute values
#                     if self.use_critic:
#                         with marked_timer("values", timing_raw, color="cyan"):
#                             values = self.critic_wg.compute_values(batch)
#                             batch = batch.union(values)

#                     with marked_timer("adv", timing_raw, color="brown"):
#                         if self.config.reward_model.launch_reward_fn_async:
#                             reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
#                         batch.batch["token_level_scores"] = reward_tensor

#                         if reward_extra_infos_dict:
#                             batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

#                         if self.config.algorithm.use_kl_in_reward:
#                             batch, kl_metrics = apply_kl_penalty(
#                                 batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
#                             )
#                             metrics.update(kl_metrics)
#                         else:
#                             batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

#                         norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
#                         batch = compute_advantage(
#                             batch,
#                             adv_estimator=self.config.algorithm.adv_estimator,
#                             gamma=self.config.algorithm.gamma,
#                             lam=self.config.algorithm.lam,
#                             num_repeat=self.config.actor_rollout_ref.rollout.n,
#                             norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
#                             config=self.config.algorithm,
#                         )

#                     # update critic
#                     if self.use_critic:
#                         with marked_timer("update_critic", timing_raw, color="pink"):
#                             critic_output = self.critic_wg.update_critic(batch)
#                         critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
#                         metrics.update(critic_output_metrics)

#                     # implement critic warmup
#                     if self.config.trainer.critic_warmup <= self.global_steps:
#                         with marked_timer("update_actor", timing_raw, color="red"):
#                             batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
#                             actor_output = self.actor_rollout_wg.update_actor(batch)
#                         actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
#                         metrics.update(actor_output_metrics)

#                     rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
#                     if rollout_data_dir:
#                         with marked_timer("dump_rollout_generations", timing_raw, color="green"):
#                             inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
#                             outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
#                             scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            
#                             # 🔧 修复: 正确提取 ground_truth
#                             if "reward_model" in batch.non_tensor_batch and "ground_truth" in batch.non_tensor_batch["reward_model"]:
#                                 sample_gts = batch.non_tensor_batch["reward_model"]["ground_truth"].tolist() if hasattr(batch.non_tensor_batch["reward_model"]["ground_truth"], 'tolist') else list(batch.non_tensor_batch["reward_model"]["ground_truth"])
#                             else:
#                                 sample_gts = [None] * len(inputs)
                            
#                             # 🔧 添加 data_source 信息
#                             if "data_source" in batch.non_tensor_batch:
#                                 data_sources = batch.non_tensor_batch["data_source"].tolist() if hasattr(batch.non_tensor_batch["data_source"], 'tolist') else list(batch.non_tensor_batch["data_source"])
#                                 reward_extra_infos_dict.setdefault("data_source", data_sources)
                            
#                             if "request_id" in batch.non_tensor_batch:
#                                 reward_extra_infos_dict.setdefault("request_id", batch.non_tensor_batch["request_id"].tolist())
                            
#                             # 🎯 限制只打印前5个样本（节省磁盘空间）
#                             num_examine_rollout = self.config.reward_model.get("num_examine_rollout", 5)
#                             if num_examine_rollout > 0 and num_examine_rollout < len(inputs):
#                                 inputs_to_dump = inputs[:num_examine_rollout]
#                                 outputs_to_dump = outputs[:num_examine_rollout]
#                                 gts_to_dump = sample_gts[:num_examine_rollout]
#                                 scores_to_dump = scores[:num_examine_rollout]
#                                 reward_extra_to_dump = {k: v[:num_examine_rollout] if len(v) == len(inputs) else v for k, v in reward_extra_infos_dict.items()}
#                             else:
#                                 inputs_to_dump = inputs
#                                 outputs_to_dump = outputs
#                                 gts_to_dump = sample_gts
#                                 scores_to_dump = scores
#                                 reward_extra_to_dump = reward_extra_infos_dict
                            
#                             self._dump_generations(
#                                 inputs=inputs_to_dump,
#                                 outputs=outputs_to_dump,
#                                 gts=gts_to_dump,
#                                 scores=scores_to_dump,
#                                 reward_extra_infos_dict=reward_extra_to_dump,
#                                 dump_path=rollout_data_dir,
#                             )

#                 # validate
#                 if (
#                     self.val_reward_fn is not None
#                     and self.config.trainer.test_freq > 0
#                     and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
#                 ):
#                     with marked_timer("testing", timing_raw, color="green"):
#                         if len(self.val_dataloader) > 0:
#                             val_metrics: dict = self._validate()
#                             if is_last_step:
#                                 last_val_metrics = val_metrics
#                             metrics.update(val_metrics)
#                         else:
#                             print("⚠️ Validation dataloader is empty, skipping periodic validation.")

#                 esi_close_to_expiration = should_save_ckpt_esi(
#                     max_steps_duration=self.max_steps_duration,
#                     redundant_time=self.config.trainer.esi_redundant_time,
#                 )
#                 if self.config.trainer.save_freq > 0 and (
#                     is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
#                 ):
#                     if esi_close_to_expiration:
#                         print("Force saving checkpoint: ESI instance expiration approaching.")
#                     with marked_timer("save_checkpoint", timing_raw, color="green"):
#                         self._save_checkpoint()

#                 with marked_timer("stop_profile", timing_raw):
#                     next_step_profile = (
#                         self.global_steps + 1 in self.config.global_profiler.steps
#                         if self.config.global_profiler.steps is not None
#                         else False
#                     )
#                     self._stop_profiling(
#                         curr_step_profile and not next_step_profile
#                         if self.config.global_profiler.profile_continuous_steps
#                         else curr_step_profile
#                     )
#                     prev_step_profile = curr_step_profile
#                     curr_step_profile = next_step_profile

#                 steps_duration = timing_raw["step"]
#                 self.max_steps_duration = max(self.max_steps_duration, steps_duration)

#                 metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
#                 metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
#                 metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
#                 n_gpus = self.resource_pool_manager.get_n_gpus()
#                 metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

#                 if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
#                     self.train_dataloader.sampler.update(batch=batch)

#                 logger.log(data=metrics, step=self.global_steps)
#                 progress_bar.update(1)
#                 self.global_steps += 1

#                 if (
#                     hasattr(self.config.actor_rollout_ref.actor, "profiler")
#                     and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
#                 ):
#                     self.actor_rollout_wg.dump_memory_snapshot(
#                         tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
#                     )

#                 if is_last_step:
#                     pprint(f"Final validation metrics: {last_val_metrics}")
#                     progress_bar.close()
#                     return

#                 if hasattr(self.train_dataset, "on_batch_end"):
#                     self.train_dataset.on_batch_end(batch=batch)




# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional
import logging

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"
        
        # Initialize COT augmenter for GRPO if enabled
        self._init_cot_augmenter()

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _init_cot_augmenter(self):
        """Initialize the COT augmenter for GRPO if configured."""
        from verl.utils.grpo_cot_augmentation import GRPOCOTAugmenter, load_cot_examples_from_file
        
        # Check if COT augmentation is enabled in config
        cot_config = self.config.actor_rollout_ref.rollout.get("cot_augmentation", None)
        
        if cot_config is None or not cot_config.get("enable", False):
            self.cot_augmenter = None
            print("COT augmentation is disabled.")
            return
        
        # Load COT examples
        cot_examples = None
        cot_examples_getter = None
        
        # ⭐ NEW: Multi-dataset COT support (simple matching)
        if cot_config.get("use_multi_dataset", False) or cot_config.get("dataset_cot_mapping", None):
            print("Initializing multi-dataset simple COT loader...")
            
            try:
                import sys
                import os
                # Add examples directory to path
                # Go up 4 levels: ppo -> trainer -> verl -> project_root
                examples_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "examples", "grpo_trainer")
                if examples_path not in sys.path:
                    sys.path.insert(0, examples_path)
                
                from multi_dataset_simple_loader import initialize_multi_dataset_simple_cot_loader, get_multi_dataset_simple_cot_examples
                
                # Get COT file mapping from config
                cot_file_mapping = cot_config.get("dataset_cot_mapping", cot_config.get("cot_file_mapping", {}))
                if isinstance(cot_file_mapping, str):
                    # If it's a string, try to parse it as JSON
                    import json
                    cot_file_mapping = json.loads(cot_file_mapping)
                
                if not cot_file_mapping:
                    raise ValueError("dataset_cot_mapping is required for multi-dataset COT but not provided")
                
                # Initialize multi-dataset COT loader
                cot_loader = initialize_multi_dataset_simple_cot_loader(
                    cot_file_mapping=cot_file_mapping,
                    cot_format_template=cot_config.get("cot_format_template",
                        "Here is a reference example that demonstrates the problem-solving approach:\n\n<Example>\nQuestion: {question}\n\nStep-by-step Solution:\n{rationale}\n\nFinal Answer: {final_answer}\n</Example>\n\nNow, please solve the following problem using similar reasoning:"),
                    use_full_cot=cot_config.get("use_full_cot", True),
                    skip_on_mismatch=cot_config.get("skip_on_mismatch", True),
                    verbose=cot_config.get("verbose", False),
                )
                
                # Wrap the getter function
                def multi_cot_getter_wrapper(batch, prompt_idx, num_repeats):
                    return get_multi_dataset_simple_cot_examples(batch, prompt_idx, num_repeats, tokenizer=self.tokenizer)
                
                cot_examples_getter = multi_cot_getter_wrapper
                print(f"Multi-dataset simple COT loader initialized successfully")
                print(f"Loaded COT data for {len(cot_file_mapping)} datasets: {list(cot_file_mapping.keys())}")
                
            except Exception as e:
                import traceback
                print(f"Error: Failed to initialize multi-dataset COT loader: {e}")
                traceback.print_exc()
                raise
        
        # Special handling for GSM8K-style COT files with selected_cots (single dataset)
        elif cot_config.get("cot_file_path", None):
            # This is the GSM8K format with JSONL containing selected_cots
            print(f"Initializing GSM8K COT loader from {cot_config.cot_file_path}")
            
            # Import and initialize the GSM8K COT loader
            try:
                from verl.utils.import_utils import load_extern_type
                # Try to load custom GSM8K COT loader if user provided it
                if cot_config.get("loader_path", None):
                    loader_module = load_extern_type(
                        cot_config.loader_path,
                        "initialize_gsm8k_cot_loader"
                    )
                    cot_loader = loader_module(
                        cot_file_path=cot_config.cot_file_path,
                        cot_format_template=cot_config.get("cot_format_template", 
                            "Here is a reference example that demonstrates the problem-solving approach:\n\n<Example>\nQuestion: {question}\n\nStep-by-step Solution:\n{rationale}\n\nFinal Answer: {final_answer}\n</Example>\n\nNow, please solve the following problem using similar reasoning:"),
                        match_by=cot_config.get("match_by", "question"),
                        use_full_cot=cot_config.get("use_full_cot", True),
                    )
                    getter_func = load_extern_type(cot_config.loader_path, "get_gsm8k_cot_examples")
                else:
                    # Use the default GSM8K COT loader from examples
                    import sys
                    import os
                    # Add examples directory to path
                    # Go up 4 levels: ppo -> trainer -> verl -> project_root
                    examples_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "examples", "grpo_trainer")
                    if examples_path not in sys.path:
                        sys.path.insert(0, examples_path)
                    
                    from gsm8k_cot_loader import initialize_gsm8k_cot_loader, get_gsm8k_cot_examples
                    
                    cot_loader = initialize_gsm8k_cot_loader(
                        cot_file_path=cot_config.cot_file_path,
                        cot_format_template=cot_config.get("cot_format_template",
                            "Here is a reference example that demonstrates the problem-solving approach:\n\n<Example>\nQuestion: {question}\n\nStep-by-step Solution:\n{rationale}\n\nFinal Answer: {final_answer}\n</Example>\n\nNow, please solve the following problem using similar reasoning:"),
                        match_by=cot_config.get("match_by", "question"),
                        use_full_cot=cot_config.get("use_full_cot", True),
                    )
                    getter_func = get_gsm8k_cot_examples
                
                # Wrap the getter function to match expected signature
                def cot_getter_wrapper(batch, prompt_idx, num_repeats):
                    return getter_func(batch, prompt_idx, num_repeats, tokenizer=self.tokenizer)
                
                cot_examples_getter = cot_getter_wrapper
                print(f"GSM8K COT loader initialized successfully")
                
            except Exception as e:
                print(f"Warning: Failed to initialize GSM8K COT loader: {e}")
                print("Falling back to standard examples_file loading")
                # Fallback to standard file loading
                cot_examples = load_cot_examples_from_file(cot_config.cot_file_path)
                print(f"Loaded {len(cot_examples)} COT examples from {cot_config.cot_file_path}")
        
        elif cot_config.get("examples_file", None):
            # Load from file (simple text/json format)
            cot_examples = load_cot_examples_from_file(cot_config.examples_file)
            print(f"Loaded {len(cot_examples)} COT examples from {cot_config.examples_file}")
        elif cot_config.get("examples", None):
            # Use examples from config
            cot_examples = list(cot_config.examples)
            print(f"Using {len(cot_examples)} COT examples from config")
        elif cot_config.get("examples_getter", None):
            # Use custom getter function
            from verl.utils.import_utils import load_extern_type
            getter_path = cot_config.examples_getter.get("path")
            getter_name = cot_config.examples_getter.get("name")
            cot_examples_getter = load_extern_type(getter_path, getter_name)
            print(f"Using custom COT examples getter: {getter_name} from {getter_path}")
        else:
            raise ValueError(
                "COT augmentation is enabled but no examples provided. "
                "Please specify 'cot_file_path', 'examples_file', 'examples', or 'examples_getter' in config."
            )
        
        # Initialize augmenter
        self.cot_augmenter = GRPOCOTAugmenter(
            cot_examples=cot_examples,
            cot_examples_getter=cot_examples_getter,
            tokenizer=self.tokenizer,
            num_repeats=self.config.actor_rollout_ref.rollout.n,
            sampling_strategy=cot_config.get("sampling_strategy", "sequential"),
            add_separator=cot_config.get("add_separator", True),
            separator=cot_config.get("separator", "\n\n"),
            enable=cot_config.get("enable", False),
            seed=cot_config.get("seed", None),
            debug_print_augmented_prompts=cot_config.get("debug_print_augmented_prompts", True),
            debug_num_samples=cot_config.get("debug_num_samples", 3),
            debug_print_full_prompt=cot_config.get("debug_print_full_prompt", False),  # 🆕 传递新参数
        )
        print(f"COT augmenter initialized with strategy: {cot_config.get('sampling_strategy', 'sequential')}")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor, is_train=True
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor, is_train=False
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        # 🔑 修复：StatefulDataLoader 不支持空数据集，需要特殊处理
        if len(self.val_dataset) == 0:
            # 验证集为空时，创建一个虚拟的空 DataLoader
            print("⚠️ Warning: Validation dataset is empty. Creating a dummy validation dataloader.")
            # 使用一个简单的空列表作为空 DataLoader 的替代
            self.val_dataloader = []  # 空列表，len() 返回 0，迭代时不会产生任何 batch
        else:
            # 正常情况：创建真实的验证集 DataLoader
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        # 打印 DataLoader 信息
        if len(self.val_dataset) == 0:
            print(f"Size of train dataloader: {len(self.train_dataloader)}")
            print(f"⚠️ Validation dataloader is empty (validation will be skipped)")
        else:
            print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")
        

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL.
        
        🎯 用于验证 reward 计算的正确性
        输出文件: {dump_path}/{step}.jsonl
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "ground_truth": gts,  # 🔧 改名为 ground_truth，更清晰
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"\n{'='*80}")
        print(f"✅ Dumped {n} generations to: {filename}")
        print(f"{'='*80}")
        
        # 🎯 打印前3个样本到控制台，方便快速检查
        print(f"\n📊 Sample Preview (first 3 of {n}):")
        for i in range(min(3, n)):
            print(f"\n--- Sample {i+1} ---")
            print(f"[data_source]: {base_data.get('data_source', ['N/A'])[i] if 'data_source' in base_data else 'N/A'}")
            print(f"[input]: {inputs[i][:100]}...")  # 只显示前100个字符
            print(f"[output]: {outputs[i][:200]}...")  # 只显示前200个字符
            print(f"[ground_truth]: {gts[i][:100] if gts[i] else 'None'}...")
            print(f"[score]: {scores[i]}")
        print(f"{'='*80}\n")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        # 🔑 修复：只pop存在的字段，避免position_ids不存在时的AssertionError
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        existing_batch_keys = [k for k in batch_keys_to_pop if k in batch.batch.keys()]
        
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=existing_batch_keys,  # ← 只pop存在的字段
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

# 文件: verl/trainer/ppo/ray_trainer.py
# 替换函数: RayPPOTrainer._validate

    def _validate(self):
        # 🔑 FIX 1: 检查 dataloader 是否为空 (依赖于您的 MultiDatasetWithCOT 类的修复)
        if len(self.val_dataloader) == 0:
            print("⚠️ Validation dataloader is empty, skipping validation.")
            return {}
            
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # 🔑 FIX 2: 在连接之前检查列表是否为空
        if not data_source_lst:
            print("⚠️ Validation data list is empty after processing. Skipping metric computation.")
            return {}

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    # 文件: verl/trainer/ppo/ray_trainer.py
# 替换函数: RayPPOTrainer.fit

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        val_metrics = {}  # 确保 val_metrics 始终被初始化
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            
            if val_metrics: # 只有当 metrics 非空时才打印和记录
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
            elif len(self.val_dataloader) == 0:
                 print("⚠️ Initial validation skipped (Validation dataloader is empty).")
                 
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # 🔑 **关键修复**: 检查批次是否有效，防止因空批次导致的崩溃
                # 注意：batch.batch 是 TensorDict，不能直接用 if 判断
                if batch.batch is None or len(batch.batch) == 0:
                    print("⚠️ Warning: Skipped an empty or invalid batch from the dataloader.")
                    continue  # 跳过这个批次

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                
                # 新增：传递 epoch 和 step 信息给 entropy writer
                gen_batch.meta_info["epoch"] = epoch
                gen_batch.meta_info["step"] = self.global_steps
                
                
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                
                # 🔍 DEBUG: 打印 repeat 后的 non_tensor_batch keys
                print(f"[DEBUG] After repeat, gen_batch.non_tensor_batch keys: {list(gen_batch.non_tensor_batch.keys())}")
                if 'pure_question' in gen_batch.non_tensor_batch:
                    print(f"[DEBUG] pure_question length: {len(gen_batch.non_tensor_batch['pure_question'])}")
                
                # Apply COT augmentation if enabled (for GRPO with different examples per rollout)
                if self.cot_augmenter is not None:
                    gen_batch = self.cot_augmenter.augment(gen_batch)
                    
                    # 🔍 DEBUG: 打印 augment 后的 non_tensor_batch keys
                    print(f"[DEBUG] After augment, gen_batch.non_tensor_batch keys: {list(gen_batch.non_tensor_batch.keys())}")
                    if 'pure_question' in gen_batch.non_tensor_batch:
                        print(f"[DEBUG] pure_question length: {len(gen_batch.non_tensor_batch['pure_question'])}")
                    else:
                        print(f"[DEBUG] ❌ pure_question is MISSING after augment!")

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    # 🆕 检测 CVAE 分叉，调整 repeat 次数
                    if gen_batch_output.meta_info.get("cvae_branching_enabled", False):
                        # CVAE 分叉启用：repeat 次数 = rollout.n × expansion_factor
                        expansion_factor = gen_batch_output.meta_info.get("cvae_expansion_factor", 1)
                        repeat_times = self.config.actor_rollout_ref.rollout.n * expansion_factor
                        print(f"[CVAE Branching] 调整 repeat 次数: {self.config.actor_rollout_ref.rollout.n} × {expansion_factor} = {repeat_times}")
                    else:
                        # 正常情况：repeat 次数 = rollout.n
                        repeat_times = self.config.actor_rollout_ref.rollout.n
                    
                    batch = batch.repeat(repeat_times=repeat_times, interleave=True)
                    batch = batch.union(gen_batch_output)

                    ''' 新增：提取 rollout 阶段的熵值'''
                    if "rollout_entropies" in batch.batch.keys():
                        rollout_entropies = batch.batch["rollout_entropies"]
                        # 提取出来了 tensor shape [batch_size, seq_len]
                        # 每个位置存储该 token 的熵值
                        response_masks = compute_response_mask(batch)
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        rollout_entropy_agg = agg_loss(
                            loss_mat=rollout_entropies, 
                            loss_mask=response_masks, 
                            loss_agg_mode=loss_agg_mode
                        )
                        rollout_entropy_metrics = {"rollout/entropy": rollout_entropy_agg.detach().item()}
                        metrics.update(rollout_entropy_metrics)
                        print(f"[Rollout Entropy] Step {self.global_steps}: {rollout_entropy_agg.detach().item():.4f}")

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            from verl.utils.debug.metrics import calculate_debug_metrics
                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    
                    # 🔍 调试信息：打印 rollout_data_dir 配置
                    print(f"\n{'='*80}")
                    print(f"[DEBUG] rollout_data_dir 配置检查:")
                    print(f"  - 配置值: {rollout_data_dir}")
                    print(f"  - 类型: {type(rollout_data_dir)}")
                    print(f"  - 是否为空: {rollout_data_dir is None or rollout_data_dir == ''}")
                    print(f"{'='*80}\n")
                    
                    if rollout_data_dir:
                        print(f"✅ 准备保存 rollout 数据到: {rollout_data_dir}")
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            
                            # 🔧 修复: 正确提取 ground_truth
                            if "reward_model" in batch.non_tensor_batch and "ground_truth" in batch.non_tensor_batch["reward_model"]:
                                sample_gts = batch.non_tensor_batch["reward_model"]["ground_truth"].tolist() if hasattr(batch.non_tensor_batch["reward_model"]["ground_truth"], 'tolist') else list(batch.non_tensor_batch["reward_model"]["ground_truth"])
                            else:
                                sample_gts = [None] * len(inputs)
                            
                            # 🔧 添加 data_source 信息
                            if "data_source" in batch.non_tensor_batch:
                                data_sources = batch.non_tensor_batch["data_source"].tolist() if hasattr(batch.non_tensor_batch["data_source"], 'tolist') else list(batch.non_tensor_batch["data_source"])
                                reward_extra_infos_dict.setdefault("data_source", data_sources)
                            
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault("request_id", batch.non_tensor_batch["request_id"].tolist())
                            
                            # 🎯 限制只打印前5个样本（节省磁盘空间）
                            num_examine_rollout = self.config.reward_model.get("num_examine_rollout", 5)
                            if num_examine_rollout > 0 and num_examine_rollout < len(inputs):
                                inputs_to_dump = inputs[:num_examine_rollout]
                                outputs_to_dump = outputs[:num_examine_rollout]
                                gts_to_dump = sample_gts[:num_examine_rollout]
                                scores_to_dump = scores[:num_examine_rollout]
                                reward_extra_to_dump = {k: v[:num_examine_rollout] if len(v) == len(inputs) else v for k, v in reward_extra_infos_dict.items()}
                            else:
                                inputs_to_dump = inputs
                                outputs_to_dump = outputs
                                gts_to_dump = sample_gts
                                scores_to_dump = scores
                                reward_extra_to_dump = reward_extra_infos_dict
                            
                            self._dump_generations(
                                inputs=inputs_to_dump,
                                outputs=outputs_to_dump,
                                gts=gts_to_dump,
                                scores=scores_to_dump,
                                reward_extra_infos_dict=reward_extra_to_dump,
                                dump_path=rollout_data_dir,
                            )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        if len(self.val_dataloader) > 0:
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                            metrics.update(val_metrics)
                        else:
                            print("⚠️ Validation dataloader is empty, skipping periodic validation.")

                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)