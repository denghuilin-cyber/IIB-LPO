


# # Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# Multi-dataset support for GRPO training with COT.

# This dataset class handles multiple datasets (GSM8K, MATH, NuminaMath-CoT, etc.)
# and includes dataset source information for COT matching.
# """

# import os
# import pandas as pd
# from torch.utils.data import Dataset
# from typing import List, Dict, Optional, Any, Union


# class MultiDatasetWithCOT(Dataset):
#     """
#     Dataset that combines multiple datasets and tracks their sources.
    
#     Each sample includes a 'dataset_name' field that identifies which
#     dataset it comes from, enabling correct COT selection.
    
#     Example usage:
#         dataset = MultiDatasetWithCOT(
#             dataset_configs=[
#                 {
#                     "name": "gsm8k",
#                     "files": ["/path/to/gsm8k/train.parquet"],
#                     "prompt_key": "question"
#                 },
#                 {
#                     "name": "math",
#                     "files": ["/path/to/math/train.parquet"],
#                     "prompt_key": "problem"
#                 }
#             ],
#             tokenizer=tokenizer,
#             config=config
#         )
#     """
    
#     def __init__(
#         self,
#         data_files=None,  # ⭐ 兼容标准调用方式（会被忽略）
#         tokenizer=None,
#         processor=None,
#         config=None,
#         dataset_configs: Optional[List[Dict]] = None,
#         # 🆕 简化的配置方式（从config中读取）
#         gsm8k_path: Optional[str] = None,
#         math_path: Optional[str] = None,
#         numina_path: Optional[str] = None,
#         is_train: bool = True,  # 🆕 添加训练/验证标识
#     ):
#         """
#         Initialize the multi-dataset.
        
#         Args:
#             data_files: Ignored (for compatibility with standard dataset interface)
#             tokenizer: Tokenizer for encoding text
#             processor: Processor (can be None)
#             config: Data configuration
#             dataset_configs: List of dataset configurations, each containing:
#                 - name: Dataset identifier (e.g., "gsm8k", "math")
#                 - files: List of file paths
#                 - prompt_key: Field name for the question/problem
#                 - answer_key: (Optional) Field name for the answer
            
#             # 🆕 简化配置（Hydra友好）：
#             gsm8k_path: Path to GSM8K dataset file
#             math_path: Path to MATH dataset file  
#             numina_path: Path to Numina dataset file
#         """
#         self.tokenizer = tokenizer
#         self.processor = processor
#         self.config = config if config is not None else {}
#         self.is_train = is_train
        
#         # 🆕 从config中读取简化路径和字段名（优先级最高）
#         # 🔑 关键修复：验证集模式下不读取训练数据路径
#         # 初始化默认值
#         _gsm8k_path, _math_path, _numina_path = None, None, None

#         if config is not None:
#             # 只有在训练模式下，才从 config 中读取训练数据路径
#             if is_train:
#                 _gsm8k_path = gsm8k_path or config.get('gsm8k_path')
#                 _math_path = math_path or config.get('math_path')
#                 _numina_path = numina_path or config.get('numina_path')
            
#             # 字段名读取（不区分训练/验证模式）
#             gsm8k_prompt_key = config.get('gsm8k_prompt_key', 'prompt')
#             gsm8k_answer_key = config.get('gsm8k_answer_key', 'extra_info.answer')
#             math_prompt_key = config.get('math_prompt_key', 'prompt')
#             math_answer_key = config.get('math_answer_key', 'extra_info.answer')
#             numina_prompt_key = config.get('numina_prompt_key', 'problem')
#             numina_answer_key = config.get('numina_answer_key', 'solution')
#         else:
#             # 没有 config 时的默认字段名
#             gsm8k_prompt_key = 'prompt'
#             gsm8k_answer_key = 'extra_info.answer'
#             math_prompt_key = 'prompt'
#             math_answer_key = 'extra_info.answer'
#             numina_prompt_key = 'problem'
#             numina_answer_key = 'solution'
        
#         # --- 自动构建 dataset_configs ---
        
#         # 如果提供了简化路径，自动构建 dataset_configs
#         if dataset_configs is None and (_gsm8k_path or _math_path or _numina_path):
#             dataset_configs = []
#             if _gsm8k_path:
#                 dataset_configs.append({
#                     'name': 'gsm8k',
#                     'files': [_gsm8k_path],
#                     'prompt_key': gsm8k_prompt_key,
#                     'answer_key': gsm8k_answer_key
#                 })
#             if _math_path:
#                 dataset_configs.append({
#                     'name': 'math',
#                     'files': [_math_path],
#                     'prompt_key': math_prompt_key,
#                     'answer_key': math_answer_key
#                 })
#             if _numina_path:
#                 dataset_configs.append({
#                     'name': 'numina',
#                     'files': [_numina_path],
#                     'prompt_key': numina_prompt_key,
#                     'answer_key': numina_answer_key
#                 })
#             print(f"📝 使用简化配置，自动构建了 {len(dataset_configs)} 个数据集")
#             print(f"   模式: {'训练' if is_train else '验证'}")
        
#         # 🔑 关键修复：处理空的验证集 (确保 self.samples 被清空)
#         is_config_empty = not dataset_configs or all(not config.get('files') for config in dataset_configs)
        
#         if not is_train and is_config_empty:
#             print(f"⚠️ 验证集为空，创建空数据集")
#             dataset_configs = []
        
#         if not dataset_configs:
#             if is_train:
#                 raise ValueError(
#                     "训练集必须提供 dataset_configs 或至少一个数据集路径（gsm8k_path/math_path/numina_path）。"
#                 )
#             else:
#                 print(f"⚠️ 验证集为空，将创建空数据集")
        
#         self.dataset_configs = dataset_configs
        
#         # Store all samples with their metadata
#         self.samples = []
#         self.dataset_names = []
#         self.questions = []
#         self.question_ids = []
        
#         # Load data from all datasets
#         self._load_all_datasets()
        
#         print(f"Loaded {len(self.samples)} total samples from {len(self.dataset_configs)} datasets")
        
#         # 🔍 验证集调试信息
#         if not is_train:
#             print(f"\n🔍 验证集调试信息:")
#             print(f"   样本数量: {len(self.samples)}")
#             if len(self.samples) > 0:
#                 sample = self.samples[0]
#                 print(f"   第一个样本的字段: {list(sample.keys())}")
#                 print(f"   dataset_name: {sample.get('dataset_name', 'N/A')}")
#                 print(f"   question类型: {type(sample.get('question', 'N/A'))}")
#                 print(f"   answer类型: {type(sample.get('answer', 'N/A'))}")
#             else:
#                 print("   ⚠️ 验证集为空！")
    
#     def _load_all_datasets(self):
#         """Load data from all configured datasets."""
#         global_idx = 0
        
#         # 🔑 处理空的验证集
#         if not self.dataset_configs:
#             # 如果配置为空 (即在验证模式下被清空)，则直接返回
#             print(f"⚠️ 验证数据集配置为空，创建空数据集")
#             return
        
#         for dataset_config in self.dataset_configs:
#             dataset_name = dataset_config["name"]
#             data_files = dataset_config["files"]
#             prompt_key = dataset_config.get("prompt_key", "prompt")
#             answer_key = dataset_config.get("answer_key", "extra_info.answer")
            
#             print(f"\nLoading dataset: {dataset_name}")
#             print(f"  Files: {data_files}")
#             print(f"  Prompt key: {prompt_key}")
            
#             # Load files for this dataset
#             if isinstance(data_files, str):
#                 data_files = [data_files]
            
#             dfs = []
#             for file_path in data_files:
#                 if file_path.endswith('.parquet'):
#                     df = pd.read_parquet(file_path)
#                 elif file_path.endswith('.jsonl'):
#                     df = pd.read_json(file_path, lines=True)
#                 elif file_path.endswith('.json'):
#                     df = pd.read_json(file_path)
#                 else:
#                     raise ValueError(f"Unsupported file format: {file_path}")
#                 dfs.append(df)
            
#             dataset_df = pd.concat(dfs, ignore_index=True)
#             print(f"  Loaded {len(dataset_df)} samples")
            
#             # 🔍 调试信息：打印数据集的列名和前几行
#             print(f"\n🔍 调试信息 - 数据集 '{dataset_name}' 的结构:")
#             print(f"  列名: {list(dataset_df.columns)}")
#             print(f"  数据类型:\n{dataset_df.dtypes}")
#             if len(dataset_df) > 0:
#                 print(f"\n  前3行数据预览:")
#                 print(dataset_df.head(3))
#                 print(f"\n  尝试访问的 prompt_key: '{prompt_key}'")
#                 if prompt_key not in dataset_df.columns:
#                     print(f"  ❌ 错误: '{prompt_key}' 不在列名中!")
#                     print(f"  💡 可能的字段名: {[col for col in dataset_df.columns if 'question' in col.lower() or 'prompt' in col.lower() or 'problem' in col.lower()]}")
#                     raise KeyError(f"字段 '{prompt_key}' 不存在。可用字段: {list(dataset_df.columns)}")
#                 else:
#                     print(f"  ✅ '{prompt_key}' 字段存在")
#             print("=" * 80 + "\n")
            
#             # Process each sample
#             for idx, row in dataset_df.iterrows():
#                 # 🔍 智能获取问题文本（支持多种格式）
#                 question = None
#                 question_for_matching = None  # 用于COT匹配的纯文本问题
                
#                 # 方式1：从extra_info.question获取（最优先，用于COT匹配）
#                 if 'extra_info' in row and isinstance(row['extra_info'], dict):
#                     if 'question' in row['extra_info']:
#                         question_for_matching = row['extra_info']['question']
#                         question = question_for_matching  # 默认也用这个
                
#                 # 方式2：从prompt字段获取（用于训练）
#                 if prompt_key == 'prompt' and 'prompt' in row:
#                     prompt_value = row['prompt']
#                     if isinstance(prompt_value, list):
#                         # Chat格式: [{'content': '...', 'role': 'user'}]
#                         if len(prompt_value) > 0 and 'content' in prompt_value[0]:
#                             question = prompt_value[0]['content']
#                     elif isinstance(prompt_value, str):
#                         question = prompt_value
                
#                 # 方式3：支持嵌套字段访问（例如 'extra_info.question'）
#                 elif '.' in prompt_key:
#                     parts = prompt_key.split('.')
#                     question = row[parts[0]]
#                     for part in parts[1:]:
#                         if isinstance(question, dict):
#                             question = question[part]
#                         else:
#                             raise ValueError(f"Cannot access nested field: {prompt_key}")
                
#                 # 方式4：直接字段访问
#                 elif prompt_key in row:
#                     question = row[prompt_key]
                
#                 # 如果question_for_matching还是None，使用question
#                 if question_for_matching is None:
#                     question_for_matching = question
                
#                 if question is None:
#                     print(f"⚠️  警告: 样本 {idx} 无法提取问题，跳过")
#                     continue
                
#                 # Get ID
#                 if 'id' in row:
#                     question_id = row['id']
#                 elif 'extra_info' in row and isinstance(row['extra_info'], dict) and 'index' in row['extra_info']:
#                     question_id = row['extra_info']['index']
#                 else:
#                     question_id = global_idx
                
#                 # Get answer（支持嵌套字段）
#                 answer = None
#                 if '.' in answer_key:
#                     # 嵌套访问，例如 'extra_info.answer'
#                     parts = answer_key.split('.')
#                     answer = row.get(parts[0])
#                     if answer is not None:
#                         for part in parts[1:]:
#                             if isinstance(answer, dict):
#                                 answer = answer.get(part)
#                             else:
#                                 answer = None
#                                 break
#                 else:
#                     answer = row.get(answer_key)
                
#                 # 如果答案仍然是None，尝试从extra_info中获取
#                 if answer is None and 'extra_info' in row and isinstance(row['extra_info'], dict):
#                     answer = row['extra_info'].get('answer') or row['extra_info'].get('response')
                
#                 # Store sample info
#                 sample = {
#                     'dataset_name': dataset_name,
#                     'question': question,  # 用于训练的问题文本
#                     'question_for_matching': question_for_matching,  # 用于COT匹配的纯文本
#                     'question_id': question_id,
#                     'answer': answer,
#                     'original_row': row.to_dict()
#                 }
                
#                 self.samples.append(sample)
#                 self.dataset_names.append(dataset_name)
#                 self.questions.append(question_for_matching)  # 存储用于匹配的版本
#                 self.question_ids.append(question_id)
                
#                 global_idx += 1
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, index):
#         """Get a data sample with dataset source information."""
#         sample = self.samples[index]
        
#         question = sample['question']  # 用于训练的问题
#         question_for_matching = sample.get('question_for_matching', question)  # 用于COT匹配
#         dataset_name = sample['dataset_name']
#         question_id = sample['question_id']
        
#         # Apply chat template if available
#         if hasattr(self.tokenizer, 'apply_chat_template'):
#             messages = [{"role": "user", "content": question}]
#             prompt_text = self.tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
#         else:
#             prompt_text = question
        
#         # Tokenize
#         max_prompt_length = self.config.get('max_prompt_length', 512)
#         encoded = self.tokenizer(
#             prompt_text,
#             truncation=True,
#             max_length=max_prompt_length,
#             padding=False,
#             return_tensors=None,
#         )
        
#         # Prepare the output dict
#         result = {
#             'input_ids': encoded['input_ids'],
#             'attention_mask': encoded['attention_mask'],
#             'dataset_name': dataset_name,  # ⭐ Key field for COT matching
#             'question': question_for_matching,  # ⭐ 用于COT匹配的纯文本问题
#             'question_id': int(question_id),
#         }
        
#         # Add answer if available
#         if sample['answer'] is not None:
#             result['data_answer'] = str(sample['answer'])
        
#         return result


# def create_multi_dataset_from_config(config_path: str = None, **kwargs):
#     """
#     Helper function to create multi-dataset from a YAML config file.
    
#     Config format:
#         datasets:
#           - name: gsm8k
#             files:
#               - /path/to/gsm8k/train.parquet
#             prompt_key: question
#             answer_key: answer
          
#           - name: math
#             files:
#               - /path/to/math/train.parquet
#             prompt_key: problem
#             answer_key: solution
#     """
#     import yaml
    
#     if config_path and os.path.exists(config_path):
#         with open(config_path, 'r') as f:
#             config = yaml.safe_load(f)
#         dataset_configs = config['datasets']
#     else:
#         # Fallback to kwargs
#         dataset_configs = kwargs.get('dataset_configs', [])
    
#     return dataset_configs




# MultiDatasetWithCOT.py (完整修复后的代码)
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
# """
# Multi-dataset support for GRPO training with COT.

# This dataset class handles multiple datasets (GSM8K, MATH, NuminaMath-CoT, etc.)
# and includes dataset source information for COT matching.
# """

# import os
# import pandas as pd
# from torch.utils.data import Dataset
# from typing import List, Dict, Optional, Any, Union


# class MultiDatasetWithCOT(Dataset):
#     """
#     Dataset that combines multiple datasets and tracks their sources.
    
#     Each sample includes a 'dataset_name' field that identifies which
#     dataset it comes from, enabling correct COT selection.
#     """
    
#     def __init__(
#         self,
#         data_files=None,  # ⭐ 兼容标准调用方式（会被忽略）
#         tokenizer=None,
#         processor=None,
#         config=None,
#         dataset_configs: Optional[List[Dict]] = None,
#         # 🆕 简化的配置方式（从config中读取）
#         gsm8k_path: Optional[str] = None,
#         math_path: Optional[str] = None,
#         numina_path: Optional[str] = None,
#         is_train: bool = True,  # 🆕 添加训练/验证标识
#     ):
#         """
#         Initialize the multi-dataset.
#         """
#         self.tokenizer = tokenizer
#         self.processor = processor
#         self.config = config if config is not None else {}
#         self.is_train = is_train
        
#         # 🆕 从config中读取简化路径和字段名（优先级最高）
#         # 初始化默认值
#         _gsm8k_path, _math_path, _numina_path = None, None, None

#         if config is not None:
#             # 只有在训练模式下，才从 config 中读取训练数据路径
#             if is_train:
#                 _gsm8k_path = gsm8k_path or config.get('gsm8k_path')
#                 _math_path = math_path or config.get('math_path')
#                 _numina_path = numina_path or config.get('numina_path')
            
#             # 字段名读取（不区分训练/验证模式）
#             gsm8k_prompt_key = config.get('gsm8k_prompt_key', 'prompt')
#             gsm8k_answer_key = config.get('gsm8k_answer_key', 'extra_info.answer')
#             math_prompt_key = config.get('math_prompt_key', 'prompt')
#             math_answer_key = config.get('math_answer_key', 'extra_info.answer')
#             numina_prompt_key = config.get('numina_prompt_key', 'problem')
#             numina_answer_key = config.get('numina_answer_key', 'solution')
#         else:
#             # 没有 config 时的默认字段名
#             gsm8k_prompt_key = 'prompt'
#             gsm8k_answer_key = 'extra_info.answer'
#             math_prompt_key = 'prompt'
#             math_answer_key = 'extra_info.answer'
#             numina_prompt_key = 'problem'
#             numina_answer_key = 'solution'
        
#         # --- 自动构建 dataset_configs ---
        
#         dataset_configs_list = dataset_configs or []

#         # 🆕 如果提供了简化路径，自动构建 dataset_configs (仅在训练模式下触发简化加载)
#         if self.is_train and not dataset_configs_list and (_gsm8k_path or _math_path or _numina_path):
#             if _gsm8k_path:
#                 dataset_configs_list.append({
#                     'name': 'gsm8k', 'files': [_gsm8k_path],
#                     'prompt_key': gsm8k_prompt_key, 'answer_key': gsm8k_answer_key
#                 })
#             if _math_path:
#                 dataset_configs_list.append({
#                     'name': 'math', 'files': [_math_path],
#                     'prompt_key': math_prompt_key, 'answer_key': math_answer_key
#                 })
#             if _numina_path:
#                 dataset_configs_list.append({
#                     'name': 'numina', 'files': [_numina_path],
#                     'prompt_key': numina_prompt_key, 'answer_key': numina_answer_key
#                 })
#             print(f"📝 使用简化配置，自动构建了 {len(dataset_configs_list)} 个数据集")
#             print(f"   模式: {'训练' if is_train else '验证'}")
        
#         # --- 3. 处理空数据集和断言 ---

#         is_config_empty = not dataset_configs_list or all(not config.get('files') for config in dataset_configs_list)
        
#         if not is_train and is_config_empty:
#             # 🔑 关键修复：验证模式下无配置则提前返回空数据集
#             print(f"⚠️ 验证集路径为空，跳过数据加载。")
#             self.samples = []
#             self.dataset_names = []
#             self.questions = []
#             self.question_ids = []
#             return 
        
#         if not dataset_configs_list and is_train:
#              raise ValueError(
#                 "训练集必须提供 dataset_configs 或至少一个数据集路径（gsm8k_path/math_path/numina_path）。"
#             )
        
#         self.dataset_configs = dataset_configs_list
        
#         # Store all samples with their metadata
#         self.samples = []
#         self.dataset_names = []
#         self.questions = []
#         self.question_ids = []
        
#         # Load data from all datasets
#         self._load_all_datasets()
        
#         print(f"Loaded {len(self.samples)} total samples from {len(self.dataset_configs)} datasets")
        
#         # 🔍 验证集调试信息
#         if not is_train:
#             print(f"\n🔍 验证集调试信息:")
#             print(f"   样本数量: {len(self.samples)}")
#             if len(self.samples) > 0:
#                 sample = self.samples[0]
#                 print(f"   第一个样本的字段: {list(sample.keys())}")
#                 print(f"   dataset_name: {sample.get('dataset_name', 'N/A')}")
#                 print(f"   question类型: {type(sample.get('question', 'N/A'))}")
#                 print(f"   answer类型: {type(sample.get('answer', 'N/A'))}")
#             else:
#                 print("   ⚠️ 验证集为空！")

    
#     def _load_all_datasets(self):
#         """Load data from all configured datasets."""
#         global_idx = 0
        
#         # 🔑 处理空的验证集
#         if not self.dataset_configs:
#             print(f"⚠️ 验证数据集配置为空，创建空数据集")
#             return
        
#         for dataset_config in self.dataset_configs:
#             dataset_name = dataset_config["name"]
#             data_files = dataset_config["files"]
#             prompt_key = dataset_config.get("prompt_key", "prompt")
#             answer_key = dataset_config.get("answer_key", "extra_info.answer")
            
#             print(f"\nLoading dataset: {dataset_name}")
#             print(f"  Files: {data_files}")
#             print(f"  Prompt key: {prompt_key}")
            
#             # Load files for this dataset
#             if isinstance(data_files, str):
#                 data_files = [data_files]
            
#             dfs = []
#             for file_path in data_files:
#                 if file_path.endswith('.parquet'):
#                     df = pd.read_parquet(file_path)
#                 elif file_path.endswith('.jsonl'):
#                     df = pd.read_json(file_path, lines=True)
#                 elif file_path.endswith('.json'):
#                     df = pd.read_json(file_path)
#                 else:
#                     raise ValueError(f"Unsupported file format: {file_path}")
#                 dfs.append(df)
            
#             dataset_df = pd.concat(dfs, ignore_index=True)
#             print(f"  Loaded {len(dataset_df)} samples")
            
#             # 🔍 调试信息：打印数据集的列名和前几行
#             print(f"\n🔍 调试信息 - 数据集 '{dataset_name}' 的结构:")
#             print(f"  列名: {list(dataset_df.columns)}")
#             print(f"  数据类型:\n{dataset_df.dtypes}")
#             if len(dataset_df) > 0:
#                 print(f"\n  前3行数据预览:")
#                 print(dataset_df.head(3))
#                 print(f"\n  尝试访问的 prompt_key: '{prompt_key}'")
#                 if prompt_key not in dataset_df.columns:
#                     print(f"  ❌ 错误: '{prompt_key}' 不在列名中!")
#                     print(f"  💡 可能的字段名: {[col for col in dataset_df.columns if 'question' in col.lower() or 'prompt' in col.lower() or 'problem' in col.lower()]}")
#                     raise KeyError(f"字段 '{prompt_key}' 不存在。可用字段: {list(dataset_df.columns)}")
#                 else:
#                     print(f"  ✅ '{prompt_key}' 字段存在")
#             print("=" * 80 + "\n")
            
#             # Process each sample
#             for idx, row in dataset_df.iterrows():
#                 # 🔍 智能获取问题文本（支持多种格式）
#                 question = None
#                 question_for_matching = None  # 用于COT匹配的纯文本问题
                
#                 # 方式1：从extra_info.question获取（最优先，用于COT匹配）
#                 if 'extra_info' in row and isinstance(row['extra_info'], dict):
#                     if 'question' in row['extra_info']:
#                         question_for_matching = row['extra_info']['question']
#                         question = question_for_matching  # 默认也用这个
                
#                 # 方式2：从prompt字段获取（用于训练）
#                 if prompt_key == 'prompt' and 'prompt' in row:
#                     prompt_value = row['prompt']
#                     if isinstance(prompt_value, list):
#                         # Chat格式: [{'content': '...', 'role': 'user'}]
#                         if len(prompt_value) > 0 and 'content' in prompt_value[0]:
#                             question = prompt_value[0]['content']
#                     elif isinstance(prompt_value, str):
#                         question = prompt_value
                
#                 # 方式3：支持嵌套字段访问（例如 'extra_info.question'）
#                 elif '.' in prompt_key:
#                     parts = prompt_key.split('.')
#                     question = row[parts[0]]
#                     for part in parts[1:]:
#                         if isinstance(question, dict):
#                             question = question[part]
#                         else:
#                             raise ValueError(f"Cannot access nested field: {prompt_key}")
                
#                 # 方式4：直接字段访问
#                 elif prompt_key in row:
#                     question = row[prompt_key]
                
#                 # 如果question_for_matching还是None，使用question
#                 if question_for_matching is None:
#                     question_for_matching = question
                
#                 if question is None:
#                     print(f"⚠️  警告: 样本 {idx} 无法提取问题，跳过")
#                     continue
                
#                 # Get ID
#                 if 'id' in row:
#                     question_id = row['id']
#                 elif 'extra_info' in row and isinstance(row['extra_info'], dict) and 'index' in row['extra_info']:
#                     question_id = row['extra_info']['index']
#                 else:
#                     question_id = global_idx
                
#                 # Get answer（支持嵌套字段）
#                 answer = None
#                 if '.' in answer_key:
#                     # 嵌套访问，例如 'extra_info.answer'
#                     parts = answer_key.split('.')
#                     answer = row.get(parts[0])
#                     if answer is not None:
#                         for part in parts[1:]:
#                             if isinstance(answer, dict):
#                                 answer = answer.get(part)
#                             else:
#                                 answer = None
#                                 break
#                 else:
#                     answer = row.get(answer_key)
                
#                 # 如果答案仍然是None，尝试从extra_info中获取
#                 if answer is None and 'extra_info' in row and isinstance(row['extra_info'], dict):
#                     answer = row['extra_info'].get('answer') or row['extra_info'].get('response')
                
#                 # Store sample info
#                 sample = {
#                     'dataset_name': dataset_name,
#                     'question': question,  # 用于训练的问题文本
#                     'question_for_matching': question_for_matching,  # 用于COT匹配的纯文本
#                     'question_id': question_id,
#                     'answer': answer,
#                     'original_row': row.to_dict()
#                 }
                
#                 self.samples.append(sample)
#                 self.dataset_names.append(dataset_name)
#                 self.questions.append(question_for_matching)  # 存储用于匹配的版本
#                 self.question_ids.append(question_id)
                
#                 global_idx += 1
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, index):
#         """Get a data sample with dataset source information."""
#         sample = self.samples[index]
        
#         question = sample['question']  # 用于训练的问题
#         question_for_matching = sample.get('question_for_matching', question)  # 用于COT匹配
#         dataset_name = sample['dataset_name']
#         question_id = sample['question_id']
        
#         # Apply chat template if available
#         if hasattr(self.tokenizer, 'apply_chat_template'):
#             messages = [{"role": "user", "content": question}]
#             prompt_text = self.tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
#         else:
#             prompt_text = question
        
#         # Tokenize
#         max_prompt_length = self.config.get('max_prompt_length', 512)
#         encoded = self.tokenizer(
#             prompt_text,
#             truncation=True,
#             max_length=max_prompt_length,
#             padding='max_length',  # ← 修复：padding到最大长度，确保所有样本长度一致
#             return_tensors='pt',   # ← 返回 PyTorch Tensor
#         )
        
#         # 🔑 转换为1D tensor（去掉batch维度，因为这是单个样本）
#         # tokenizer 返回的是 (1, seq_len)，我们需要 (seq_len,)
#         encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        
#         # Prepare the output dict
#         result = {
#             'input_ids': encoded['input_ids'],
#             'attention_mask': encoded['attention_mask'],
#             'dataset_name': dataset_name,  # ⭐ Key field for COT matching
#             'question': question_for_matching,  # ⭐ 用于COT匹配的纯文本问题
#             'question_id': int(question_id),
#         }
        
#         # Add answer if available
#         if sample['answer'] is not None:
#             result['data_answer'] = str(sample['answer'])
        
#         return result


# def create_multi_dataset_from_config(config_path: str = None, **kwargs):
#     """
#     Helper function to create multi-dataset from a YAML config file.
#     """
#     import yaml
    
#     if config_path and os.path.exists(config_path):
#         with open(config_path, 'r') as f:
#             config = yaml.safe_load(f)
#         dataset_configs = config['datasets']
#     else:
#         # Fallback to kwargs
#         dataset_configs = kwargs.get('dataset_configs', [])
    
#     return dataset_configs








# # ==============================================================================
# # 调试脚本：用于验证 MultiDatasetWithCOT 是否能正确加载和生成批次
# # ==============================================================================
# if __name__ == '__main__':
#     import torch
#     from transformers import AutoTokenizer
#     from torch.utils.data import DataLoader
#     from verl.utils.dataset.rl_dataset import collate_fn  # 假设 collate_fn 在这个路径下

#     print("🚀 [调试模式] 正在启动数据集加载测试...")

#     # --- 1. 配置 (与您的训练脚本保持一致) ---
#     ACTOR_MODEL_PATH = "/nas/models/Qwen3-8B"
#     TRAIN_FILES_GSM8K = "/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
#     TRAIN_FILES_MATH = "/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
#     BATCH_SIZE = 8
#     MAX_PROMPT_LENGTH = 2048

#     # 模拟 verl 框架传入的 config.data
#     mock_data_config = {
#         'gsm8k_path': TRAIN_FILES_GSM8K,
#         'math_path': TRAIN_FILES_MATH,
#         'max_prompt_length': MAX_PROMPT_LENGTH,
#     }

#     # --- 2. 初始化 Tokenizer ---
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH, trust_remote_code=True)
#         # 确保有 pad_token
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#         print(f"✅ Tokenizer from '{ACTOR_MODEL_PATH}' loaded successfully.")
#     except Exception as e:
#         print(f"❌ Error loading tokenizer: {e}")
#         exit()

#     # --- 3. 初始化训练数据集 ---
#     try:
#         print("\n--- 初始化训练数据集 (is_train=True) ---")
#         train_dataset = MultiDatasetWithCOT(
#             tokenizer=tokenizer,
#             config=mock_data_config,
#             is_train=True
#         )
#         if len(train_dataset) == 0:
#             print("❌ 错误: 训练数据集为空，请检查文件路径和内容。")
#             exit()
        
#         print(f"✅ 训练数据集初始化成功，总样本数: {len(train_dataset)}")

#     except Exception as e:
#         print(f"❌ 错误: 初始化训练数据集时失败: {e}")
#         import traceback
#         traceback.print_exc()
#         exit()

#     # --- 4. 创建 Dataloader 并迭代 ---
#     try:
#         train_dataloader = DataLoader(
#             dataset=train_dataset,
#             batch_size=BATCH_SIZE,
#             collate_fn=collate_fn,
#             shuffle=False  # 关闭 shuffle 以便复现问题
#         )
#         print(f"\n--- 迭代检查前 5 个批次的数据 ---")
        
#         found_bad_batch = False
#         for i, batch_dict in enumerate(train_dataloader):
#             if i >= 5: # 只检查前5个批次
#                 break

#             print(f"\n--- Batch {i+1} ---")
            
#             # 检查 collate_fn 的输出
#             if not batch_dict or not isinstance(batch_dict, dict):
#                 print(f"❌ 严重错误: collate_fn 返回了非字典或空对象: {batch_dict}")
#                 found_bad_batch = True
#                 break

#             print(f"  collate_fn 输出的 keys: {list(batch_dict.keys())}")

#             # 模拟 DataProto 的行为
#             has_tensors = any(isinstance(v, torch.Tensor) for v in batch_dict.values())
            
#             if not has_tensors:
#                 print("❌ 错误: 这个批次中没有任何张量数据！")
#                 print(f"   批次内容: {batch_dict}")
#                 found_bad_batch = True
#                 # 进一步检查是哪个样本导致了问题
#                 start_index = i * BATCH_SIZE
#                 end_index = start_index + BATCH_SIZE
#                 print("   正在检查导致此问题的原始样本...")
#                 for sample_idx in range(start_index, min(end_index, len(train_dataset))):
#                     try:
#                         # 尝试单独处理每个样本
#                         single_sample = train_dataset[sample_idx]
#                         if not single_sample or not single_sample.get('input_ids'):
#                             print(f"   -> 样本 {sample_idx} 是空的或无效的: {train_dataset.samples[sample_idx]}")
#                     except Exception as e:
#                         print(f"   -> 处理样本 {sample_idx} 时出错: {e}")
#                 break
#             else:
#                 print("  ✅ 批次包含张量数据，看起来正常。")

#         if not found_bad_batch:
#             print("\n✅ [调试成功] 前 5 个批次的结构都正常，没有发现空批次。")
#             print("   这意味着问题可能发生在训练循环的其他地方，或者在后续的数据批次中。")

#     except Exception as e:
#         print(f"❌ 错误: 在创建或迭代 Dataloader 时失败: {e}")
#         import traceback
#         traceback.print_exc()








# # Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# Multi-dataset support for GRPO training with COT.

# This dataset class handles multiple datasets (GSM8K, MATH, NuminaMath-CoT, etc.)
# and includes dataset source information for COT matching.
# """

# import os
# import pandas as pd
# from torch.utils.data import Dataset
# from typing import List, Dict, Optional, Any, Union


# class MultiDatasetWithCOT(Dataset):
#     """
#     Dataset that combines multiple datasets and tracks their sources.
    
#     Each sample includes a 'dataset_name' field that identifies which
#     dataset it comes from, enabling correct COT selection.
    
#     Example usage:
#         dataset = MultiDatasetWithCOT(
#             dataset_configs=[
#                 {
#                     "name": "gsm8k",
#                     "files": ["/path/to/gsm8k/train.parquet"],
#                     "prompt_key": "question"
#                 },
#                 {
#                     "name": "math",
#                     "files": ["/path/to/math/train.parquet"],
#                     "prompt_key": "problem"
#                 }
#             ],
#             tokenizer=tokenizer,
#             config=config
#         )
#     """
    
#     def __init__(
#         self,
#         data_files=None,  # ⭐ 兼容标准调用方式（会被忽略）
#         tokenizer=None,
#         processor=None,
#         config=None,
#         dataset_configs: Optional[List[Dict]] = None,
#         # 🆕 简化的配置方式（从config中读取）
#         gsm8k_path: Optional[str] = None,
#         math_path: Optional[str] = None,
#         numina_path: Optional[str] = None,
#         is_train: bool = True,  # 🆕 添加训练/验证标识
#     ):
#         """
#         Initialize the multi-dataset.
        
#         Args:
#             data_files: Ignored (for compatibility with standard dataset interface)
#             tokenizer: Tokenizer for encoding text
#             processor: Processor (can be None)
#             config: Data configuration
#             dataset_configs: List of dataset configurations, each containing:
#                 - name: Dataset identifier (e.g., "gsm8k", "math")
#                 - files: List of file paths
#                 - prompt_key: Field name for the question/problem
#                 - answer_key: (Optional) Field name for the answer
            
#             # 🆕 简化配置（Hydra友好）：
#             gsm8k_path: Path to GSM8K dataset file
#             math_path: Path to MATH dataset file  
#             numina_path: Path to Numina dataset file
#         """
#         self.tokenizer = tokenizer
#         self.processor = processor
#         self.config = config if config is not None else {}
#         self.is_train = is_train
        
#         # 🆕 从config中读取简化路径和字段名（优先级最高）
#         # 🔑 关键修复：验证集模式下不读取训练数据路径
#         # 初始化默认值
#         _gsm8k_path, _math_path, _numina_path = None, None, None

#         if config is not None:
#             # 只有在训练模式下，才从 config 中读取训练数据路径
#             if is_train:
#                 _gsm8k_path = gsm8k_path or config.get('gsm8k_path')
#                 _math_path = math_path or config.get('math_path')
#                 _numina_path = numina_path or config.get('numina_path')
            
#             # 字段名读取（不区分训练/验证模式）
#             gsm8k_prompt_key = config.get('gsm8k_prompt_key', 'prompt')
#             gsm8k_answer_key = config.get('gsm8k_answer_key', 'extra_info.answer')
#             math_prompt_key = config.get('math_prompt_key', 'prompt')
#             math_answer_key = config.get('math_answer_key', 'extra_info.answer')
#             numina_prompt_key = config.get('numina_prompt_key', 'problem')
#             numina_answer_key = config.get('numina_answer_key', 'solution')
#         else:
#             # 没有 config 时的默认字段名
#             gsm8k_prompt_key = 'prompt'
#             gsm8k_answer_key = 'extra_info.answer'
#             math_prompt_key = 'prompt'
#             math_answer_key = 'extra_info.answer'
#             numina_prompt_key = 'problem'
#             numina_answer_key = 'solution'
        
#         # --- 自动构建 dataset_configs ---
        
#         # 如果提供了简化路径，自动构建 dataset_configs
#         if dataset_configs is None and (_gsm8k_path or _math_path or _numina_path):
#             dataset_configs = []
#             if _gsm8k_path:
#                 dataset_configs.append({
#                     'name': 'gsm8k',
#                     'files': [_gsm8k_path],
#                     'prompt_key': gsm8k_prompt_key,
#                     'answer_key': gsm8k_answer_key
#                 })
#             if _math_path:
#                 dataset_configs.append({
#                     'name': 'math',
#                     'files': [_math_path],
#                     'prompt_key': math_prompt_key,
#                     'answer_key': math_answer_key
#                 })
#             if _numina_path:
#                 dataset_configs.append({
#                     'name': 'numina',
#                     'files': [_numina_path],
#                     'prompt_key': numina_prompt_key,
#                     'answer_key': numina_answer_key
#                 })
#             print(f"📝 使用简化配置，自动构建了 {len(dataset_configs)} 个数据集")
#             print(f"   模式: {'训练' if is_train else '验证'}")
        
#         # 🔑 关键修复：处理空的验证集 (确保 self.samples 被清空)
#         is_config_empty = not dataset_configs or all(not config.get('files') for config in dataset_configs)
        
#         if not is_train and is_config_empty:
#             print(f"⚠️ 验证集为空，创建空数据集")
#             dataset_configs = []
        
#         if not dataset_configs:
#             if is_train:
#                 raise ValueError(
#                     "训练集必须提供 dataset_configs 或至少一个数据集路径（gsm8k_path/math_path/numina_path）。"
#                 )
#             else:
#                 print(f"⚠️ 验证集为空，将创建空数据集")
        
#         self.dataset_configs = dataset_configs
        
#         # Store all samples with their metadata
#         self.samples = []
#         self.dataset_names = []
#         self.questions = []
#         self.question_ids = []
        
#         # Load data from all datasets
#         self._load_all_datasets()
        
#         print(f"Loaded {len(self.samples)} total samples from {len(self.dataset_configs)} datasets")
        
#         # 🔍 验证集调试信息
#         if not is_train:
#             print(f"\n🔍 验证集调试信息:")
#             print(f"   样本数量: {len(self.samples)}")
#             if len(self.samples) > 0:
#                 sample = self.samples[0]
#                 print(f"   第一个样本的字段: {list(sample.keys())}")
#                 print(f"   dataset_name: {sample.get('dataset_name', 'N/A')}")
#                 print(f"   question类型: {type(sample.get('question', 'N/A'))}")
#                 print(f"   answer类型: {type(sample.get('answer', 'N/A'))}")
#             else:
#                 print("   ⚠️ 验证集为空！")
    
#     def _load_all_datasets(self):
#         """Load data from all configured datasets."""
#         global_idx = 0
        
#         # 🔑 处理空的验证集
#         if not self.dataset_configs:
#             # 如果配置为空 (即在验证模式下被清空)，则直接返回
#             print(f"⚠️ 验证数据集配置为空，创建空数据集")
#             return
        
#         for dataset_config in self.dataset_configs:
#             dataset_name = dataset_config["name"]
#             data_files = dataset_config["files"]
#             prompt_key = dataset_config.get("prompt_key", "prompt")
#             answer_key = dataset_config.get("answer_key", "extra_info.answer")
            
#             print(f"\nLoading dataset: {dataset_name}")
#             print(f"  Files: {data_files}")
#             print(f"  Prompt key: {prompt_key}")
            
#             # Load files for this dataset
#             if isinstance(data_files, str):
#                 data_files = [data_files]
            
#             dfs = []
#             for file_path in data_files:
#                 if file_path.endswith('.parquet'):
#                     df = pd.read_parquet(file_path)
#                 elif file_path.endswith('.jsonl'):
#                     df = pd.read_json(file_path, lines=True)
#                 elif file_path.endswith('.json'):
#                     df = pd.read_json(file_path)
#                 else:
#                     raise ValueError(f"Unsupported file format: {file_path}")
#                 dfs.append(df)
            
#             dataset_df = pd.concat(dfs, ignore_index=True)
#             print(f"  Loaded {len(dataset_df)} samples")
            
#             # 🔍 调试信息：打印数据集的列名和前几行
#             print(f"\n🔍 调试信息 - 数据集 '{dataset_name}' 的结构:")
#             print(f"  列名: {list(dataset_df.columns)}")
#             print(f"  数据类型:\n{dataset_df.dtypes}")
#             if len(dataset_df) > 0:
#                 print(f"\n  前3行数据预览:")
#                 print(dataset_df.head(3))
#                 print(f"\n  尝试访问的 prompt_key: '{prompt_key}'")
#                 if prompt_key not in dataset_df.columns:
#                     print(f"  ❌ 错误: '{prompt_key}' 不在列名中!")
#                     print(f"  💡 可能的字段名: {[col for col in dataset_df.columns if 'question' in col.lower() or 'prompt' in col.lower() or 'problem' in col.lower()]}")
#                     raise KeyError(f"字段 '{prompt_key}' 不存在。可用字段: {list(dataset_df.columns)}")
#                 else:
#                     print(f"  ✅ '{prompt_key}' 字段存在")
#             print("=" * 80 + "\n")
            
#             # Process each sample
#             for idx, row in dataset_df.iterrows():
#                 # 🔍 智能获取问题文本（支持多种格式）
#                 question = None
#                 question_for_matching = None  # 用于COT匹配的纯文本问题
                
#                 # 方式1：从extra_info.question获取（最优先，用于COT匹配）
#                 if 'extra_info' in row and isinstance(row['extra_info'], dict):
#                     if 'question' in row['extra_info']:
#                         question_for_matching = row['extra_info']['question']
#                         question = question_for_matching  # 默认也用这个
                
#                 # 方式2：从prompt字段获取（用于训练）
#                 if prompt_key == 'prompt' and 'prompt' in row:
#                     prompt_value = row['prompt']
#                     if isinstance(prompt_value, list):
#                         # Chat格式: [{'content': '...', 'role': 'user'}]
#                         if len(prompt_value) > 0 and 'content' in prompt_value[0]:
#                             question = prompt_value[0]['content']
#                     elif isinstance(prompt_value, str):
#                         question = prompt_value
                
#                 # 方式3：支持嵌套字段访问（例如 'extra_info.question'）
#                 elif '.' in prompt_key:
#                     parts = prompt_key.split('.')
#                     question = row[parts[0]]
#                     for part in parts[1:]:
#                         if isinstance(question, dict):
#                             question = question[part]
#                         else:
#                             raise ValueError(f"Cannot access nested field: {prompt_key}")
                
#                 # 方式4：直接字段访问
#                 elif prompt_key in row:
#                     question = row[prompt_key]
                
#                 # 如果question_for_matching还是None，使用question
#                 if question_for_matching is None:
#                     question_for_matching = question
                
#                 if question is None:
#                     print(f"⚠️  警告: 样本 {idx} 无法提取问题，跳过")
#                     continue
                
#                 # Get ID
#                 if 'id' in row:
#                     question_id = row['id']
#                 elif 'extra_info' in row and isinstance(row['extra_info'], dict) and 'index' in row['extra_info']:
#                     question_id = row['extra_info']['index']
#                 else:
#                     question_id = global_idx
                
#                 # Get answer（支持嵌套字段）
#                 answer = None
#                 if '.' in answer_key:
#                     # 嵌套访问，例如 'extra_info.answer'
#                     parts = answer_key.split('.')
#                     answer = row.get(parts[0])
#                     if answer is not None:
#                         for part in parts[1:]:
#                             if isinstance(answer, dict):
#                                 answer = answer.get(part)
#                             else:
#                                 answer = None
#                                 break
#                 else:
#                     answer = row.get(answer_key)
                
#                 # 如果答案仍然是None，尝试从extra_info中获取
#                 if answer is None and 'extra_info' in row and isinstance(row['extra_info'], dict):
#                     answer = row['extra_info'].get('answer') or row['extra_info'].get('response')
                
#                 # Store sample info
#                 sample = {
#                     'dataset_name': dataset_name,
#                     'question': question,  # 用于训练的问题文本
#                     'question_for_matching': question_for_matching,  # 用于COT匹配的纯文本
#                     'question_id': question_id,
#                     'answer': answer,
#                     'original_row': row.to_dict()
#                 }
                
#                 self.samples.append(sample)
#                 self.dataset_names.append(dataset_name)
#                 self.questions.append(question_for_matching)  # 存储用于匹配的版本
#                 self.question_ids.append(question_id)
                
#                 global_idx += 1
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, index):
#         """Get a data sample with dataset source information."""
#         sample = self.samples[index]
        
#         question = sample['question']  # 用于训练的问题
#         question_for_matching = sample.get('question_for_matching', question)  # 用于COT匹配
#         dataset_name = sample['dataset_name']
#         question_id = sample['question_id']
        
#         # Apply chat template if available
#         if hasattr(self.tokenizer, 'apply_chat_template'):
#             messages = [{"role": "user", "content": question}]
#             prompt_text = self.tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
#         else:
#             prompt_text = question
        
#         # Tokenize
#         max_prompt_length = self.config.get('max_prompt_length', 512)
#         encoded = self.tokenizer(
#             prompt_text,
#             truncation=True,
#             max_length=max_prompt_length,
#             padding=False,
#             return_tensors=None,
#         )
        
#         # Prepare the output dict
#         result = {
#             'input_ids': encoded['input_ids'],
#             'attention_mask': encoded['attention_mask'],
#             'dataset_name': dataset_name,  # ⭐ Key field for COT matching
#             'question': question_for_matching,  # ⭐ 用于COT匹配的纯文本问题
#             'question_id': int(question_id),
#         }
        
#         # Add answer if available
#         if sample['answer'] is not None:
#             result['data_answer'] = str(sample['answer'])
        
#         return result


# def create_multi_dataset_from_config(config_path: str = None, **kwargs):
#     """
#     Helper function to create multi-dataset from a YAML config file.
    
#     Config format:
#         datasets:
#           - name: gsm8k
#             files:
#               - /path/to/gsm8k/train.parquet
#             prompt_key: question
#             answer_key: answer
          
#           - name: math
#             files:
#               - /path/to/math/train.parquet
#             prompt_key: problem
#             answer_key: solution
#     """
#     import yaml
    
#     if config_path and os.path.exists(config_path):
#         with open(config_path, 'r') as f:
#             config = yaml.safe_load(f)
#         dataset_configs = config['datasets']
#     else:
#         # Fallback to kwargs
#         dataset_configs = kwargs.get('dataset_configs', [])
    
#     return dataset_configs






# # MultiDatasetWithCOT.py (完整修复后的代码)
# # Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# Multi-dataset support for GRPO training with COT.

# This dataset class handles multiple datasets (GSM8K, MATH, NuminaMath-CoT, etc.)
# and includes dataset source information for COT matching.
# """

# import os
# import pandas as pd
# from torch.utils.data import Dataset
# from typing import List, Dict, Optional, Any, Union


# class MultiDatasetWithCOT(Dataset):
#     """
#     Dataset that combines multiple datasets and tracks their sources.
    
#     Each sample includes a 'dataset_name' field that identifies which
#     dataset it comes from, enabling correct COT selection.
#     """
    
#     def __init__(
#         self,
#         data_files=None,  # ⭐ 兼容标准调用方式（会被忽略）
#         tokenizer=None,
#         processor=None,
#         config=None,
#         dataset_configs: Optional[List[Dict]] = None,
#         # 🆕 简化的配置方式（从config中读取）
#         gsm8k_path: Optional[str] = None,
#         math_path: Optional[str] = None,
#         numina_path: Optional[str] = None,
#         is_train: bool = True,  # 🆕 添加训练/验证标识
#     ):
#         """
#         Initialize the multi-dataset.
#         """
#         self.tokenizer = tokenizer
#         self.processor = processor
#         self.config = config if config is not None else {}
#         self.is_train = is_train
        
#         # 🆕 从config中读取简化路径和字段名（优先级最高）
#         # 初始化默认值
#         _gsm8k_path, _math_path, _numina_path = None, None, None

#         if config is not None:
#             # 只有在训练模式下，才从 config 中读取训练数据路径
#             if is_train:
#                 _gsm8k_path = gsm8k_path or config.get('gsm8k_path')
#                 _math_path = math_path or config.get('math_path')
#                 _numina_path = numina_path or config.get('numina_path')
            
#             # 字段名读取（不区分训练/验证模式）
#             gsm8k_prompt_key = config.get('gsm8k_prompt_key', 'prompt')
#             gsm8k_answer_key = config.get('gsm8k_answer_key', 'extra_info.answer')
#             math_prompt_key = config.get('math_prompt_key', 'prompt')
#             math_answer_key = config.get('math_answer_key', 'extra_info.answer')
#             numina_prompt_key = config.get('numina_prompt_key', 'problem')
#             numina_answer_key = config.get('numina_answer_key', 'solution')
#         else:
#             # 没有 config 时的默认字段名
#             gsm8k_prompt_key = 'prompt'
#             gsm8k_answer_key = 'extra_info.answer'
#             math_prompt_key = 'prompt'
#             math_answer_key = 'extra_info.answer'
#             numina_prompt_key = 'problem'
#             numina_answer_key = 'solution'
        
#         # --- 自动构建 dataset_configs ---
        
#         dataset_configs_list = dataset_configs or []

#         # 🆕 如果提供了简化路径，自动构建 dataset_configs (仅在训练模式下触发简化加载)
#         if self.is_train and not dataset_configs_list and (_gsm8k_path or _math_path or _numina_path):
#             if _gsm8k_path:
#                 dataset_configs_list.append({
#                     'name': 'gsm8k', 'files': [_gsm8k_path],
#                     'prompt_key': gsm8k_prompt_key, 'answer_key': gsm8k_answer_key
#                 })
#             if _math_path:
#                 dataset_configs_list.append({
#                     'name': 'math', 'files': [_math_path],
#                     'prompt_key': math_prompt_key, 'answer_key': math_answer_key
#                 })
#             if _numina_path:
#                 dataset_configs_list.append({
#                     'name': 'numina', 'files': [_numina_path],
#                     'prompt_key': numina_prompt_key, 'answer_key': numina_answer_key
#                 })
#             print(f"📝 使用简化配置，自动构建了 {len(dataset_configs_list)} 个数据集")
#             print(f"   模式: {'训练' if is_train else '验证'}")
        
#         # --- 3. 处理空数据集和断言 ---

#         is_config_empty = not dataset_configs_list or all(not config.get('files') for config in dataset_configs_list)
        
#         if not is_train and is_config_empty:
#             # 🔑 关键修复：验证模式下无配置则提前返回空数据集
#             print(f"⚠️ 验证集路径为空，跳过数据加载。")
#             self.samples = []
#             self.dataset_names = []
#             self.questions = []
#             self.question_ids = []
#             return 
        
#         if not dataset_configs_list and is_train:
#              raise ValueError(
#                 "训练集必须提供 dataset_configs 或至少一个数据集路径（gsm8k_path/math_path/numina_path）。"
#             )
        
#         self.dataset_configs = dataset_configs_list
        
#         # Store all samples with their metadata
#         self.samples = []
#         self.dataset_names = []
#         self.questions = []
#         self.question_ids = []
        
#         # Load data from all datasets
#         self._load_all_datasets()
        
#         print(f"Loaded {len(self.samples)} total samples from {len(self.dataset_configs)} datasets")
        
#         # 🔍 验证集调试信息
#         if not is_train:
#             print(f"\n🔍 验证集调试信息:")
#             print(f"   样本数量: {len(self.samples)}")
#             if len(self.samples) > 0:
#                 sample = self.samples[0]
#                 print(f"   第一个样本的字段: {list(sample.keys())}")
#                 print(f"   dataset_name: {sample.get('dataset_name', 'N/A')}")
#                 print(f"   question类型: {type(sample.get('question', 'N/A'))}")
#                 print(f"   answer类型: {type(sample.get('answer', 'N/A'))}")
#             else:
#                 print("   ⚠️ 验证集为空！")

    
#     def _load_all_datasets(self):
#         """Load data from all configured datasets."""
#         global_idx = 0
        
#         # 🔑 处理空的验证集
#         if not self.dataset_configs:
#             print(f"⚠️ 验证数据集配置为空，创建空数据集")
#             return
        
#         for dataset_config in self.dataset_configs:
#             dataset_name = dataset_config["name"]
#             data_files = dataset_config["files"]
#             prompt_key = dataset_config.get("prompt_key", "prompt")
#             answer_key = dataset_config.get("answer_key", "extra_info.answer")
            
#             print(f"\nLoading dataset: {dataset_name}")
#             print(f"  Files: {data_files}")
#             print(f"  Prompt key: {prompt_key}")
            
#             # Load files for this dataset
#             if isinstance(data_files, str):
#                 data_files = [data_files]
            
#             dfs = []
#             for file_path in data_files:
#                 if file_path.endswith('.parquet'):
#                     df = pd.read_parquet(file_path)
#                 elif file_path.endswith('.jsonl'):
#                     df = pd.read_json(file_path, lines=True)
#                 elif file_path.endswith('.json'):
#                     df = pd.read_json(file_path)
#                 else:
#                     raise ValueError(f"Unsupported file format: {file_path}")
#                 dfs.append(df)
            
#             dataset_df = pd.concat(dfs, ignore_index=True)
#             print(f"  Loaded {len(dataset_df)} samples")
            
#             # 🔍 调试信息：打印数据集的列名和前几行
#             print(f"\n🔍 调试信息 - 数据集 '{dataset_name}' 的结构:")
#             print(f"  列名: {list(dataset_df.columns)}")
#             print(f"  数据类型:\n{dataset_df.dtypes}")
#             if len(dataset_df) > 0:
#                 print(f"\n  前3行数据预览:")
#                 print(dataset_df.head(3))
#                 print(f"\n  尝试访问的 prompt_key: '{prompt_key}'")
#                 if prompt_key not in dataset_df.columns:
#                     print(f"  ❌ 错误: '{prompt_key}' 不在列名中!")
#                     print(f"  💡 可能的字段名: {[col for col in dataset_df.columns if 'question' in col.lower() or 'prompt' in col.lower() or 'problem' in col.lower()]}")
#                     raise KeyError(f"字段 '{prompt_key}' 不存在。可用字段: {list(dataset_df.columns)}")
#                 else:
#                     print(f"  ✅ '{prompt_key}' 字段存在")
#             print("=" * 80 + "\n")
            
#             # Process each sample
#             for idx, row in dataset_df.iterrows():
#                 # 🔍 智能获取问题文本（支持多种格式）
#                 question = None
#                 question_for_matching = None  # 用于COT匹配的纯文本问题
                
#                 # 方式1：从extra_info.question获取（最优先，用于COT匹配）
#                 if 'extra_info' in row and isinstance(row['extra_info'], dict):
#                     if 'question' in row['extra_info']:
#                         question_for_matching = row['extra_info']['question']
#                         question = question_for_matching  # 默认也用这个
                
#                 # 方式2：从prompt字段获取（用于训练）
#                 if prompt_key == 'prompt' and 'prompt' in row:
#                     prompt_value = row['prompt']
#                     if isinstance(prompt_value, list):
#                         # Chat格式: [{'content': '...', 'role': 'user'}]
#                         if len(prompt_value) > 0 and 'content' in prompt_value[0]:
#                             question = prompt_value[0]['content']
#                     elif isinstance(prompt_value, str):
#                         question = prompt_value
                
#                 # 方式3：支持嵌套字段访问（例如 'extra_info.question'）
#                 elif '.' in prompt_key:
#                     parts = prompt_key.split('.')
#                     question = row[parts[0]]
#                     for part in parts[1:]:
#                         if isinstance(question, dict):
#                             question = question[part]
#                         else:
#                             raise ValueError(f"Cannot access nested field: {prompt_key}")
                
#                 # 方式4：直接字段访问
#                 elif prompt_key in row:
#                     question = row[prompt_key]
                
#                 # 如果question_for_matching还是None，使用question
#                 if question_for_matching is None:
#                     question_for_matching = question
                
#                 if question is None:
#                     print(f"⚠️  警告: 样本 {idx} 无法提取问题，跳过")
#                     continue
                
#                 # Get ID
#                 if 'id' in row:
#                     question_id = row['id']
#                 elif 'extra_info' in row and isinstance(row['extra_info'], dict) and 'index' in row['extra_info']:
#                     question_id = row['extra_info']['index']
#                 else:
#                     question_id = global_idx
                
#                 # Get answer（支持嵌套字段）
#                 answer = None
#                 if '.' in answer_key:
#                     # 嵌套访问，例如 'extra_info.answer'
#                     parts = answer_key.split('.')
#                     answer = row.get(parts[0])
#                     if answer is not None:
#                         for part in parts[1:]:
#                             if isinstance(answer, dict):
#                                 answer = answer.get(part)
#                             else:
#                                 answer = None
#                                 break
#                 else:
#                     answer = row.get(answer_key)
                
#                 # 如果答案仍然是None，尝试从extra_info中获取
#                 if answer is None and 'extra_info' in row and isinstance(row['extra_info'], dict):
#                     answer = row['extra_info'].get('answer') or row['extra_info'].get('response')
                
#                 # Store sample info
#                 sample = {
#                     'dataset_name': dataset_name,
#                     'question': question,  # 用于训练的问题文本
#                     'question_for_matching': question_for_matching,  # 用于COT匹配的纯文本
#                     'question_id': question_id,
#                     'answer': answer,
#                     'original_row': row.to_dict()
#                 }
                
#                 self.samples.append(sample)
#                 self.dataset_names.append(dataset_name)
#                 self.questions.append(question_for_matching)  # 存储用于匹配的版本
#                 self.question_ids.append(question_id)
                
#                 global_idx += 1
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, index):
#         """Get a data sample with dataset source information."""
#         sample = self.samples[index]
        
#         question = sample['question']  # 用于训练的问题
#         question_for_matching = sample.get('question_for_matching', question)  # 用于COT匹配
#         dataset_name = sample['dataset_name']
#         question_id = sample['question_id']
        
#         # Apply chat template if available
#         if hasattr(self.tokenizer, 'apply_chat_template'):
#             messages = [{"role": "user", "content": question}]
#             prompt_text = self.tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
#         else:
#             prompt_text = question
        
#         # Tokenize
#         max_prompt_length = self.config.get('max_prompt_length', 512)
#         encoded = self.tokenizer(
#             prompt_text,
#             truncation=True,
#             max_length=max_prompt_length,
#             padding='max_length',  # ← 修复：padding到最大长度，确保所有样本长度一致
#             return_tensors='pt',   # ← 返回 PyTorch Tensor
#         )
        
#         # 🔑 转换为1D tensor（去掉batch维度，因为这是单个样本）
#         # tokenizer 返回的是 (1, seq_len)，我们需要 (seq_len,)
#         encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        
#         # 🔑 生成 position_ids（参考 RLHFDataset 第309行）
#         from verl.utils.model import compute_position_id_with_mask
#         position_ids = compute_position_id_with_mask(encoded['attention_mask'].unsqueeze(0))
#         position_ids = position_ids[0]  # squeeze batch dimension
        
#         # Prepare the output dict
#         result = {
#             'input_ids': encoded['input_ids'],
#             'attention_mask': encoded['attention_mask'],
#             'position_ids': position_ids,  # ⭐ 添加 position_ids（verl 标准要求）
#             'dataset_name': dataset_name,  # ⭐ Key field for COT matching
#             'question': question_for_matching,  # ⭐ 用于COT匹配的纯文本问题
#             'question_id': int(question_id),
#         }
        
#         # Add answer if available
#         if sample['answer'] is not None:
#             result['data_answer'] = str(sample['answer'])
        
#         return result


# def create_multi_dataset_from_config(config_path: str = None, **kwargs):
#     """
#     Helper function to create multi-dataset from a YAML config file.
#     """
#     import yaml
    
#     if config_path and os.path.exists(config_path):
#         with open(config_path, 'r') as f:
#             config = yaml.safe_load(f)
#         dataset_configs = config['datasets']
#     else:
#         # Fallback to kwargs
#         dataset_configs = kwargs.get('dataset_configs', [])
    
#     return dataset_configs








# # ==============================================================================
# # 调试脚本：用于验证 MultiDatasetWithCOT 是否能正确加载和生成批次
# # ==============================================================================
# if __name__ == '__main__':
#     import torch
#     from transformers import AutoTokenizer
#     from torch.utils.data import DataLoader
#     from verl.utils.dataset.rl_dataset import collate_fn  # 假设 collate_fn 在这个路径下

#     print("🚀 [调试模式] 正在启动数据集加载测试...")

#     # --- 1. 配置 (与您的训练脚本保持一致) ---
#     ACTOR_MODEL_PATH = "/nas/models/Qwen3-8B"
#     TRAIN_FILES_GSM8K = "/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
#     TRAIN_FILES_MATH = "/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
#     BATCH_SIZE = 8
#     MAX_PROMPT_LENGTH = 2048

#     # 模拟 verl 框架传入的 config.data
#     mock_data_config = {
#         'gsm8k_path': TRAIN_FILES_GSM8K,
#         'math_path': TRAIN_FILES_MATH,
#         'max_prompt_length': MAX_PROMPT_LENGTH,
#     }

#     # --- 2. 初始化 Tokenizer ---
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH, trust_remote_code=True)
#         # 确保有 pad_token
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#         print(f"✅ Tokenizer from '{ACTOR_MODEL_PATH}' loaded successfully.")
#     except Exception as e:
#         print(f"❌ Error loading tokenizer: {e}")
#         exit()

#     # --- 3. 初始化训练数据集 ---
#     try:
#         print("\n--- 初始化训练数据集 (is_train=True) ---")
#         train_dataset = MultiDatasetWithCOT(
#             tokenizer=tokenizer,
#             config=mock_data_config,
#             is_train=True
#         )
#         if len(train_dataset) == 0:
#             print("❌ 错误: 训练数据集为空，请检查文件路径和内容。")
#             exit()
        
#         print(f"✅ 训练数据集初始化成功，总样本数: {len(train_dataset)}")

#     except Exception as e:
#         print(f"❌ 错误: 初始化训练数据集时失败: {e}")
#         import traceback
#         traceback.print_exc()
#         exit()

#     # --- 4. 创建 Dataloader 并迭代 ---
#     try:
#         train_dataloader = DataLoader(
#             dataset=train_dataset,
#             batch_size=BATCH_SIZE,
#             collate_fn=collate_fn,
#             shuffle=False  # 关闭 shuffle 以便复现问题
#         )
#         print(f"\n--- 迭代检查前 5 个批次的数据 ---")
        
#         found_bad_batch = False
#         for i, batch_dict in enumerate(train_dataloader):
#             if i >= 5: # 只检查前5个批次
#                 break

#             print(f"\n--- Batch {i+1} ---")
            
#             # 检查 collate_fn 的输出
#             if not batch_dict or not isinstance(batch_dict, dict):
#                 print(f"❌ 严重错误: collate_fn 返回了非字典或空对象: {batch_dict}")
#                 found_bad_batch = True
#                 break

#             print(f"  collate_fn 输出的 keys: {list(batch_dict.keys())}")

#             # 模拟 DataProto 的行为
#             has_tensors = any(isinstance(v, torch.Tensor) for v in batch_dict.values())
            
#             if not has_tensors:
#                 print("❌ 错误: 这个批次中没有任何张量数据！")
#                 print(f"   批次内容: {batch_dict}")
#                 found_bad_batch = True
#                 # 进一步检查是哪个样本导致了问题
#                 start_index = i * BATCH_SIZE
#                 end_index = start_index + BATCH_SIZE
#                 print("   正在检查导致此问题的原始样本...")
#                 for sample_idx in range(start_index, min(end_index, len(train_dataset))):
#                     try:
#                         # 尝试单独处理每个样本
#                         single_sample = train_dataset[sample_idx]
#                         if not single_sample or not single_sample.get('input_ids'):
#                             print(f"   -> 样本 {sample_idx} 是空的或无效的: {train_dataset.samples[sample_idx]}")
#                     except Exception as e:
#                         print(f"   -> 处理样本 {sample_idx} 时出错: {e}")
#                 break
#             else:
#                 print("  ✅ 批次包含张量数据，看起来正常。")

#         if not found_bad_batch:
#             print("\n✅ [调试成功] 前 5 个批次的结构都正常，没有发现空批次。")
#             print("   这意味着问题可能发生在训练循环的其他地方，或者在后续的数据批次中。")

#     except Exception as e:
#         print(f"❌ 错误: 在创建或迭代 Dataloader 时失败: {e}")
#         import traceback
#         traceback.print_exc()









# # Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# Multi-dataset support for GRPO training with COT.

# This dataset class handles multiple datasets (GSM8K, MATH, NuminaMath-CoT, etc.)
# and includes dataset source information for COT matching.
# """

# import os
# import pandas as pd
# from torch.utils.data import Dataset
# from typing import List, Dict, Optional, Any, Union


# class MultiDatasetWithCOT(Dataset):
#     """
#     Dataset that combines multiple datasets and tracks their sources.
    
#     Each sample includes a 'dataset_name' field that identifies which
#     dataset it comes from, enabling correct COT selection.
    
#     Example usage:
#         dataset = MultiDatasetWithCOT(
#             dataset_configs=[
#                 {
#                     "name": "gsm8k",
#                     "files": ["/path/to/gsm8k/train.parquet"],
#                     "prompt_key": "question"
#                 },
#                 {
#                     "name": "math",
#                     "files": ["/path/to/math/train.parquet"],
#                     "prompt_key": "problem"
#                 }
#             ],
#             tokenizer=tokenizer,
#             config=config
#         )
#     """
    
#     def __init__(
#         self,
#         data_files=None,  # ⭐ 兼容标准调用方式（会被忽略）
#         tokenizer=None,
#         processor=None,
#         config=None,
#         dataset_configs: Optional[List[Dict]] = None,
#         # 🆕 简化的配置方式（从config中读取）
#         gsm8k_path: Optional[str] = None,
#         math_path: Optional[str] = None,
#         numina_path: Optional[str] = None,
#         is_train: bool = True,  # 🆕 添加训练/验证标识
#     ):
#         """
#         Initialize the multi-dataset.
        
#         Args:
#             data_files: Ignored (for compatibility with standard dataset interface)
#             tokenizer: Tokenizer for encoding text
#             processor: Processor (can be None)
#             config: Data configuration
#             dataset_configs: List of dataset configurations, each containing:
#                 - name: Dataset identifier (e.g., "gsm8k", "math")
#                 - files: List of file paths
#                 - prompt_key: Field name for the question/problem
#                 - answer_key: (Optional) Field name for the answer
            
#             # 🆕 简化配置（Hydra友好）：
#             gsm8k_path: Path to GSM8K dataset file
#             math_path: Path to MATH dataset file  
#             numina_path: Path to Numina dataset file
#         """
#         self.tokenizer = tokenizer
#         self.processor = processor
#         self.config = config if config is not None else {}
#         self.is_train = is_train
        
#         # 🆕 从config中读取简化路径和字段名（优先级最高）
#         # 🔑 关键修复：验证集模式下不读取训练数据路径
#         # 初始化默认值
#         _gsm8k_path, _math_path, _numina_path = None, None, None

#         if config is not None:
#             # 只有在训练模式下，才从 config 中读取训练数据路径
#             if is_train:
#                 _gsm8k_path = gsm8k_path or config.get('gsm8k_path')
#                 _math_path = math_path or config.get('math_path')
#                 _numina_path = numina_path or config.get('numina_path')
            
#             # 字段名读取（不区分训练/验证模式）
#             gsm8k_prompt_key = config.get('gsm8k_prompt_key', 'prompt')
#             gsm8k_answer_key = config.get('gsm8k_answer_key', 'extra_info.answer')
#             math_prompt_key = config.get('math_prompt_key', 'prompt')
#             math_answer_key = config.get('math_answer_key', 'extra_info.answer')
#             numina_prompt_key = config.get('numina_prompt_key', 'problem')
#             numina_answer_key = config.get('numina_answer_key', 'solution')
#         else:
#             # 没有 config 时的默认字段名
#             gsm8k_prompt_key = 'prompt'
#             gsm8k_answer_key = 'extra_info.answer'
#             math_prompt_key = 'prompt'
#             math_answer_key = 'extra_info.answer'
#             numina_prompt_key = 'problem'
#             numina_answer_key = 'solution'
        
#         # --- 自动构建 dataset_configs ---
        
#         # 如果提供了简化路径，自动构建 dataset_configs
#         if dataset_configs is None and (_gsm8k_path or _math_path or _numina_path):
#             dataset_configs = []
#             if _gsm8k_path:
#                 dataset_configs.append({
#                     'name': 'gsm8k',
#                     'files': [_gsm8k_path],
#                     'prompt_key': gsm8k_prompt_key,
#                     'answer_key': gsm8k_answer_key
#                 })
#             if _math_path:
#                 dataset_configs.append({
#                     'name': 'math',
#                     'files': [_math_path],
#                     'prompt_key': math_prompt_key,
#                     'answer_key': math_answer_key
#                 })
#             if _numina_path:
#                 dataset_configs.append({
#                     'name': 'numina',
#                     'files': [_numina_path],
#                     'prompt_key': numina_prompt_key,
#                     'answer_key': numina_answer_key
#                 })
#             print(f"📝 使用简化配置，自动构建了 {len(dataset_configs)} 个数据集")
#             print(f"   模式: {'训练' if is_train else '验证'}")
        
#         # 🔑 关键修复：处理空的验证集 (确保 self.samples 被清空)
#         is_config_empty = not dataset_configs or all(not config.get('files') for config in dataset_configs)
        
#         if not is_train and is_config_empty:
#             print(f"⚠️ 验证集为空，创建空数据集")
#             dataset_configs = []
        
#         if not dataset_configs:
#             if is_train:
#                 raise ValueError(
#                     "训练集必须提供 dataset_configs 或至少一个数据集路径（gsm8k_path/math_path/numina_path）。"
#                 )
#             else:
#                 print(f"⚠️ 验证集为空，将创建空数据集")
        
#         self.dataset_configs = dataset_configs
        
#         # Store all samples with their metadata
#         self.samples = []
#         self.dataset_names = []
#         self.questions = []
#         self.question_ids = []
        
#         # Load data from all datasets
#         self._load_all_datasets()
        
#         print(f"Loaded {len(self.samples)} total samples from {len(self.dataset_configs)} datasets")
        
#         # 🔍 验证集调试信息
#         if not is_train:
#             print(f"\n🔍 验证集调试信息:")
#             print(f"   样本数量: {len(self.samples)}")
#             if len(self.samples) > 0:
#                 sample = self.samples[0]
#                 print(f"   第一个样本的字段: {list(sample.keys())}")
#                 print(f"   dataset_name: {sample.get('dataset_name', 'N/A')}")
#                 print(f"   question类型: {type(sample.get('question', 'N/A'))}")
#                 print(f"   answer类型: {type(sample.get('answer', 'N/A'))}")
#             else:
#                 print("   ⚠️ 验证集为空！")
    
#     def _load_all_datasets(self):
#         """Load data from all configured datasets."""
#         global_idx = 0
        
#         # 🔑 处理空的验证集
#         if not self.dataset_configs:
#             # 如果配置为空 (即在验证模式下被清空)，则直接返回
#             print(f"⚠️ 验证数据集配置为空，创建空数据集")
#             return
        
#         for dataset_config in self.dataset_configs:
#             dataset_name = dataset_config["name"]
#             data_files = dataset_config["files"]
#             prompt_key = dataset_config.get("prompt_key", "prompt")
#             answer_key = dataset_config.get("answer_key", "extra_info.answer")
            
#             print(f"\nLoading dataset: {dataset_name}")
#             print(f"  Files: {data_files}")
#             print(f"  Prompt key: {prompt_key}")
            
#             # Load files for this dataset
#             if isinstance(data_files, str):
#                 data_files = [data_files]
            
#             dfs = []
#             for file_path in data_files:
#                 if file_path.endswith('.parquet'):
#                     df = pd.read_parquet(file_path)
#                 elif file_path.endswith('.jsonl'):
#                     df = pd.read_json(file_path, lines=True)
#                 elif file_path.endswith('.json'):
#                     df = pd.read_json(file_path)
#                 else:
#                     raise ValueError(f"Unsupported file format: {file_path}")
#                 dfs.append(df)
            
#             dataset_df = pd.concat(dfs, ignore_index=True)
#             print(f"  Loaded {len(dataset_df)} samples")
            
#             # 🔍 调试信息：打印数据集的列名和前几行
#             print(f"\n🔍 调试信息 - 数据集 '{dataset_name}' 的结构:")
#             print(f"  列名: {list(dataset_df.columns)}")
#             print(f"  数据类型:\n{dataset_df.dtypes}")
#             if len(dataset_df) > 0:
#                 print(f"\n  前3行数据预览:")
#                 print(dataset_df.head(3))
#                 print(f"\n  尝试访问的 prompt_key: '{prompt_key}'")
#                 if prompt_key not in dataset_df.columns:
#                     print(f"  ❌ 错误: '{prompt_key}' 不在列名中!")
#                     print(f"  💡 可能的字段名: {[col for col in dataset_df.columns if 'question' in col.lower() or 'prompt' in col.lower() or 'problem' in col.lower()]}")
#                     raise KeyError(f"字段 '{prompt_key}' 不存在。可用字段: {list(dataset_df.columns)}")
#                 else:
#                     print(f"  ✅ '{prompt_key}' 字段存在")
#             print("=" * 80 + "\n")
            
#             # Process each sample
#             for idx, row in dataset_df.iterrows():
#                 # 🔍 智能获取问题文本（支持多种格式）
#                 question = None
#                 question_for_matching = None  # 用于COT匹配的纯文本问题
                
#                 # 方式1：从extra_info.question获取（最优先，用于COT匹配）
#                 if 'extra_info' in row and isinstance(row['extra_info'], dict):
#                     if 'question' in row['extra_info']:
#                         question_for_matching = row['extra_info']['question']
#                         question = question_for_matching  # 默认也用这个
                
#                 # 方式2：从prompt字段获取（用于训练）
#                 if prompt_key == 'prompt' and 'prompt' in row:
#                     prompt_value = row['prompt']
#                     if isinstance(prompt_value, list):
#                         # Chat格式: [{'content': '...', 'role': 'user'}]
#                         if len(prompt_value) > 0 and 'content' in prompt_value[0]:
#                             question = prompt_value[0]['content']
#                     elif isinstance(prompt_value, str):
#                         question = prompt_value
                
#                 # 方式3：支持嵌套字段访问（例如 'extra_info.question'）
#                 elif '.' in prompt_key:
#                     parts = prompt_key.split('.')
#                     question = row[parts[0]]
#                     for part in parts[1:]:
#                         if isinstance(question, dict):
#                             question = question[part]
#                         else:
#                             raise ValueError(f"Cannot access nested field: {prompt_key}")
                
#                 # 方式4：直接字段访问
#                 elif prompt_key in row:
#                     question = row[prompt_key]
                
#                 # 如果question_for_matching还是None，使用question
#                 if question_for_matching is None:
#                     question_for_matching = question
                
#                 if question is None:
#                     print(f"⚠️  警告: 样本 {idx} 无法提取问题，跳过")
#                     continue
                
#                 # Get ID
#                 if 'id' in row:
#                     question_id = row['id']
#                 elif 'extra_info' in row and isinstance(row['extra_info'], dict) and 'index' in row['extra_info']:
#                     question_id = row['extra_info']['index']
#                 else:
#                     question_id = global_idx
                
#                 # Get answer（支持嵌套字段）
#                 answer = None
#                 if '.' in answer_key:
#                     # 嵌套访问，例如 'extra_info.answer'
#                     parts = answer_key.split('.')
#                     answer = row.get(parts[0])
#                     if answer is not None:
#                         for part in parts[1:]:
#                             if isinstance(answer, dict):
#                                 answer = answer.get(part)
#                             else:
#                                 answer = None
#                                 break
#                 else:
#                     answer = row.get(answer_key)
                
#                 # 如果答案仍然是None，尝试从extra_info中获取
#                 if answer is None and 'extra_info' in row and isinstance(row['extra_info'], dict):
#                     answer = row['extra_info'].get('answer') or row['extra_info'].get('response')
                
#                 # Store sample info
#                 sample = {
#                     'dataset_name': dataset_name,
#                     'question': question,  # 用于训练的问题文本
#                     'question_for_matching': question_for_matching,  # 用于COT匹配的纯文本
#                     'question_id': question_id,
#                     'answer': answer,
#                     'original_row': row.to_dict()
#                 }
                
#                 self.samples.append(sample)
#                 self.dataset_names.append(dataset_name)
#                 self.questions.append(question_for_matching)  # 存储用于匹配的版本
#                 self.question_ids.append(question_id)
                
#                 global_idx += 1
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, index):
#         """Get a data sample with dataset source information."""
#         sample = self.samples[index]
        
#         question = sample['question']  # 用于训练的问题
#         question_for_matching = sample.get('question_for_matching', question)  # 用于COT匹配
#         dataset_name = sample['dataset_name']
#         question_id = sample['question_id']
        
#         # Apply chat template if available
#         if hasattr(self.tokenizer, 'apply_chat_template'):
#             messages = [{"role": "user", "content": question}]
#             prompt_text = self.tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
#         else:
#             prompt_text = question
        
#         # Tokenize
#         max_prompt_length = self.config.get('max_prompt_length', 512)
#         encoded = self.tokenizer(
#             prompt_text,
#             truncation=True,
#             max_length=max_prompt_length,
#             padding=False,
#             return_tensors=None,
#         )
        
#         # Prepare the output dict
#         result = {
#             'input_ids': encoded['input_ids'],
#             'attention_mask': encoded['attention_mask'],
#             'dataset_name': dataset_name,  # ⭐ Key field for COT matching
#             'question': question_for_matching,  # ⭐ 用于COT匹配的纯文本问题
#             'question_id': int(question_id),
#         }
        
#         # Add answer if available
#         if sample['answer'] is not None:
#             result['data_answer'] = str(sample['answer'])
        
#         return result


# def create_multi_dataset_from_config(config_path: str = None, **kwargs):
#     """
#     Helper function to create multi-dataset from a YAML config file.
    
#     Config format:
#         datasets:
#           - name: gsm8k
#             files:
#               - /path/to/gsm8k/train.parquet
#             prompt_key: question
#             answer_key: answer
          
#           - name: math
#             files:
#               - /path/to/math/train.parquet
#             prompt_key: problem
#             answer_key: solution
#     """
#     import yaml
    
#     if config_path and os.path.exists(config_path):
#         with open(config_path, 'r') as f:
#             config = yaml.safe_load(f)
#         dataset_configs = config['datasets']
#     else:
#         # Fallback to kwargs
#         dataset_configs = kwargs.get('dataset_configs', [])
    
#     return dataset_configs




# MultiDatasetWithCOT.py (完整修复后的代码)
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
"""
Multi-dataset support for GRPO training with COT.

This dataset class handles multiple datasets (GSM8K, MATH, NuminaMath-CoT, etc.)
and includes dataset source information for COT matching.
"""

import os
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Any, Union


class MultiDatasetWithCOT(Dataset):
    """
    Dataset that combines multiple datasets and tracks their sources.
    
    Each sample includes a 'dataset_name' field that identifies which
    dataset it comes from, enabling correct COT selection.
    """
    
    def __init__(
        self,
        data_files=None,  # ⭐ 兼容标准调用方式（会被忽略）
        tokenizer=None,
        processor=None,
        config=None,
        dataset_configs: Optional[List[Dict]] = None,
        # 🆕 简化的配置方式（从config中读取）
        gsm8k_path: Optional[str] = None,
        math_path: Optional[str] = None,
        numina_path: Optional[str] = None,
        is_train: bool = True,  # 🆕 添加训练/验证标识
    ):
        """
        Initialize the multi-dataset.
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config if config is not None else {}
        self.is_train = is_train
        
        # 🆕 从config中读取简化路径和字段名（优先级最高）
        # 初始化默认值
        _gsm8k_path, _math_path, _numina_path = None, None, None

        if config is not None:
            # 只有在训练模式下，才从 config 中读取训练数据路径
            if is_train:
                _gsm8k_path = gsm8k_path or config.get('gsm8k_path')
                _math_path = math_path or config.get('math_path')
                _numina_path = numina_path or config.get('numina_path')
            
            # 字段名读取（不区分训练/验证模式）
            gsm8k_prompt_key = config.get('gsm8k_prompt_key', 'prompt')
            gsm8k_answer_key = config.get('gsm8k_answer_key', 'extra_info.answer')
            math_prompt_key = config.get('math_prompt_key', 'prompt')
            math_answer_key = config.get('math_answer_key', 'extra_info.answer')
            numina_prompt_key = config.get('numina_prompt_key', 'problem')
            numina_answer_key = config.get('numina_answer_key', 'solution')
        else:
            # 没有 config 时的默认字段名
            gsm8k_prompt_key = 'prompt'
            gsm8k_answer_key = 'extra_info.answer'
            math_prompt_key = 'prompt'
            math_answer_key = 'extra_info.answer'
            numina_prompt_key = 'problem'
            numina_answer_key = 'solution'
        
        # --- 自动构建 dataset_configs ---
        
        dataset_configs_list = dataset_configs or []

        # 🆕 如果提供了简化路径，自动构建 dataset_configs (仅在训练模式下触发简化加载)
        if self.is_train and not dataset_configs_list and (_gsm8k_path or _math_path or _numina_path):
            if _gsm8k_path:
                dataset_configs_list.append({
                    'name': 'gsm8k', 'files': [_gsm8k_path],
                    'prompt_key': gsm8k_prompt_key, 'answer_key': gsm8k_answer_key
                })
            if _math_path:
                dataset_configs_list.append({
                    'name': 'math', 'files': [_math_path],
                    'prompt_key': math_prompt_key, 'answer_key': math_answer_key
                })
            if _numina_path:
                dataset_configs_list.append({
                    'name': 'numina', 'files': [_numina_path],
                    'prompt_key': numina_prompt_key, 'answer_key': numina_answer_key
                })
            print(f"📝 使用简化配置，自动构建了 {len(dataset_configs_list)} 个数据集")
            print(f"   模式: {'训练' if is_train else '验证'}")
        
        # --- 3. 处理空数据集和断言 ---

        is_config_empty = not dataset_configs_list or all(not config.get('files') for config in dataset_configs_list)
        
        if not is_train and is_config_empty:
            # 🔑 关键修复：验证模式下无配置则提前返回空数据集
            print(f"⚠️ 验证集路径为空，跳过数据加载。")
            self.samples = []
            self.dataset_names = []
            self.questions = []
            self.question_ids = []
            return 
        
        if not dataset_configs_list and is_train:
             raise ValueError(
                "训练集必须提供 dataset_configs 或至少一个数据集路径（gsm8k_path/math_path/numina_path）。"
            )
        
        self.dataset_configs = dataset_configs_list
        
        # Store all samples with their metadata
        self.samples = []
        self.dataset_names = []
        self.questions = []
        self.question_ids = []
        
        # Load data from all datasets
        self._load_all_datasets()
        
        print(f"Loaded {len(self.samples)} total samples from {len(self.dataset_configs)} datasets")
        
        # 🔍 验证集调试信息
        if not is_train:
            print(f"\n🔍 验证集调试信息:")
            print(f"   样本数量: {len(self.samples)}")
            if len(self.samples) > 0:
                sample = self.samples[0]
                print(f"   第一个样本的字段: {list(sample.keys())}")
                print(f"   dataset_name: {sample.get('dataset_name', 'N/A')}")
                print(f"   question类型: {type(sample.get('question', 'N/A'))}")
                print(f"   answer类型: {type(sample.get('answer', 'N/A'))}")
            else:
                print("   ⚠️ 验证集为空！")

    
    def _load_all_datasets(self):
        """Load data from all configured datasets."""
        global_idx = 0
        
        # 🔑 处理空的验证集
        if not self.dataset_configs:
            print(f"⚠️ 验证数据集配置为空，创建空数据集")
            return
        
        for dataset_config in self.dataset_configs:
            dataset_name = dataset_config["name"]
            data_files = dataset_config["files"]
            prompt_key = dataset_config.get("prompt_key", "prompt")
            answer_key = dataset_config.get("answer_key", "extra_info.answer")
            
            print(f"\nLoading dataset: {dataset_name}")
            print(f"  Files: {data_files}")
            print(f"  Prompt key: {prompt_key}")
            
            # Load files for this dataset
            if isinstance(data_files, str):
                data_files = [data_files]
            
            dfs = []
            for file_path in data_files:
                if file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                elif file_path.endswith('.jsonl'):
                    df = pd.read_json(file_path, lines=True)
                elif file_path.endswith('.json'):
                    df = pd.read_json(file_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_path}")
                dfs.append(df)
            
            dataset_df = pd.concat(dfs, ignore_index=True)
            print(f"  Loaded {len(dataset_df)} samples")
            
            # 🔍 调试信息：打印数据集的列名和前几行
            print(f"\n🔍 调试信息 - 数据集 '{dataset_name}' 的结构:")
            print(f"  列名: {list(dataset_df.columns)}")
            print(f"  数据类型:\n{dataset_df.dtypes}")
            if len(dataset_df) > 0:
                print(f"\n  前3行数据预览:")
                print(dataset_df.head(3))
                print(f"\n  尝试访问的 prompt_key: '{prompt_key}'")
                if prompt_key not in dataset_df.columns:
                    print(f"  ❌ 错误: '{prompt_key}' 不在列名中!")
                    print(f"  💡 可能的字段名: {[col for col in dataset_df.columns if 'question' in col.lower() or 'prompt' in col.lower() or 'problem' in col.lower()]}")
                    raise KeyError(f"字段 '{prompt_key}' 不存在。可用字段: {list(dataset_df.columns)}")
                else:
                    print(f"  ✅ '{prompt_key}' 字段存在")
            print("=" * 80 + "\n")
            
            # Process each sample
            for idx, row in dataset_df.iterrows():
                # 🔍 智能获取问题文本（支持多种格式）
                question = None
                question_for_matching = None  # 用于COT匹配的纯文本问题
                
                # 方式1：从extra_info.question获取（最优先，用于COT匹配）
                if 'extra_info' in row and isinstance(row['extra_info'], dict):
                    if 'question' in row['extra_info']:
                        question_for_matching = row['extra_info']['question']
                        question = question_for_matching  # 默认也用这个
                
                # 方式2：从prompt字段获取（用于训练）
                if prompt_key == 'prompt' and 'prompt' in row:
                    prompt_value = row['prompt']
                    if isinstance(prompt_value, list):
                        # Chat格式: [{'content': '...', 'role': 'user'}]
                        if len(prompt_value) > 0 and 'content' in prompt_value[0]:
                            question = prompt_value[0]['content']
                    elif isinstance(prompt_value, str):
                        question = prompt_value
                
                # 方式3：支持嵌套字段访问（例如 'extra_info.question'）
                elif '.' in prompt_key:
                    parts = prompt_key.split('.')
                    question = row[parts[0]]
                    for part in parts[1:]:
                        if isinstance(question, dict):
                            question = question[part]
                        else:
                            raise ValueError(f"Cannot access nested field: {prompt_key}")
                
                # 方式4：直接字段访问
                elif prompt_key in row:
                    question = row[prompt_key]
                
                # 如果question_for_matching还是None，使用question
                if question_for_matching is None:
                    question_for_matching = question
                
                if question is None:
                    print(f"⚠️  警告: 样本 {idx} 无法提取问题，跳过")
                    continue
                
                # Get ID
                if 'id' in row:
                    question_id = row['id']
                elif 'extra_info' in row and isinstance(row['extra_info'], dict) and 'index' in row['extra_info']:
                    question_id = row['extra_info']['index']
                else:
                    question_id = global_idx
                
                # Get answer（支持嵌套字段）
                answer = None
                if '.' in answer_key:
                    # 嵌套访问，例如 'extra_info.answer'
                    parts = answer_key.split('.')
                    answer = row.get(parts[0])
                    if answer is not None:
                        for part in parts[1:]:
                            if isinstance(answer, dict):
                                answer = answer.get(part)
                            else:
                                answer = None
                                break
                else:
                    answer = row.get(answer_key)
                
                # 如果答案仍然是None，尝试从extra_info中获取
                if answer is None and 'extra_info' in row and isinstance(row['extra_info'], dict):
                    answer = row['extra_info'].get('answer') or row['extra_info'].get('response')
                
                # Store sample info
                sample = {
                    'dataset_name': dataset_name,
                    'question': question,  # 用于训练的问题文本
                    'question_for_matching': question_for_matching,  # 用于COT匹配的纯文本
                    'question_id': question_id,
                    'answer': answer,
                    'original_row': row.to_dict()
                }
                
                self.samples.append(sample)
                self.dataset_names.append(dataset_name)
                self.questions.append(question_for_matching)  # 存储用于匹配的版本
                self.question_ids.append(question_id)
                
                global_idx += 1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """Get a data sample with dataset source information."""
        sample = self.samples[index]
        
        question = sample['question']  # 用于训练的问题
        question_for_matching = sample.get('question_for_matching', question)  # 用于COT匹配
        dataset_name = sample['dataset_name']
        question_id = sample['question_id']
        
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": question}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt_text = question
        
        # Tokenize
        max_prompt_length = self.config.get('max_prompt_length', 512)
        encoded = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_prompt_length,
            padding='max_length',  # ← 修复：padding到最大长度，确保所有样本长度一致
            return_tensors='pt',   # ← 返回 PyTorch Tensor
        )
        
        # 🔑 转换为1D tensor（去掉batch维度，因为这是单个样本）
        # tokenizer 返回的是 (1, seq_len)，我们需要 (seq_len,)
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        
        # 🔑 生成 position_ids（参考 RLHFDataset 第309行）
        from verl.utils.model import compute_position_id_with_mask
        position_ids = compute_position_id_with_mask(encoded['attention_mask'].unsqueeze(0))
        position_ids = position_ids[0]  # squeeze batch dimension
        
        # Prepare the output dict
        result = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'position_ids': position_ids,  # ⭐ 添加 position_ids（verl 标准要求）
            'dataset_name': dataset_name,  # ⭐ Key field for COT matching
            'question': question_for_matching,  # ⭐ 用于COT匹配的纯文本问题
            'question_id': int(question_id),
        }
        
        # 🔑 添加 reward_model 字段（verl 标准要求）
        # reward_fn 需要这个字段来获取 ground_truth
        original_row = sample['original_row']
        if 'reward_model' in original_row:
            result['reward_model'] = original_row['reward_model']
        else:
            # 如果原始数据没有 reward_model 字段，使用 answer 构造一个
            if sample['answer'] is not None:
                result['reward_model'] = {
                    'style': 'rule',
                    'ground_truth': str(sample['answer'])
                }
        
        # Add answer if available (向后兼容)
        if sample['answer'] is not None:
            result['data_answer'] = str(sample['answer'])
        
        return result


def create_multi_dataset_from_config(config_path: str = None, **kwargs):
    """
    Helper function to create multi-dataset from a YAML config file.
    """
    import yaml
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        dataset_configs = config['datasets']
    else:
        # Fallback to kwargs
        dataset_configs = kwargs.get('dataset_configs', [])
    
    return dataset_configs








# ==============================================================================
# 调试脚本：用于验证 MultiDatasetWithCOT 是否能正确加载和生成批次
# ==============================================================================
if __name__ == '__main__':
    import torch
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from verl.utils.dataset.rl_dataset import collate_fn  # 假设 collate_fn 在这个路径下

    print("🚀 [调试模式] 正在启动数据集加载测试...")

    # --- 1. 配置 (与您的训练脚本保持一致) ---
    ACTOR_MODEL_PATH = "/nas/models/Qwen3-8B"
    TRAIN_FILES_GSM8K = "/nas/dhl/Datasets/my_Datasets/gsm8k/train.parquet"
    TRAIN_FILES_MATH = "/nas/dhl/Datasets/my_Datasets/MATH/train.parquet"
    BATCH_SIZE = 8
    MAX_PROMPT_LENGTH = 2048

    # 模拟 verl 框架传入的 config.data
    mock_data_config = {
        'gsm8k_path': TRAIN_FILES_GSM8K,
        'math_path': TRAIN_FILES_MATH,
        'max_prompt_length': MAX_PROMPT_LENGTH,
    }

    # --- 2. 初始化 Tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH, trust_remote_code=True)
        # 确保有 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer from '{ACTOR_MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        exit()

    # --- 3. 初始化训练数据集 ---
    try:
        print("\n--- 初始化训练数据集 (is_train=True) ---")
        train_dataset = MultiDatasetWithCOT(
            tokenizer=tokenizer,
            config=mock_data_config,
            is_train=True
        )
        if len(train_dataset) == 0:
            print("❌ 错误: 训练数据集为空，请检查文件路径和内容。")
            exit()
        
        print(f"✅ 训练数据集初始化成功，总样本数: {len(train_dataset)}")

    except Exception as e:
        print(f"❌ 错误: 初始化训练数据集时失败: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- 4. 创建 Dataloader 并迭代 ---
    try:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=False  # 关闭 shuffle 以便复现问题
        )
        print(f"\n--- 迭代检查前 5 个批次的数据 ---")
        
        found_bad_batch = False
        for i, batch_dict in enumerate(train_dataloader):
            if i >= 5: # 只检查前5个批次
                break

            print(f"\n--- Batch {i+1} ---")
            
            # 检查 collate_fn 的输出
            if not batch_dict or not isinstance(batch_dict, dict):
                print(f"❌ 严重错误: collate_fn 返回了非字典或空对象: {batch_dict}")
                found_bad_batch = True
                break

            print(f"  collate_fn 输出的 keys: {list(batch_dict.keys())}")

            # 模拟 DataProto 的行为
            has_tensors = any(isinstance(v, torch.Tensor) for v in batch_dict.values())
            
            if not has_tensors:
                print("❌ 错误: 这个批次中没有任何张量数据！")
                print(f"   批次内容: {batch_dict}")
                found_bad_batch = True
                # 进一步检查是哪个样本导致了问题
                start_index = i * BATCH_SIZE
                end_index = start_index + BATCH_SIZE
                print("   正在检查导致此问题的原始样本...")
                for sample_idx in range(start_index, min(end_index, len(train_dataset))):
                    try:
                        # 尝试单独处理每个样本
                        single_sample = train_dataset[sample_idx]
                        if not single_sample or not single_sample.get('input_ids'):
                            print(f"   -> 样本 {sample_idx} 是空的或无效的: {train_dataset.samples[sample_idx]}")
                    except Exception as e:
                        print(f"   -> 处理样本 {sample_idx} 时出错: {e}")
                break
            else:
                print("  ✅ 批次包含张量数据，看起来正常。")

        if not found_bad_batch:
            print("\n✅ [调试成功] 前 5 个批次的结构都正常，没有发现空批次。")
            print("   这意味着问题可能发生在训练循环的其他地方，或者在后续的数据批次中。")

    except Exception as e:
        print(f"❌ 错误: 在创建或迭代 Dataloader 时失败: {e}")
        import traceback
        traceback.print_exc()