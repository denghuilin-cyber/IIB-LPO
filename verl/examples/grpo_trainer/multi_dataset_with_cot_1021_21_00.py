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
from typing import List, Dict, Optional


class MultiDatasetWithCOT(Dataset):
    """
    Dataset that combines multiple datasets and tracks their sources.
    
    Each sample includes a 'dataset_name' field that identifies which
    dataset it comes from, enabling correct COT selection.
    
    Example usage:
        dataset = MultiDatasetWithCOT(
            dataset_configs=[
                {
                    "name": "gsm8k",
                    "files": ["/path/to/gsm8k/train.parquet"],
                    "prompt_key": "question"
                },
                {
                    "name": "math",
                    "files": ["/path/to/math/train.parquet"],
                    "prompt_key": "problem"
                }
            ],
            tokenizer=tokenizer,
            config=config
        )
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
        
        Args:
            data_files: Ignored (for compatibility with standard dataset interface)
            tokenizer: Tokenizer for encoding text
            processor: Processor (can be None)
            config: Data configuration
            dataset_configs: List of dataset configurations, each containing:
                - name: Dataset identifier (e.g., "gsm8k", "math")
                - files: List of file paths
                - prompt_key: Field name for the question/problem
                - answer_key: (Optional) Field name for the answer
            
            # 🆕 简化配置（Hydra友好）：
            gsm8k_path: Path to GSM8K dataset file
            math_path: Path to MATH dataset file  
            numina_path: Path to Numina dataset file
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config if config is not None else {}
        
        # 🆕 从config中读取简化路径和字段名（优先级最高）
        if config is not None:
            gsm8k_path = gsm8k_path or config.get('gsm8k_path')
            math_path = math_path or config.get('math_path')
            numina_path = numina_path or config.get('numina_path')
            
            # 允许自定义字段名
            gsm8k_prompt_key = config.get('gsm8k_prompt_key', 'prompt')  # 默认用'prompt'
            gsm8k_answer_key = config.get('gsm8k_answer_key', 'extra_info.answer')  # 嵌套字段
            math_prompt_key = config.get('math_prompt_key', 'prompt')    # 默认用'prompt'
            math_answer_key = config.get('math_answer_key', 'extra_info.answer')  # 嵌套字段
            numina_prompt_key = config.get('numina_prompt_key', 'problem')
            numina_answer_key = config.get('numina_answer_key', 'solution')
        else:
            gsm8k_prompt_key = 'prompt'
            gsm8k_answer_key = 'extra_info.answer'
            math_prompt_key = 'prompt'
            math_answer_key = 'extra_info.answer'
            numina_prompt_key = 'problem'
            numina_answer_key = 'solution'
        
        # 🆕 如果提供了简化路径，自动构建dataset_configs
        if dataset_configs is None and (gsm8k_path or math_path or numina_path):
            dataset_configs = []
            if gsm8k_path:
                dataset_configs.append({
                    'name': 'gsm8k',
                    'files': [gsm8k_path],
                    'prompt_key': gsm8k_prompt_key,
                    'answer_key': gsm8k_answer_key
                })
            if math_path:
                dataset_configs.append({
                    'name': 'math',
                    'files': [math_path],
                    'prompt_key': math_prompt_key,
                    'answer_key': math_answer_key
                })
            if numina_path:
                dataset_configs.append({
                    'name': 'numina',
                    'files': [numina_path],
                    'prompt_key': numina_prompt_key,
                    'answer_key': numina_answer_key
                })
            print(f"📝 使用简化配置，自动构建了 {len(dataset_configs)} 个数据集")
            print(f"   模式: {'训练' if is_train else '验证'}")
        
        if not dataset_configs:
            raise ValueError(
                "必须提供 dataset_configs 或至少一个数据集路径（gsm8k_path/math_path/numina_path）。\n"
                "请在训练脚本中配置：\n"
                "  ++data.gsm8k_path=/path/to/gsm8k.parquet\n"
                "  ++data.math_path=/path/to/math.parquet"
            )
        
        self.dataset_configs = dataset_configs
        
        # Store all samples with their metadata
        self.samples = []
        self.dataset_names = []
        self.questions = []
        self.question_ids = []
        
        # Load data from all datasets
        self._load_all_datasets()
        
        print(f"Loaded {len(self.samples)} total samples from {len(dataset_configs)} datasets")
        
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
        
        for dataset_config in self.dataset_configs:
            dataset_name = dataset_config["name"]
            data_files = dataset_config["files"]
            prompt_key = dataset_config.get("prompt_key", "prompt")  # ⭐ 修改默认值为'prompt'
            answer_key = dataset_config.get("answer_key", "extra_info.answer")  # ⭐ 修改默认值
            
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
            padding=False,
            return_tensors=None,
        )
        
        # Prepare the output dict
        result = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'dataset_name': dataset_name,  # ⭐ Key field for COT matching
            'question': question_for_matching,  # ⭐ 用于COT匹配的纯文本问题
            'question_id': int(question_id),
        }
        
        # Add answer if available
        if sample['answer'] is not None:
            result['data_answer'] = str(sample['answer'])
        
        return result


def create_multi_dataset_from_config(config_path: str = None, **kwargs):
    """
    Helper function to create multi-dataset from a YAML config file.
    
    Config format:
        datasets:
          - name: gsm8k
            files:
              - /path/to/gsm8k/train.parquet
            prompt_key: question
            answer_key: answer
          
          - name: math
            files:
              - /path/to/math/train.parquet
            prompt_key: problem
            answer_key: solution
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

