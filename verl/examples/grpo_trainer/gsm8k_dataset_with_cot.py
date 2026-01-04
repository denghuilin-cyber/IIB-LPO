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
Custom dataset for GSM8K with COT support.

This dataset ensures that question text and ID are included in the batch,
which is needed for matching with COT examples during rollout.
"""

import pandas as pd
from torch.utils.data import Dataset
from verl.utils.dataset.rl_dataset import RLHFDataset


class GSM8KDatasetWithCOT(RLHFDataset):
    """
    Extended GSM8K dataset that includes question metadata for COT matching.
    
    This dataset inherits from RLHFDataset and adds question text/ID to
    the non_tensor_batch so that COT examples can be matched during rollout.
    """
    
    def __init__(self, data_files, tokenizer, processor, config):
        """
        Initialize the dataset.
        
        Args:
            data_files: List of paths to data files (parquet, jsonl, etc.)
            tokenizer: Tokenizer for encoding text
            processor: Processor for multimodal data (can be None)
            config: Data configuration
        """
        super().__init__(data_files, tokenizer, processor, config)
        
        # Store question metadata for COT matching
        self.questions = []
        self.question_ids = []
        
        self._extract_question_metadata()
    
    def _extract_question_metadata(self):
        """Extract question text and IDs from the dataset."""
        # The parent class has loaded data into self.data_list
        # We need to extract questions from it
        
        for idx, item in enumerate(self.data_list):
            # Try to get question from the data item
            # The exact field name depends on your dataset format
            
            if isinstance(item, dict):
                # If data is a dict
                question = item.get('question', item.get('prompt', ''))
                question_id = item.get('id', idx)
            else:
                # If data is a different format, adjust accordingly
                question = str(item) if item else ''
                question_id = idx
            
            self.questions.append(question)
            self.question_ids.append(question_id)
        
        print(f"Extracted metadata for {len(self.questions)} questions")
    
    def __getitem__(self, index):
        """
        Get a data sample with question metadata.
        
        Returns:
            dict: Data sample with tensors and non-tensor metadata
        """
        # Get the base item from parent class
        item = super().__getitem__(index)
        
        # Add question metadata to the item
        # This will be included in non_tensor_batch during collation
        item['question'] = self.questions[index]
        item['question_id'] = self.question_ids[index]
        
        return item


class GSM8KParquetDatasetWithCOT(Dataset):
    """
    Simple dataset for loading GSM8K from parquet with COT support.
    
    Use this if you want a simpler implementation that directly reads from parquet.
    """
    
    def __init__(self, data_files, tokenizer, processor, config):
        """
        Initialize the dataset.
        
        Args:
            data_files: List of paths to parquet files
            tokenizer: Tokenizer for encoding text
            processor: Processor (unused for GSM8K)
            config: Data configuration
        """
        self.tokenizer = tokenizer
        self.config = config
        
        # Load data from parquet
        if isinstance(data_files, str):
            data_files = [data_files]
        
        dfs = []
        for file_path in data_files:
            df = pd.read_parquet(file_path)
            dfs.append(df)
        
        self.data = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.data)} samples from {len(data_files)} parquet file(s)")
        
        # Configuration
        self.prompt_key = config.get('prompt_key', 'question')
        self.max_prompt_length = config.get('max_prompt_length', 512)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """Get a data sample."""
        row = self.data.iloc[index]
        
        # Get the question/prompt
        question = row[self.prompt_key]
        
        # Apply chat template if needed
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
        encoded = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_prompt_length,
            padding=False,
            return_tensors=None,
        )
        
        # Prepare the output dict
        result = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'question': question,  # Include question text for COT matching
            'question_id': int(row.get('id', index)),  # Include ID if available
        }
        
        # Add ground truth answer if available
        if 'answer' in row:
            result['data_answer'] = str(row['answer'])
        
        return result

