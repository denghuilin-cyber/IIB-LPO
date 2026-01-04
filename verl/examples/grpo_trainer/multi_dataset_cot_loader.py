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
Multi-dataset COT loader for GRPO training.

Supports loading COT examples from multiple datasets (GSM8K, MATH, NuminaMath-CoT, etc.)
and automatically selecting the correct COT based on the data source.
"""

import json
from typing import Dict, List, Optional
import numpy as np


class MultiDatasetCOTLoader:
    """
    Manages COT examples for multiple datasets.
    
    Each dataset has its own COT file, and the loader automatically
    selects the correct COT based on the data source identifier.
    
    Example:
        loader = MultiDatasetCOTLoader(
            cot_file_mapping={
                "gsm8k": "/path/to/gsm8k_cot.jsonl",
                "math": "/path/to/math_cot.jsonl",
                "numina": "/path/to/numina_cot.jsonl",
            }
        )
    """
    
    def __init__(
        self,
        cot_file_mapping: Dict[str, str],
        cot_format_template: str = "Here's a similar example:\n\nQuestion: {question}\n\nStep-by-step solution:\n{rationale}\n\nFinal Answer: {final_answer}\n\nNow, let's solve the current problem:",
        match_by: str = "question",  # or "id"
        use_full_cot: bool = True,
    ):
        """
        Initialize the multi-dataset COT loader.
        
        Args:
            cot_file_mapping: Dict mapping dataset name to COT file path.
                Example: {"gsm8k": "/path/to/gsm8k_cot.jsonl", "math": "/path/to/math_cot.jsonl"}
            cot_format_template: Template for formatting COT examples.
            match_by: How to match training data with COT data ("question" or "id").
            use_full_cot: Whether to use the full COT or just rationale.
        """
        self.cot_file_mapping = cot_file_mapping
        self.cot_format_template = cot_format_template
        self.match_by = match_by
        self.use_full_cot = use_full_cot
        
        # Store COT data for each dataset: dataset_name -> {question/id -> [COT examples]}
        self.dataset_cot_data: Dict[str, Dict] = {}
        
        # Load COT data from all files
        self._load_all_cot_data()
        
        print(f"Loaded COT data for {len(self.dataset_cot_data)} datasets:")
        for dataset_name, cot_data in self.dataset_cot_data.items():
            print(f"  - {dataset_name}: {len(cot_data)} questions")
    
    def _load_all_cot_data(self):
        """Load COT data from all dataset files."""
        for dataset_name, cot_file_path in self.cot_file_mapping.items():
            print(f"Loading COT data for {dataset_name} from {cot_file_path}...")
            self.dataset_cot_data[dataset_name] = self._load_single_cot_file(cot_file_path)
    
    def _load_single_cot_file(self, cot_file_path: str) -> Dict:
        """Load COT data from a single JSONL file."""
        cot_data = {}
        
        with open(cot_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                # Get the key (question text or id)
                if self.match_by == "question":
                    key = data["question"].strip()
                elif self.match_by == "id":
                    key = data["id"]
                else:
                    raise ValueError(f"Unknown match_by: {self.match_by}")
                
                # Extract and format COT examples
                selected_cots = data.get("selected_cots", [])
                formatted_cots = []
                
                for cot in selected_cots:
                    if self.use_full_cot:
                        # Format the full COT example
                        formatted_cot = self.cot_format_template.format(
                            question=cot.get("question", ""),
                            rationale=cot.get("rationale", ""),
                            final_answer=cot.get("final_answer", "")
                        )
                    else:
                        # Only use the rationale
                        formatted_cot = cot.get("rationale", "")
                    
                    formatted_cots.append(formatted_cot)
                
                cot_data[key] = formatted_cots
        
        return cot_data
    
    def get_cot_examples(
        self,
        dataset_name: str,
        question: Optional[str] = None,
        question_id: Optional[int] = None,
        num_examples: Optional[int] = None
    ) -> List[str]:
        """
        Get COT examples for a given question from a specific dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., "gsm8k", "math").
            question: The question text (if match_by="question").
            question_id: The question ID (if match_by="id").
            num_examples: Number of examples to return. If None, return all.
        
        Returns:
            List of formatted COT example strings.
        """
        # Get COT data for this dataset
        if dataset_name not in self.dataset_cot_data:
            print(f"Warning: Unknown dataset name: {dataset_name}")
            print(f"Available datasets: {list(self.dataset_cot_data.keys())}")
            return []
        
        cot_data = self.dataset_cot_data[dataset_name]
        
        # Get the key
        if self.match_by == "question":
            if question is None:
                raise ValueError("question must be provided when match_by='question'")
            key = question.strip()
        elif self.match_by == "id":
            if question_id is None:
                raise ValueError("question_id must be provided when match_by='id'")
            key = question_id
        else:
            raise ValueError(f"Unknown match_by: {self.match_by}")
        
        # Get COT examples for this question
        cot_examples = cot_data.get(key, [])
        
        if not cot_examples:
            print(f"Warning: No COT examples found for {dataset_name}, key: {key}")
            return []
        
        # Return the requested number of examples
        if num_examples is None:
            return cot_examples
        elif num_examples <= len(cot_examples):
            return cot_examples[:num_examples]
        else:
            # If we need more examples than available, cycle through them
            result = []
            for i in range(num_examples):
                result.append(cot_examples[i % len(cot_examples)])
            return result


# Global multi-dataset COT loader instance
_global_multi_cot_loader: Optional[MultiDatasetCOTLoader] = None


def initialize_multi_dataset_cot_loader(
    cot_file_mapping: Dict[str, str],
    cot_format_template: str = "Here's a similar example:\n\nQuestion: {question}\n\nLet's solve it step by step:\n{rationale}\n\nFinal Answer: {final_answer}\n\nNow, let's solve the current problem:",
    match_by: str = "question",
    use_full_cot: bool = True,
):
    """
    Initialize the global multi-dataset COT loader.
    
    Args:
        cot_file_mapping: Dict mapping dataset name to COT file path.
            Example:
            {
                "gsm8k": "/nas/dhl/CVAE/Datasets/GSM8K/processed/train_k_shot_GSM8K.jsonl",
                "math": "/nas/dhl/CVAE/Datasets/MATH/processed/train_k_shot_MATH.jsonl",
                "numina": "/nas/dhl/CVAE/Datasets/NuminaMath-CoT/processed/train_k_shot_MATH.jsonl",
            }
    """
    global _global_multi_cot_loader
    _global_multi_cot_loader = MultiDatasetCOTLoader(
        cot_file_mapping=cot_file_mapping,
        cot_format_template=cot_format_template,
        match_by=match_by,
        use_full_cot=use_full_cot,
    )
    return _global_multi_cot_loader


def get_multi_dataset_cot_examples(batch, prompt_idx: int, num_repeats: int, tokenizer=None) -> List[str]:
    """
    COT example getter function for multi-dataset GRPO.
    
    This function will be called by the GRPOCOTAugmenter for each prompt.
    It automatically determines which dataset the question comes from and
    retrieves the appropriate COT examples.
    
    Args:
        batch: DataProto batch containing prompts
        prompt_idx: Index of the original prompt (before repetition)
        num_repeats: Number of times this prompt will be repeated
        tokenizer: Tokenizer (optional)
    
    Returns:
        List of COT example strings, one for each repetition.
    """
    global _global_multi_cot_loader
    
    if _global_multi_cot_loader is None:
        raise RuntimeError(
            "Multi-dataset COT loader not initialized. "
            "Call initialize_multi_dataset_cot_loader() first."
        )
    
    # Get dataset name (source) from batch
    # This is the key field that identifies which dataset the question comes from
    dataset_name = None
    if "dataset_name" in batch.non_tensor_batch:
        dataset_names = batch.non_tensor_batch["dataset_name"]
        if isinstance(dataset_names, np.ndarray):
            dataset_name = str(dataset_names[prompt_idx * num_repeats])
        else:
            dataset_name = dataset_names[prompt_idx * num_repeats]
    elif "source" in batch.non_tensor_batch:
        # Alternative field name
        sources = batch.non_tensor_batch["source"]
        if isinstance(sources, np.ndarray):
            dataset_name = str(sources[prompt_idx * num_repeats])
        else:
            dataset_name = sources[prompt_idx * num_repeats]
    
    if dataset_name is None:
        print("Error: No dataset_name or source field found in batch!")
        print(f"Available non_tensor_batch keys: {batch.non_tensor_batch.keys()}")
        return [""] * num_repeats
    
    # Normalize dataset name (handle case variations)
    dataset_name = dataset_name.lower().strip()
    
    # Get question or question_id
    question = None
    question_id = None
    
    if "question" in batch.non_tensor_batch:
        questions = batch.non_tensor_batch["question"]
        if isinstance(questions, np.ndarray):
            question = questions[prompt_idx * num_repeats]
        else:
            question = questions[prompt_idx * num_repeats]
    
    if "id" in batch.non_tensor_batch or "question_id" in batch.non_tensor_batch:
        id_field = "id" if "id" in batch.non_tensor_batch else "question_id"
        ids = batch.non_tensor_batch[id_field]
        if isinstance(ids, np.ndarray):
            question_id = int(ids[prompt_idx * num_repeats])
        else:
            question_id = ids[prompt_idx * num_repeats]
    
    # Decode from input_ids if needed
    if question is None and tokenizer is not None:
        try:
            input_ids = batch.batch["input_ids"][prompt_idx * num_repeats]
            attention_mask = batch.batch["attention_mask"][prompt_idx * num_repeats]
            prompt_length = attention_mask.sum().item()
            prompt_tokens = input_ids[:prompt_length]
            question = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"Warning: Failed to decode question: {e}")
    
    # Get COT examples from the appropriate dataset
    try:
        cot_examples = _global_multi_cot_loader.get_cot_examples(
            dataset_name=dataset_name,
            question=question,
            question_id=question_id,
            num_examples=num_repeats
        )
        
        if len(cot_examples) < num_repeats:
            print(f"Warning: Got {len(cot_examples)} COT examples but need {num_repeats}")
            while len(cot_examples) < num_repeats:
                if len(cot_examples) > 0:
                    cot_examples.append(cot_examples[len(cot_examples) % len(cot_examples)])
                else:
                    cot_examples.append("")
        
        return cot_examples[:num_repeats]
    
    except Exception as e:
        print(f"Error getting COT examples for dataset {dataset_name}: {e}")
        return [""] * num_repeats

