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
GSM8K COT loader for GRPO training.

Loads COT examples from a separate JSONL file and matches them with training data.
"""

import json
from typing import Dict, List, Optional
import numpy as np


class GSM8KCOTLoader:
    """
    Loads and manages COT examples for GSM8K training.
    
    Matches training questions with their corresponding COT examples
    from a separate JSONL file.
    """
    
    def __init__(
        self,
        cot_file_path: str,
        cot_format_template: str = "Question: {question}\nAnswer: {rationale}\nFinal Answer: {final_answer}",
        match_by: str = "question",  # or "id"
        use_full_cot: bool = True,  # If False, only use rationale
    ):
        """
        Initialize the COT loader.
        
        Args:
            cot_file_path: Path to the JSONL file containing COT examples.
            cot_format_template: Template for formatting COT examples.
                                Use {question}, {rationale}, {final_answer} as placeholders.
            match_by: How to match training data with COT data ("question" or "id").
            use_full_cot: Whether to use the full COT (question+rationale+answer) or just rationale.
        """
        self.cot_file_path = cot_file_path
        self.cot_format_template = cot_format_template
        self.match_by = match_by
        self.use_full_cot = use_full_cot
        
        # Load COT data: question/id -> list of COT examples
        self.cot_data: Dict = {}
        self._load_cot_data()
        
        print(f"Loaded COT data for {len(self.cot_data)} questions from {cot_file_path}")
    
    def _load_cot_data(self):
        """Load COT data from JSONL file."""
        with open(self.cot_file_path, 'r', encoding='utf-8') as f:
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
                        # Only use the rationale (reasoning steps)
                        formatted_cot = cot.get("rationale", "")
                    
                    formatted_cots.append(formatted_cot)
                
                self.cot_data[key] = formatted_cots
    
    def get_cot_examples(
        self,
        question: Optional[str] = None,
        question_id: Optional[int] = None,
        num_examples: Optional[int] = None
    ) -> List[str]:
        """
        Get COT examples for a given question.
        
        Args:
            question: The question text (if match_by="question").
            question_id: The question ID (if match_by="id").
            num_examples: Number of examples to return. If None, return all.
                         If more than available, will cycle through available examples.
        
        Returns:
            List of formatted COT example strings.
        """
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
        cot_examples = self.cot_data.get(key, [])
        
        if not cot_examples:
            print(f"Warning: No COT examples found for key: {key}")
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


# Global COT loader instance (will be initialized in the getter function)
_global_cot_loader: Optional[GSM8KCOTLoader] = None


def initialize_gsm8k_cot_loader(
    cot_file_path: str,
    cot_format_template: str = "Here's a similar example:\n\nQuestion: {question}\n\nLet's solve it step by step:\n{rationale}\n\nFinal Answer: {final_answer}\n\nNow, let's solve the current problem:",
    match_by: str = "question",
    use_full_cot: bool = True,
):
    """
    Initialize the global COT loader.
    
    Call this function once during trainer initialization.
    """
    global _global_cot_loader
    _global_cot_loader = GSM8KCOTLoader(
        cot_file_path=cot_file_path,
        cot_format_template=cot_format_template,
        match_by=match_by,
        use_full_cot=use_full_cot,
    )
    return _global_cot_loader


def get_gsm8k_cot_examples(batch, prompt_idx: int, num_repeats: int, tokenizer=None) -> List[str]:
    """
    COT example getter function for GRPO.
    
    This function will be called by the GRPOCOTAugmenter for each prompt.
    
    Args:
        batch: DataProto batch containing prompts
        prompt_idx: Index of the original prompt (before repetition)
        num_repeats: Number of times this prompt will be repeated (GRPO rollout count)
        tokenizer: Tokenizer (optional, for decoding if needed)
    
    Returns:
        List of COT example strings, one for each repetition.
    """
    global _global_cot_loader
    
    if _global_cot_loader is None:
        raise RuntimeError(
            "COT loader not initialized. Call initialize_gsm8k_cot_loader() first."
        )
    
    # Try to get the question from non_tensor_batch
    # The exact field name depends on your dataset format
    question = None
    question_id = None
    
    # Method 1: Try to get from non_tensor_batch (if dataset provides it)
    if "question" in batch.non_tensor_batch:
        questions = batch.non_tensor_batch["question"]
        # Since batch is already repeated, we need to map back to original index
        # prompt_idx is the original index before repeat
        if isinstance(questions, np.ndarray):
            question = questions[prompt_idx * num_repeats]
        else:
            question = questions[prompt_idx * num_repeats]
    
    # Method 2: Try to get ID if available
    if "id" in batch.non_tensor_batch or "question_id" in batch.non_tensor_batch:
        id_field = "id" if "id" in batch.non_tensor_batch else "question_id"
        ids = batch.non_tensor_batch[id_field]
        if isinstance(ids, np.ndarray):
            question_id = int(ids[prompt_idx * num_repeats])
        else:
            question_id = ids[prompt_idx * num_repeats]
    
    # Method 3: Decode from input_ids if tokenizer is available
    if question is None and tokenizer is not None:
        try:
            input_ids = batch.batch["input_ids"][prompt_idx * num_repeats]
            attention_mask = batch.batch["attention_mask"][prompt_idx * num_repeats]
            prompt_length = attention_mask.sum().item()
            prompt_tokens = input_ids[:prompt_length]
            question = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"Warning: Failed to decode question from input_ids: {e}")
    
    # Get COT examples
    try:
        cot_examples = _global_cot_loader.get_cot_examples(
            question=question,
            question_id=question_id,
            num_examples=num_repeats
        )
        
        # If we got fewer examples than needed, pad with empty strings or repeat
        if len(cot_examples) < num_repeats:
            print(f"Warning: Got {len(cot_examples)} COT examples but need {num_repeats}")
            # Cycle through available examples
            while len(cot_examples) < num_repeats:
                cot_examples.append(cot_examples[len(cot_examples) % max(1, len(cot_examples))])
        
        return cot_examples[:num_repeats]
    
    except Exception as e:
        print(f"Error getting COT examples: {e}")
        # Return empty strings as fallback
        return [""] * num_repeats

