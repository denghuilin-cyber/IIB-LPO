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
Example custom COT example getter for GRPO.

This shows how to dynamically generate COT examples based on the prompt content.
You can use this approach to:
1. Select examples based on problem type
2. Generate examples based on difficulty
3. Use different templates for different prompts
"""

import random
from typing import List
from verl import DataProto


def get_cot_examples_for_math_problems(batch: DataProto, prompt_idx: int, num_repeats: int) -> List[str]:
    """
    Generate COT examples dynamically based on the prompt.
    
    This example shows how to:
    1. Access the prompt content from the batch
    2. Determine what type of problem it is
    3. Return appropriate COT examples
    
    Args:
        batch: The DataProto batch containing prompts
        prompt_idx: Index of the original prompt (before repetition)
        num_repeats: Number of times this prompt will be repeated
    
    Returns:
        List of COT example strings, one for each repetition
    """
    
    # Basic COT examples for different problem types
    arithmetic_examples = [
        "Let's solve this arithmetic problem step by step.",
        "I'll work through this calculation carefully, one operation at a time.",
        "Let me break down this arithmetic problem into simpler steps.",
        "First, I'll identify the numbers and operations involved.",
    ]
    
    word_problem_examples = [
        "Let's understand what this word problem is asking: First, identify the key information.",
        "I'll translate this word problem into mathematical expressions step by step.",
        "Let me break down this problem: What do we know? What do we need to find?",
        "To solve this word problem, I'll: 1) Understand the scenario 2) Set up equations 3) Solve.",
    ]
    
    algebra_examples = [
        "Let's solve this algebraic equation systematically.",
        "I'll isolate the variable step by step, applying algebraic rules carefully.",
        "For this algebra problem, let me: 1) Simplify 2) Isolate the variable 3) Solve.",
        "Let's work through this algebraically, showing each transformation clearly.",
    ]
    
    general_examples = [
        "Let's think step by step about how to approach this problem.",
        "I'll work through this problem methodically, checking each step.",
        "Let me break this down into manageable parts and solve each one.",
        "First, let me understand what we're trying to find, then plan my approach.",
    ]
    
    # You can access the prompt text from the batch if needed
    # For example, to determine problem type based on keywords
    try:
        # Get the input_ids for this prompt
        input_ids = batch.batch["input_ids"][prompt_idx * num_repeats]
        # Note: You would need the tokenizer to decode this
        # For now, we'll use a simple heuristic or random selection
        
        # In a real implementation, you might:
        # 1. Decode the prompt text
        # 2. Analyze it to determine problem type
        # 3. Select appropriate examples
        
        # For this example, we'll randomly choose a category
        categories = [arithmetic_examples, word_problem_examples, algebra_examples, general_examples]
        selected_category = random.choice(categories)
        
    except Exception:
        # Fallback to general examples if we can't access the prompt
        selected_category = general_examples
    
    # Sample num_repeats examples from the selected category
    if len(selected_category) >= num_repeats:
        return random.sample(selected_category, num_repeats)
    else:
        # If we don't have enough examples, sample with replacement
        return random.choices(selected_category, k=num_repeats)


def get_cot_examples_with_difficulty_scaling(batch: DataProto, prompt_idx: int, num_repeats: int) -> List[str]:
    """
    Another example: Scale COT example complexity based on training progress.
    
    You could access global_steps from batch.meta_info to adjust difficulty.
    """
    
    # Early training: More detailed guidance
    detailed_examples = [
        "Let's solve this very carefully with detailed steps: Step 1...",
        "I'll break this down into very small, manageable pieces...",
        "Let me explain my thinking in detail as I solve this...",
    ]
    
    # Later training: More concise guidance
    concise_examples = [
        "Let's solve this step by step.",
        "I'll work through this systematically.",
        "Let me approach this problem logically.",
    ]
    
    # You could use batch.meta_info["global_steps"] to decide which set to use
    # For now, we'll return a mix
    all_examples = detailed_examples + concise_examples
    
    if len(all_examples) >= num_repeats:
        return random.sample(all_examples, num_repeats)
    else:
        return random.choices(all_examples, k=num_repeats)


def get_cot_examples_from_few_shot_pool(batch: DataProto, prompt_idx: int, num_repeats: int) -> List[str]:
    """
    Example: Use a pool of few-shot examples from a dataset.
    
    You could load actual solved examples from your training data
    and use them as COT demonstrations.
    """
    
    # In a real implementation, you might:
    # 1. Load a pool of solved examples from a file
    # 2. For each repetition, sample a different few-shot example
    # 3. Format it as a demonstration
    
    few_shot_pool = [
        "Example: Q: What is 2+2? A: Let me add: 2+2 = 4. Now let's solve the current problem:",
        "Example: Q: Solve x+3=7. A: Subtracting 3 from both sides: x=4. Now for this problem:",
        "Example: Q: Find 10% of 50. A: 10% = 0.1, so 0.1*50 = 5. Now let's solve:",
        "Example: Q: What's 5*6? A: Counting by 5s six times: 5,10,15,20,25,30. Now:",
    ]
    
    # Return different examples for each repetition
    if len(few_shot_pool) >= num_repeats:
        return random.sample(few_shot_pool, num_repeats)
    else:
        return random.choices(few_shot_pool, k=num_repeats)

