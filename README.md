# IIB-LPO: Latent Policy Optimization via Iterative Information Bottleneck

Official code for IIB-LPO: Latent Policy Optimization via Iterative Information Bottleneck, an ACL 2026 paper project built on verl.

## Overview
IIB-LPO extends the verl RL training stack with patched vLLM rollout, token-level entropy tracing, CVAE-guided branching, and multi-dataset CoT augmentation for math RL training.

## Environment Setup
1. Python and verl: Install dependencies and the modified local verl source.
2. 2. Patched vLLM: Install vLLM 0.8.5.post1 and apply patches for entropy extraction.
  
   3. ## Dataset Preparation
   4. Download public datasets (GSM8K, MATH, DAPO) and processed files for reproduction.
  
   5. ## RL Training
   6. Run the modified verl training pipeline:
   7. ```bash
      bash verl/examples/grpo_trainer/train_MATH_DAPO.sh
      ```

      ## Evaluation
      1. Rollout and Answer Extraction: `sh Eval/math-eval/my_eval.sh`
      2. 2. Calculate Metrics: `python Eval/math-eval/calculate_scores.py`
         3. 
