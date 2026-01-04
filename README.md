# IIB-LPO


**Environment Setup:**

1.Please use build_verl_fast.sh to quickly set up the reinforcement learning training environment for verl.

2.Use replace.sh followed by replace_vllm_va.sh to replace the modified files in vllm, including token-level entropy computation and the PSA operation.

**CVAE Training:**

train_multi_datasets.sh

**Verl Training:**

1.train_MATH_DAPO.sh for 8-GPU training with qwen2.5-7B.

2.train_MATH_DAPO_large.sh for 16-GPU training with qwen3-14B.



