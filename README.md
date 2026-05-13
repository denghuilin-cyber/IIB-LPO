# <img src="assets/iib-lpo-logo.png" alt="I²B-LPO logo" width="64" valign="middle"> I²B-LPO: Latent Policy Optimization via Iterative Information Bottleneck

Official code for **I²B-LPO: Latent Policy Optimization via Iterative Information Bottleneck**, an ACL 2026 paper project built on `verl`.

## News

- `[2026-03-25]` We released the original code without the complete README documentation.
- `[2026-04-03]` I²B-LPO was accepted to ACL Main 2026.
- `[2026-05-08]` We prepared the public GitHub homepage and documented the main code path, environment setup, vLLM patching, data preparation, CVAE selector, and RL training entry points.

## Overview

I²B-LPO extends the `verl` RL training stack with patched vLLM rollout, token-level entropy tracing, CVAE-guided branching, and multi-dataset CoT augmentation for math RL training.

The reproduction workflow has three main stages. The first stage is environment preparation, which includes both the `verl` environment and the patched vLLM runtime required by IIB-LPO.

1. **Environment setup**  
   Install the Python/CUDA dependencies, prepare the `verl`-based RL training environment, and patch vLLM so rollout can return token-level entropy and support IIB-LPO branching behavior.

2. **CVAE selector preparation**  
   Train or load the CVAE selector used to sample latent reasoning branches.

3. **RL training**  
   Run the modified `verl` training pipeline with entropy-aware rollout, CVAE-guided branching, and multi-dataset CoT augmentation.

Repository layout:

```text
IIB-LPO-main/
├── README.md                         # GitHub homepage
├── build_verl_fast.sh                # Environment setup helper
├── replace.sh / replace_vllm_v1.sh   # vLLM replacement scripts
├── CVAE_train/                       # CVAE selector training
├── Eval/                             # Evaluation and scoring scripts
├── verl/                             # Core: modified verl project
│   ├── verl/trainer/main_ppo.py      # RL entry point
│   ├── verl/trainer/ppo/ray_trainer.py
│   ├── verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
│   ├── verl/utils/cvae_branching.py
│   ├── forward_context.py
│   └── examples/grpo_trainer/
└── vllm_20251119_0849/               # Modified vLLM snapshot
```

Stage entry points:

- ⚙️ **[Setup] Base Environment**: [`build_verl_fast.sh`](build_verl_fast.sh)
- ⚙️ **[Setup] vLLM Patching**: [`replace.sh`](replace.sh), [`replace_vllm_v1.sh`](replace_vllm_v1.sh)
- 🚀 **[Train] CVAE Selector**: [`CVAE_train/train_multi_datasets.sh`](CVAE_train/train_multi_datasets.sh)
- 🚀 **[Train] RL Pipeline**: [`verl/examples/grpo_trainer/train_MATH_DAPO.sh`](verl/examples/grpo_trainer/train_MATH_DAPO.sh)

## Environment Setup

### 🛠️ 1. Python and verl

The project was developed around Python 3.10, PyTorch 2.6, vLLM 0.8.5.post1, Ray, and the modified local `verl` (`0.5.0.dev`) source in this repository.

Before running, edit the machine-specific paths in [`build_verl_fast.sh`](build_verl_fast.sh), especially `ENV_PATH` and local wheel paths.

Then create the environment and run the helper:

```bash
conda create -n IIB python=3.10 -y
conda activate IIB
bash build_verl_fast.sh
```

The helper installs the runtime dependencies and installs this repository's modified local `verl` in editable mode during `step_9`. Do not replace it with the clean upstream `verl`; otherwise, the training path will miss the project-specific modules. After installation, the package versions should be close to [`requirements_verl_fast.txt`](requirements_verl_fast.txt).

### 🛠️ 2. Patched vLLM Setup

The RL rollout path requires a **patched vLLM installation**. In practice, this means installing the base vLLM package first and then replacing specific files inside your local vLLM package directory with the modified files provided in this repository.

✨ **The patch enables:**

- `compute_entropy=True` in rollout sampling;
- Token-level entropy extraction into `rollout_entropies`;
- CVAE branching metadata and response-mask alignment;
- `z`-injection through the patched forward context.

---

**Step 1: Install base vLLM** 📦

Install the official base version first:

```bash
pip install vllm==0.8.5.post1
```

**Step 2: Find the installed vLLM directory** 🔍

Locate where `pip` installed `vllm` in your active environment:

```bash
python -c "import vllm, os; print(os.path.dirname(vllm.__file__))"
```

The output should look something like this:

```text
/path/to/conda/envs/IIB/lib/python3.10/site-packages/vllm
```

**Step 3: Configure the patch scripts** 📝

This output is your **target directory** that will be patched. Edit [`replace.sh`](replace.sh) or [`replace_vllm_v1.sh`](replace_vllm_v1.sh) before running, and update these variables:

- 🎯 `TARGET_ROOT` / `TARGET_VLLM`: the installed vLLM directory in your conda environment.
- 📂 `SOURCE_ROOT` / `SOURCE_VLLM`: the modified vLLM source directory from this repository.

**Step 4: Apply the patch** 🚀

Then replace the installed local vLLM files:

```bash
bash replace.sh
bash replace_vllm_v1.sh
```

---

📚 **Reference files for the patch:**

- [`verl/VERL_ENTROPY_INTEGRATION.md`](verl/VERL_ENTROPY_INTEGRATION.md)
- [`verl/VLLM_ENTROPY_MODIFICATIONS.md`](verl/VLLM_ENTROPY_MODIFICATIONS.md)

## 📦 Dataset Preparation

To reproduce the default IIB-LPO training, download the public raw datasets listed below for provenance, and use our processed files from Google Drive for exact runs. The required files are split into RL training files and CVAE selector files.

### 🌐 Public Raw Datasets

| Dataset        | Used for                         | Link                                                         |
| -------------- | -------------------------------- | ------------------------------------------------------------ |
| GSM8K          | CVAE selector training source    | [Hugging Face](https://huggingface.co/datasets/openai/gsm8k) |
| MATH-lighteval | MATH source for RL and CVAE data | [Hugging Face](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval) |
| DAPO-Math-17k  | DAPO source for RL data          | [Hugging Face](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) |

### 🗂️ RL Training Files

| File                                | Used for                                                     | Download                                                     |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `MATH-train-final-filtered.parquet` | MATH training data for [`train_MATH_DAPO.sh`](verl/examples/grpo_trainer/train_MATH_DAPO.sh) | [Google Drive](https://drive.google.com/file/d/10h-HWUYvqhLn0vL9hTXl1k6Eoo8MBUtp/view?usp=drive_link) |
| `dapo-final-filtered.parquet`       | DAPO training data for [`train_MATH_DAPO.sh`](verl/examples/grpo_trainer/train_MATH_DAPO.sh) | [Google Drive](https://drive.google.com/file/d/14MEanTN-JDIoOGuCnaesFbB9fPVAeYCB/view?usp=drive_link) |
| `train_k_shot_MATH.jsonl`           | Retrieved CoT examples for MATH prompt augmentation          | [Google Drive](https://drive.google.com/file/d/1uh96LWVkvLxQ-Fm7v_pC6TDGpw5EC2o7/view?usp=sharing) |
| `lars_selector_GSM8K-MATH.pth`      | Trained CVAE selector weights for CVAE-guided branching during RL rollout | [Google Drive](https://drive.google.com/file/d/14Hpu89IFsV3lvVtxT46epMtpiK-w3A1R/view?usp=drive_link) |

### 🗂️ CVAE Training Files

| File                                    | Used for                                      | Download                                                     |
| --------------------------------------- | --------------------------------------------- | ------------------------------------------------------------ |
| `GSM8K/processed/train.jsonl`           | CVAE selector training data                   | [Hugging Face](https://huggingface.co/datasets/openai/gsm8k/viewer/main/train) |
| `GSM8K/processed/example_bank_1k.jsonl` | GSM8K CoT example bank                        | [Google Drive](https://drive.google.com/file/d/1kP_TQruhdXAgx6lEDm3KLfWadVeq_KEh/view?usp=sharing) |
| `MATH/processed/train.jsonl`            | CVAE selector training data                   | [Hugging Face](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval/viewer/default/train) |
| `MATH/processed/example_bank_1k.jsonl`  | MATH CoT example bank                         | [Google Drive](https://drive.google.com/file/d/1ZLhPKvwPlNIC0xRuBhzTDCWmtWNC0-_S/view?usp=sharing) |
| `microsoft/deberta-v2-xlarge`           | Embedding model for CVAE training and rollout | [Hugging Face](https://huggingface.co/microsoft/deberta-v2-xlarge) |

Default path mapping:

```bash
TRAIN_FILES_MATH="/path/to/MATH-train-final-filtered.parquet"
TRAIN_FILES_DAPO="/path/to/dapo-final-filtered.parquet"
MATH_COT="/path/to/train_k_shot_MATH.jsonl"
CVAE_MODEL_PATH="/path/to/lars_selector_GSM8K-MATH.pth"
CVAE_EMBEDDING_MODEL_PATH="/path/to/deberta-v2-xlarge"
```

## 🧬 CVAE Selector

Use [`CVAE_train/train_multi_datasets.sh`](CVAE_train/train_multi_datasets.sh) to train the CVAE selector. The repository also contains the current training implementation in [`CVAE_train/gemini_multi_cache_datasets_upgrade.py`](CVAE_train/gemini_multi_cache_datasets_upgrade.py); update the absolute Python path in the shell script before running it on a new machine.

```bash
bash CVAE_train/train_multi_datasets.sh
```

The script currently launches a Python training file under `/nas/dhl/CVAE/...` with arguments such as:

```bash
torchrun --nproc_per_node=1 --master_port=13254 /nas/dhl/CVAE/gemini_multi_cache_datasets_upgrade.py \
  --dataset_name GSM8K,MATH \
  --data_base_path /nas/dhl/CVAE/Datasets \
  --embedding_model_path /nas/dhl/CVAE/models/deberta-v2-xlarge \
  --output_dir /nas/dhl/CVAE/models/GSM8K-MATH-trained-1024 \
  --latent_dim 1024 \
  --batch_size 256 \
  --epochs 1000
```

Before running, update:

- `CUDA_VISIBLE_DEVICES`
- `--dataset_name`
- `--data_base_path`
- `--embedding_model_path`
- `--output_dir`
- `--latent_dim`

The CVAE implementation used by rollout is in [`verl/verl/utils/cvae_branching.py`](verl/verl/utils/cvae_branching.py):

- `LaRS_Selector_VAE`: latent selector model
- `CVAEBranchingManager`: loads CVAE and embedding model, samples z, and prepares z injection
- `create_cvae_manager`: factory used by vLLM rollout initialization

## 🚀 RL Training

Main script:

```bash
bash verl/examples/grpo_trainer/train_MATH_DAPO.sh
```

This script calls:

```bash
python3 -m verl.trainer.main_ppo
```

Key paths to edit inside [`train_MATH_DAPO.sh`](verl/examples/grpo_trainer/train_MATH_DAPO.sh):

```bash
ACTOR_MODEL_PATH="/path/to/base/model"
TRAIN_FILES_MATH="/path/to/MATH/train.parquet"
TRAIN_FILES_DAPO="/path/to/dapo-math-17k.parquet"
MATH_COT="/path/to/train_k_shot_MATH.jsonl"
CVAE_MODEL_PATH="/path/to/lars_selector.pth"
CVAE_EMBEDDING_MODEL_PATH="/path/to/deberta-v2-xlarge"
OUTPUT_DIR="/path/to/output"
```

Key IIB-LPO options:

```bash
actor_rollout_ref.rollout.compute_entropy=True
+actor_rollout_ref.rollout.enable_cvae_branching=True
+actor_rollout_ref.rollout.cvae_num_branches_per_path=3
+actor_rollout_ref.rollout.cvae_branching_mode=psa
+actor_rollout_ref.rollout.cvae_model_path="${CVAE_MODEL_PATH}"
+actor_rollout_ref.rollout.cvae_embedding_model_path="${CVAE_EMBEDDING_MODEL_PATH}"
+actor_rollout_ref.rollout.entropy_output.enabled=true
+actor_rollout_ref.rollout.entropy_output.token_entropy_to_jsonl=true
```

## 📊 Output and Analysis

Training output is controlled by `OUTPUT_DIR` in [`train_MATH_DAPO.sh`](verl/examples/grpo_trainer/train_MATH_DAPO.sh). The script creates:

- `training.log`
- offline `wandb/` logs
- rollout debug JSONL under `rollout_debug/`
- token entropy outputs under `Entropy_out/`
- checkpoints under the `verl` trainer output directory

Rollout debug files and token entropy samples are generated during training for later analysis.

## 🎯 Evaluation

We provide evaluation scripts to test the trained model's performance on standard mathematical benchmarks. The evaluation is a two-step process:

**Step 1: Rollout and Answer Extraction**  
Run the evaluation script to generate responses from the model.

```bash
sh Eval/math-eval/my_eval.sh
```

*Note:* Our parser strictly relies on the `\box{}` format to extract final answers. Ensure the model wraps its final mathematical answer in `\box{...}` so the script can accurately parse the response and save it to a `.jsonl` file.

**Step 2: Calculate Metrics**  
Run the scoring script on the generated `.jsonl` file to calculate the following metrics:

- **Unbiased Pass@k/n**
- **Average@k**
- **Standard Pass@k**
- **MajorityCorrect@n (c >= n/2)**
- **Maj@n (Vote)**

```bash
python Eval/math-eval/calculate_scores.py
```

*(Make sure to update the input file path inside `calculate_scores.py` to point to the output from Step 1, and set `n=1`, `Ks=[1]` if you only want to compute Pass@1.)*

## Code Structure

### Data Flow

```text
Processed math data + CoT files + CVAE checkpoint
        ↓
train_MATH_DAPO.sh
        ↓
verl.trainer.main_ppo
        ↓
MultiDatasetWithCOT  →  CoT augmentation
        ↓
vLLM rollout  ←  CVAE selector / z injection
        ↓
reward + entropy signals
        ↓
GRPO actor update
        ↓
checkpoints + logs
```

### Core Modules

- [`verl/verl/trainer/main_ppo.py`](verl/verl/trainer/main_ppo.py): Hydra/Ray entry point; creates datasets, reward functions, worker mappings, and trainer.
- [`verl/verl/trainer/ppo/ray_trainer.py`](verl/verl/trainer/ppo/ray_trainer.py): main RL training loop.
- [`verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`](verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py): vLLM rollout, entropy extraction, CVAE branching, response-mask construction.
- [`verl/verl/workers/config/rollout.py`](verl/verl/workers/config/rollout.py): rollout config, including `compute_entropy`, CVAE branching, CoT augmentation, and entropy output.
- [`verl/verl/utils/cvae_branching.py`](verl/verl/utils/cvae_branching.py): CVAE selector and z-injection manager.
- [`verl/forward_context.py`](verl/forward_context.py): patched forward context and `ZInjectionConfig`.
- [`verl/examples/grpo_trainer/multi_dataset_with_cot.py`](verl/examples/grpo_trainer/multi_dataset_with_cot.py): multi-dataset loader with `dataset_name`, `pure_question`, and reward-compatible `data_source`.

## Citation

```bibtex
@misc{deng2026IIB_lpo,
      title={IIB-LPO: Latent Policy Optimization via Iterative Information Bottleneck}, 
      author={Huilin Deng and Hongchen Luo and Yue Zhu and Long Li and Zhuoyue Chen and Xinghao Zhao and Ming Li and Jihai Zhang and Mengchang Wang and Yang Cao and Yu Kang},
      year={2026},
      eprint={2601.05870},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.05870}, 
}
```

## Acknowledgements

This repository builds on `verl`, vLLM, PyTorch, Hugging Face Transformers, and Ray.

**For questions, please contact huilin_deng@mail.ustc.edu.cn.**

## License

TBD. No `LICENSE` file is present in this repository yet.
