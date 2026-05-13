# #!/bin/bash
# export HF_ENDPOINT=https://hf-mirror.com
# MODEL_PATH="/nas/models/Qwen3-8B"
# OUTPUT_DIR=“/nas/dhl/Eval/Eval_Output/qwen3-8192”
# mkdir -p "$OUTPUT_DIR"
# echo "Evaluating $MODEL_PATH"
# python eval.py \
#   --model_name="$MODEL_PATH" \
#   --datasets="HuggingFaceH4/MATH-500" \
#   --split="test" \
#   --output_dir="$OUTPUT_DIR" \
#   --batch_size=1000 \
#   --max_tokens=8192 \
#   --num_gpus=4 \
#   --temperature=0 \
#   --top_p=0.95 \
#   --num_generation=1


#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
#MODEL_PATH="/nas/models/Qwen2.5-Math-7B"
MODEL_PATH="/nas/models/Qwen3-4B-Instruct-2507"
OUTPUT_DIR=“/nas/dhl/Eval/Eval_Output/Qwen3-4B-Instruct-8192” 
mkdir -p "$OUTPUT_DIR"
echo "Evaluating $MODEL_PATH"
python /nas/dhl/Eval/math-eval/eval.py \
  --model_name="$MODEL_PATH" \
  --datasets="HuggingFaceH4/MATH-500" \
  --split="test" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size=1000 \
  --max_tokens=8192 \
  --num_gpus=4 \
  --temperature=0 \
  --top_p=0.95 \
  --num_generation=1

