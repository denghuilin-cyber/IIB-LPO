# model_name: The directory path where the trained model is saved, e.g., "/nas/dhl/outputs/test_cot_entropy_output"
# steps: The training step checkpoint you want to test, e.g., 1000 or 2000
# hf_path: The directory path of the original base model, required to borrow its configuration files

python /nas/dhl/verl/merge/merge.py \
    --model_name="/nas/dhl/outputs/test_cot_entropy_output" \
    --step=1000 \
    --hf_path="/nas/models/Qwen3-8B"
