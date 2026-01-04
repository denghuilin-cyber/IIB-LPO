
# model name就是模型保存的地址 例如 "/nas/dhl/outputs/test_cot_entropy_output"
# steps 就是 需要测试模型的步数 比如 1000 或者 2000
# hf_path 就是原模型的文件夹 需要借用里面的配置文件

python /nas/dhl/verl/merge/merge.py \
    --model_name="/nas/dhl/outputs/train_math_dapo_20251120_170741" \
    --step=100 \
    --hf_path="/nas/models/Qwen3-8B"