# 运行修正后的测试代码
python -c "
from transformers import AutoTokenizer, AutoModel

model_path = '/nas/dhl/CVAE/models/deberta-v2-xlarge'
print('测试加载 (已修正)...')

# ★★★ 这里是关键的修改 ★★★
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModel.from_pretrained(model_path)

print('✅ 加载成功！')
print(f'Tokenizer: {type(tokenizer).__name__}') # 应该会输出 DebertaV2Tokenizer

# 测试编码
text = 'What is 2+2?'
tokens = tokenizer(text, return_tensors='pt')
print(f'✅ 编码测试通过！')
"


# gemini
# python /nas/dhl/CVAE/gemini_Train_Selected.py \
#     --dataset_name MATH \
#     --data_base_path /nas/dhl/CVAE/Datasets \
#     --embedding_model_path /nas/dhl/CVAE/models/deberta-v2-xlarge \
#     --output_dir /nas/dhl/CVAE/models/MATH_trained


#/nas/dhl/CVAE/gemini_multi_cache_datasets_upgrade.py 加了多卡


# torchrun --nproc_per_node=8 /nas/dhl/CVAE/gemini_multi_cache_datasets_upgrade.py \
#     --dataset_name GSM8K,MATH,NuminaMath-CoT \
#     --data_base_path /nas/dhl/CVAE/Datasets \
#     --embedding_model_path /nas/dhl/CVAE/models/deberta-v2-xlarge \
#     --output_dir /nas/dhl/CVAE/models/GSM8K-MATH-NuminaMath-trained \
#     --batch_size 256 \
#     --epochs 1000

export CUDA_VISIBLE_DEVICES="4"

torchrun --nproc_per_node=1 --master_port=13254 /nas/dhl/CVAE/gemini_multi_cache_datasets_upgrade.py \
    --dataset_name GSM8K,MATH \
    --data_base_path /nas/dhl/CVAE/Datasets \
    --embedding_model_path /nas/dhl/CVAE/models/deberta-v2-xlarge \
    --output_dir /nas/dhl/CVAE/models/GSM8K-MATH-trained-1024 \
    --latent_dim 1024 \
    --batch_size 256 \
    --epochs 1000

# NuminaMath-CoT

#cache的逻辑

#原始数据文件	处理的内容	代码中使用的text_type	最终生成的缓存文件名 (.../encode_cache/目录下)
#train.jsonl	question	"train_questions"	MATH_deberta-v2-xlarge_train_questions.pt
#train.jsonl	rationale	"train_rationales"	MATH_deberta-v2-xlarge_train_rationales.pt
#example_bank_1k.jsonl	question	"bank_questions"	MATH_deberta-v2-xlarge_bank_questions.pt


