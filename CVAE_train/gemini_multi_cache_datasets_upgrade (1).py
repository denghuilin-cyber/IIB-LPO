# run_lars_pipeline.py (Multi-GPU, Caching, Multi-Dataset Training & Multi-Dataset Selection)

import os
import json
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss

# Imports for Distributed Training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 为保证可复现性，设置随机种子
torch.manual_seed(42); np.random.seed(42); random.seed(42)

# DDP setup function
def setup_ddp():
    dist.init_process_group(backend="nccl"); rank = dist.get_rank(); world_size = dist.get_world_size(); local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank); return rank, world_size, local_rank

# =================================================================================
# 1. LaRS (Selector VAE) 模型定义 (无变化)
# =================================================================================
class LaRS_Selector_VAE(nn.Module):
    def __init__(self, embedding_dim, latent_dim=128, hidden_dim=256):
        super().__init__(); self.encoder = nn.Sequential(nn.Linear(embedding_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim * 2)); self.reasoning_policy = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim * 2)); self.decoder = nn.Sequential(nn.Linear(embedding_dim + latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim))
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar); eps = torch.randn_like(std); return mu + eps * std
    def forward(self, question_emb, rationale_emb):
        combined_qr = torch.cat([question_emb, rationale_emb], dim=1); posterior_params = self.encoder(combined_qr); posterior_mu, posterior_logvar = posterior_params.chunk(2, dim=-1); prior_params = self.reasoning_policy(question_emb); prior_mu, prior_logvar = prior_params.chunk(2, dim=-1); z = self.reparameterize(posterior_mu, posterior_logvar); combined_qz = torch.cat([question_emb, z], dim=1); reconstructed_rationale_emb = self.decoder(combined_qz); return (reconstructed_rationale_emb, posterior_mu, posterior_logvar, prior_mu, prior_logvar)

def vae_loss_function(reconstructed_emb, target_emb, p_mu, p_logvar, q_mu, q_logvar):
    recon_loss = nn.functional.mse_loss(reconstructed_emb, target_emb, reduction='sum'); kl_div = -0.5 * torch.sum(1 + q_logvar - p_logvar - ((q_mu - p_mu).pow(2) + q_logvar.exp()) / p_logvar.exp()); return recon_loss + kl_div

# =================================================================================
# 2. 数据集与预处理 (无变化)
# =================================================================================
class VAEDataset(Dataset):
    def __init__(self, question_embeddings, rationale_embeddings):
        self.question_embeddings = question_embeddings; self.rationale_embeddings = rationale_embeddings
    def __len__(self): return len(self.question_embeddings)
    def __getitem__(self, idx): return self.question_embeddings[idx], self.rationale_embeddings[idx], idx

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f: return [json.loads(line) for line in f]

# =================================================================================
# 3. 核心功能函数 (逻辑已适配多数据集)
# =================================================================================

def get_embeddings(texts, model, tokenizer, batch_size, desc, device, rank=0):
    all_embeddings = [];
    with torch.no_grad():
        progress_bar = tqdm(range(0, len(texts), batch_size), desc=desc, disable=(rank != 0))
        for i in progress_bar: batch_texts = texts[i:i+batch_size]; inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device); embeddings = model(**inputs).last_hidden_state.mean(dim=1); all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

def create_embeddings_parallel(texts, desc, dataset_name, args, tokenizer, model, device, rank, world_size):
    local_texts = texts[rank::world_size]
    if rank == 0: print(f"  > Rank {rank} 正在并行编码 {len(local_texts)} 个样本...")
    local_embeddings = get_embeddings(local_texts, model, tokenizer, args.embedding_batch_size, f"  > 并行编码 {desc}", device, rank)
    gathered_embeddings_list = [None] * world_size
    dist.all_gather_object(gathered_embeddings_list, local_embeddings)
    if rank == 0:
        # 重新构建完整的Tensor，确保顺序正确
        num_samples = len(texts)
        embedding_dim = local_embeddings.shape[1]
        full_embeddings_tensor = torch.zeros(num_samples, embedding_dim, dtype=local_embeddings.dtype)
        for i in range(world_size): full_embeddings_tensor[i::world_size] = gathered_embeddings_list[i]
        return full_embeddings_tensor
    return None

def get_or_create_embeddings_cached(texts, text_type, dataset_name, args, tokenizer, model, device, rank, world_size, is_distributed):
    model_name_slug = Path(args.embedding_model_path).name
    cache_dir = Path(args.data_base_path) / dataset_name / "encode_cache"
    cache_file = cache_dir / f"{dataset_name}_{model_name_slug}_{text_type}.pt"
    if not cache_file.exists():
        if rank == 0:
            print(f"未找到 {dataset_name} 的 {text_type} 缓存: {cache_file}"); cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"开始计算 {dataset_name} 的 {text_type} 嵌入向量...")
        if is_distributed:
            full_embeddings = create_embeddings_parallel(texts, f"{dataset_name} {text_type}", dataset_name, args, tokenizer, model, device, rank, world_size)
            if rank == 0: torch.save(full_embeddings.cpu(), cache_file); print(f"嵌入已计算并并行保存至缓存。")
        else: # 单进程情况
            embeddings_tensor = get_embeddings(texts, model, tokenizer, args.embedding_batch_size, f"编码 {text_type}", device, rank)
            torch.save(embeddings_tensor.cpu(), cache_file); print(f"嵌入已计算并保存至缓存。")
        if is_distributed: dist.barrier()
    if rank == 0: print(f"从缓存文件加载 {dataset_name} 的 {text_type} 嵌入: {cache_file}")
    embeddings = torch.load(cache_file, map_location='cpu'); return embeddings

def run_training(args, device, rank, world_size, dataset_names, combined_dataset_name_str):
    if rank == 0: print(f"--- 阶段1: 在 {len(dataset_names)} 个数据集上开始模型训练 ---")
    all_q_embeddings_list = []; all_r_embeddings_list = []
    if rank == 0: print(f"检查/创建嵌入向量缓存... (模型: {args.embedding_model_path})")
    embedding_model = AutoModel.from_pretrained(args.embedding_model_path).to(device); embedding_model.eval(); tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path, use_fast=True)
    for dataset_name in dataset_names:
        if rank == 0: print(f"\n--- 正在处理数据集: {dataset_name} ---")
        processed_data_path = Path(args.data_base_path) / dataset_name / "processed"; train_data = load_jsonl(processed_data_path / "train.jsonl"); questions = [d['question'] for d in train_data]; rationales = [d['rationale'] for d in train_data]
        q_embeddings = get_or_create_embeddings_cached(questions, "train_questions", dataset_name, args, tokenizer, embedding_model, device, rank, world_size, is_distributed=True)
        r_embeddings = get_or_create_embeddings_cached(rationales, "train_rationales", dataset_name, args, tokenizer, embedding_model, device, rank, world_size, is_distributed=True)
        all_q_embeddings_list.append(q_embeddings); all_r_embeddings_list.append(r_embeddings)
    if rank == 0: print("\n--- 所有数据集处理完毕，合并数据... ---")
    del embedding_model; torch.cuda.empty_cache()
    all_q_embeddings = torch.cat(all_q_embeddings_list, dim=0); all_r_embeddings = torch.cat(all_r_embeddings_list, dim=0); embedding_dim = all_q_embeddings.shape[1]
    if rank == 0: print(f"总计训练样本: {len(all_q_embeddings)}, 维度: {embedding_dim}")
    train_dataset = VAEDataset(all_q_embeddings, all_r_embeddings); train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True); train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, pin_memory=True)
    model = LaRS_Selector_VAE(embedding_dim=embedding_dim, latent_dim=args.latent_dim).to(device); model = DDP(model, device_ids=[device.index]); optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if rank == 0: print("LaRS VAE 模型已初始化并包裹 DDP，开始训练...")
    if rank == 0: print(f"LaRS VAE 模型初始化参数：embedding_dim:{embedding_dim},latent_dim:{args.latent_dim}")
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch); progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(rank != 0))
        for q_emb, r_emb, indices in progress_bar:
            q_emb, r_emb = q_emb.to(device), r_emb.to(device); optimizer.zero_grad(); recon_r_emb, p_mu, p_logvar, q_mu, q_logvar = model(q_emb, r_emb)
            loss = vae_loss_function(recon_r_emb, r_emb, p_mu, p_logvar, q_mu, q_logvar); loss.backward(); optimizer.step(); dist.all_reduce(loss, op=dist.ReduceOp.SUM); avg_loss = loss / world_size; progress_bar.set_postfix({'avg_loss_per_item': avg_loss.item() / q_emb.size(0)})
    dist.barrier(); model_save_path = None
    if rank == 0:
        output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True); model_save_path = output_dir / f"lars_selector_{combined_dataset_name_str}.pth"; torch.save(model.module.state_dict(), model_save_path); print(f"\n--- 训练完成！模型已保存至: {model_save_path} ---")
    return model_save_path

def run_skill_selection(args, device, model_path, target_dataset_name):
    print(f"\n--- 阶段2: 在目标数据集 {target_dataset_name} 上开始技能选择 ---")
    processed_data_path = Path(args.data_base_path) / target_dataset_name / "processed"; print(f"从 {processed_data_path} 加载数据...")
    train_data = load_jsonl(processed_data_path / "train.jsonl"); example_bank_data = load_jsonl(processed_data_path / "example_bank_1k.jsonl")
    print(f"检查/创建嵌入向量缓存... (模型: {args.embedding_model_path})")
    embedding_model = AutoModel.from_pretrained(args.embedding_model_path).to(device); embedding_model.eval(); tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path, use_fast=True)
    train_questions = [d['question'] for d in train_data]; bank_questions = [d['question'] for d in example_bank_data]
    train_q_embeddings = get_or_create_embeddings_cached(train_questions, "train_questions", target_dataset_name, args, tokenizer, embedding_model, device, rank=0, world_size=1, is_distributed=False)
    bank_q_embeddings = get_or_create_embeddings_cached(bank_questions, "bank_questions", target_dataset_name, args, tokenizer, embedding_model, device, rank=0, world_size=1, is_distributed=False)
    embedding_dim = train_q_embeddings.shape[1]; del embedding_model; torch.cuda.empty_cache()
    print(f"加载训练好的VAE模型: {model_path}")
    model = LaRS_Selector_VAE(embedding_dim=embedding_dim, latent_dim=args.latent_dim).to(device); model.load_state_dict(torch.load(model_path, map_location=device)); model.eval()
    print("计算技能向量..."); all_train_skill_vectors, all_bank_skill_vectors = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(train_q_embeddings), args.batch_size), desc="计算 train 技能向量"): batch_q_emb = train_q_embeddings[i:i+args.batch_size].to(device); prior_params = model.reasoning_policy(batch_q_emb); prior_mu, _ = prior_params.chunk(2, dim=-1); all_train_skill_vectors.append(prior_mu.cpu().numpy())
        for i in tqdm(range(0, len(bank_q_embeddings), args.batch_size), desc="计算 bank 技能向量"): batch_q_emb = bank_q_embeddings[i:i+args.batch_size].to(device); prior_params = model.reasoning_policy(batch_q_emb); prior_mu, _ = prior_params.chunk(2, dim=-1); all_bank_skill_vectors.append(prior_mu.cpu().numpy())
    all_train_skill_vectors = np.concatenate(all_train_skill_vectors, axis=0); all_bank_skill_vectors = np.concatenate(all_bank_skill_vectors, axis=0)
    print("使用 'example_bank' 的技能向量构建FAISS索引..."); index = faiss.IndexFlatL2(args.latent_dim); index.add(all_bank_skill_vectors); print("FAISS索引构建完成。")
    output_path = processed_data_path / f"train_k_shot_{target_dataset_name}.jsonl"; print(f"开始为 '{target_dataset_name}' 中的每个问题从 'example_bank' 检索 k={args.k} 个CoT...");
    with open(output_path, 'w', encoding='utf-8') as f_out:
        distances, indices = index.search(all_train_skill_vectors, args.k)
        for i in tqdm(range(len(train_data)), desc="检索并写入文件"):
            original_item = train_data[i]; selected_cots = []
            for idx in indices[i]: selected_item = example_bank_data[idx]; selected_cots.append({"id": selected_item["id"], "question": selected_item["question"], "rationale": selected_item["rationale"], "final_answer": selected_item["final_answer"]})
            output_item = original_item.copy(); output_item["selected_cots"] = selected_cots; f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
    print(f"\n--- 技能选择完成！结果已保存至: {output_path} ---")

# =================================================================================
# 4. Main Execution Block (<<< 已实现您要求的循环选择逻辑 >>>)
# =================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="复现 LaRS (Selector VAE) 的训练与技能选择流程")
    parser.add_argument("--dataset_name", type=str, required=True, help="要处理的数据集名称，可以是单个名称，或用逗号分隔的多个名称 (例如: GSM8K,MATH)")
    parser.add_argument("--data_base_path", type=str, required=True, help="存放所有数据集的总目录")
    parser.add_argument("--embedding_model_path", type=str, required=True, help="嵌入模型的本地文件夹路径")
    parser.add_argument("--output_dir", type=str, default="./lars_selector_checkpoints", help="保存训练好的VAE模型的目录")
    parser.add_argument("--latent_dim", type=int, default=1024, help="潜藏技能空间的维度")
    parser.add_argument("--batch_size", type=int, default=256, help="每个GPU的批次大小")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="学习率")
    parser.add_argument("--epochs", type=int, default=1000, help="训练轮次")
    parser.add_argument("--embedding_batch_size", type=int, default=32, help="计算嵌入时的批次大小")
    args = parser.parse_args()
    
    # 解析多数据集名称
    dataset_names = [name.strip() for name in args.dataset_name.split(',')]
    combined_dataset_name_str = "-".join(dataset_names)

    # DDP Main Execution Flow
    rank, world_size, local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    
    if rank == 0:
        print(f"使用 {world_size} 个 GPUs进行分布式训练。")
        print(f"训练数据集: {dataset_names}")

    # 1. 训练模型 (所有进程参与)
    trained_model_path = run_training(args, device, rank, world_size, dataset_names, combined_dataset_name_str)
    dist.barrier()
    
    # 2. 技能选择 (只在 rank 0 进程执行，并为每个数据集循环一次)
    if rank == 0:
        if trained_model_path:
            print("\n" + "="*20 + " 开始为所有目标数据集进行技能选择 " + "="*20)
            k_map = {"GSM8K": 8, "Spider": 8, "COGS": 8, "TabMWP": 8, "MATH":8, "NuminaMath-CoT":8}
            
            # <<< CRITICAL FIX: 循环为每个数据集执行技能选择 >>>
            for target_dataset_name in dataset_names:
                # 动态设置当前数据集的 k 值
                if target_dataset_name not in k_map:
                    print(f"警告: 在 k_map 中未找到数据集 '{target_dataset_name}' 的k值，将跳过此数据集的技能选择。")
                    continue
                
                # 创建一个新的args副本，以避免在循环中永久修改它
                current_args = argparse.Namespace(**vars(args))
                current_args.k = k_map[target_dataset_name]
                print(f"\n>>> 开始处理目标数据集: '{target_dataset_name}' (k={current_args.k})")
                
                # 为当前目标数据集调用技能选择函数
                run_skill_selection(current_args, device, trained_model_path, target_dataset_name)
        else:
            print("模型路径未找到，跳过技能选择阶段。")

    # 3. 清理进程组 (所有进程参与)
    dist.barrier()
    dist.destroy_process_group()
