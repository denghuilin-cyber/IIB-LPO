"""
CVAE Branching Module for I²B-LPO
用于思维分叉的 CVAE 加载、采样和注入功能
"""

import os
import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
# 🔧 关键修复：设置日志级别，确保 INFO 日志能够显示
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))  # 默认 INFO，而不是 WARN


# =================================================================================
# SOFTMAX Fusion: vLLM LogitsProcessor 实现
# =================================================================================
class ZInjectionLogitsProcessor:
    """
    SOFTMAX Fusion 的 LogitsProcessor 实现
    在 logits 上注入 CVAE 采样的 z 向量的影响
    
    这个类会在 vLLM 的采样过程中被调用，在 softmax 之前修改 logits
    """
    
    def __init__(self, z_proj_vocab: torch.Tensor):
        """
        初始化 LogitsProcessor
        
        Args:
            z_proj_vocab: 投影到词表空间的 z 向量 [vocab_size]
                         从 CVAE 的 z [latent_dim] 投影而来
        """
        self.z_proj_vocab = z_proj_vocab
        # 确保在正确的设备上
        if self.z_proj_vocab.device.type == 'cpu':
            logger.warning("[ZInjectionLogitsProcessor] z_proj_vocab 在 CPU 上，性能可能受影响")
        
        logger.info(f"[ZInjectionLogitsProcessor] 初始化完成")
        logger.info(f"  z_proj_vocab 形状: {self.z_proj_vocab.shape}")
        logger.info(f"  z_proj_vocab 设备: {self.z_proj_vocab.device}")
        logger.info(f"  z_proj_vocab 统计: 均值={self.z_proj_vocab.mean().item():.6f}, "
                   f"标准差={self.z_proj_vocab.std().item():.6f}")
    
    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        """
        在采样前修改 logits
        
        Args:
            token_ids: 当前已生成的 token IDs（vLLM 传入）
            logits: 当前的 logits [vocab_size]
        
        Returns:
            修改后的 logits [vocab_size]
        """
        # 🔧 关键：将 z_proj_vocab 加到 logits 上
        # logits' = logits + z_proj_vocab
        # 这会改变 token 的采样分布，引导生成朝 z 代表的方向发展
        
        # 确保在同一设备上
        if self.z_proj_vocab.device != logits.device:
            self.z_proj_vocab = self.z_proj_vocab.to(logits.device)
        
        modified_logits = logits + self.z_proj_vocab
        
        return modified_logits
    
    def __repr__(self):
        return (f"ZInjectionLogitsProcessor("
                f"z_proj_vocab_shape={self.z_proj_vocab.shape}, "
                f"device={self.z_proj_vocab.device})")


# =================================================================================
# LaRS Selector VAE 模型定义（与训练代码保持一致）
# =================================================================================
class LaRS_Selector_VAE(nn.Module):
    """
    LaRS (Latent Reasoning Skill) Selector VAE
    用于从问题嵌入生成潜在推理技能向量 z
    """
    def __init__(self, embedding_dim: int, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        # Encoder: (question_emb + rationale_emb) -> posterior distribution
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # Reasoning Policy: question_emb -> prior distribution
        self.reasoning_policy = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # Decoder: (question_emb + z) -> reconstructed rationale_emb
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧：从 N(mu, var) 采样
        Args:
            mu: 均值 [batch, latent_dim]
            logvar: log(方差) [batch, latent_dim]
        Returns:
            z: 采样的潜在向量 [batch, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, question_emb: torch.Tensor, rationale_emb: torch.Tensor):
        """
        完整的 forward（训练时使用）
        """
        # Posterior: q(z|x,y)
        combined_qr = torch.cat([question_emb, rationale_emb], dim=1)
        posterior_params = self.encoder(combined_qr)
        posterior_mu, posterior_logvar = posterior_params.chunk(2, dim=-1)
        
        # Prior: p(z|x)
        prior_params = self.reasoning_policy(question_emb)
        prior_mu, prior_logvar = prior_params.chunk(2, dim=-1)
        
        # Sample z from posterior
        z = self.reparameterize(posterior_mu, posterior_logvar)
        
        # Decode
        combined_qz = torch.cat([question_emb, z], dim=1)
        reconstructed_rationale_emb = self.decoder(combined_qz)
        
        return (reconstructed_rationale_emb, posterior_mu, posterior_logvar, 
                prior_mu, prior_logvar)
    
    def sample_z(self, question_emb: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        从先验分布 p(z|x) 采样多个 z（推理时使用）
        Args:
            question_emb: 问题嵌入 [batch, embedding_dim]
            num_samples: 采样数量
        Returns:
            z_samples: [batch * num_samples, latent_dim]
        """
        with torch.no_grad():
            # 获取先验分布参数
            prior_params = self.reasoning_policy(question_emb)
            prior_mu, prior_logvar = prior_params.chunk(2, dim=-1)
            
            # 采样多个 z
            z_samples = []
            for _ in range(num_samples):
                z = self.reparameterize(prior_mu, prior_logvar)
                z_samples.append(z)
            
            # [num_samples, batch, latent_dim] -> [batch * num_samples, latent_dim]
            return torch.cat(z_samples, dim=0)


# =================================================================================
# CVAE Branching Manager
# =================================================================================
class CVAEBranchingManager:
    """
    CVAE 分叉管理器
    负责：
    1. 加载 CVAE 模型和 Embedding 模型
    2. 将文本转换为嵌入向量
    3. 采样多个 z 向量
    4. 注入 z 到模型的 attention 层
    """
    
    def __init__(
        self,
        cvae_model_path: str,
        embedding_model_path: str,
        latent_dim: int = 128,
        embedding_dim: int = 1536,  # 你训练时用的维度
        device: str = "cuda",
        injection_layers: Union[str, int] = "all"
    ):
        """
        初始化 CVAE 分叉管理器
        
        Args:
            cvae_model_path: CVAE 模型路径
            embedding_model_path: Embedding 模型路径（如 deberta-v2-xlarge）
            latent_dim: 潜在向量维度
            embedding_dim: 嵌入向量维度
            device: 设备（cuda/cpu）
            injection_layers: 注入层数
                - "all": 所有 attention 层
                - int: 从最后一层开始往前数的层数（如 4 表示最后 4 层）
        """
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.injection_layers = injection_layers
        
        logger.info(f"正在初始化 CVAE 分叉管理器...")
        logger.info(f"  CVAE 模型路径: {cvae_model_path}")
        logger.info(f"  Embedding 模型路径: {embedding_model_path}")
        logger.info(f"  注入层配置: {injection_layers}")
        
        # 1. 加载 Embedding 模型
        self.embedding_model, self.tokenizer = self._load_embedding_model(
            embedding_model_path
        )
        
        # 2. 加载 CVAE 模型
        self.cvae_model = self._load_cvae_model(cvae_model_path)
        
        # 3. 创建 z 投影层（将 z 投影到模型的 hidden_dim）
        # 注意：这个 hidden_dim 是你的 LLM 的隐藏层维度，需要动态获取
        self.z_projection_layers = {}  # 每一层可能需要不同的投影
        
        # 4. Hook handles（用于移除 hook）
        self.hook_handles = []
        
        logger.info("✅ CVAE 分叉管理器初始化完成")
    
    def _load_embedding_model(self, model_path: str):
        """
        加载 Embedding 模型（如 deberta-v2-xlarge）
        """
        from transformers import AutoModel, AutoTokenizer
        
        logger.info(f"正在加载 Embedding 模型: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModel.from_pretrained(model_path).to(self.device)
        model.eval()
        
        logger.info(f"✅ Embedding 模型加载完成")
        return model, tokenizer
    
    def _load_cvae_model(self, model_path: str) -> LaRS_Selector_VAE:
        """
        加载训练好的 CVAE 模型
        """
        logger.info(f"正在加载 CVAE 模型: {model_path}")
        
        # 创建模型
        cvae = LaRS_Selector_VAE(
            embedding_dim=self.embedding_dim,
            latent_dim=self.latent_dim
        )
        
        # 加载权重
        state_dict = torch.load(model_path, map_location=self.device)
        cvae.load_state_dict(state_dict)
        cvae.to(self.device)
        cvae.eval()
        
        logger.info(f"✅ CVAE 模型加载完成")
        logger.info(f"  Embedding 维度: {self.embedding_dim}")
        logger.info(f"  Latent 维度: {self.latent_dim}")
        
        return cvae
    
    def text_to_embedding(self, text: str) -> torch.Tensor:
        """
        将文本转换为嵌入向量
        
        Args:
            text: 输入文本
        Returns:
            embedding: [1, embedding_dim]
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # 使用 mean pooling
            outputs = self.embedding_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)  # [1, embedding_dim]
            
            return embedding
    
    def sample_z_from_text(
        self,
        text: str,
        num_samples: int = 4
    ) -> torch.Tensor:
        """
        从文本采样多个 z 向量
        
        Args:
            text: 输入文本（question + 已生成的前缀）
            num_samples: 采样数量
        Returns:
            z_samples: [num_samples, latent_dim]
        """
        # 1. 文本 -> 嵌入向量
        question_emb = self.text_to_embedding(text)  # [1, embedding_dim]
        
        # 2. 采样 z
        z_samples = self.cvae_model.sample_z(question_emb, num_samples)
        
        logger.debug(f"从文本采样了 {num_samples} 个 z 向量")
        logger.debug(f"  文本长度: {len(text)} 字符")
        logger.debug(f"  z 形状: {z_samples.shape}")
        
        return z_samples
    
    def create_z_projection_layer(self, hidden_dim: int) -> nn.Module:
        """
        创建 z 投影层：将 z [latent_dim] 投影到 [hidden_dim]
        
        Args:
            hidden_dim: 目标模型的隐藏层维度
        Returns:
            投影层
        """
        if hidden_dim not in self.z_projection_layers:
            projection = nn.Linear(self.latent_dim, hidden_dim).to(self.device)
            # 初始化为小值，避免初始时对模型影响太大
            nn.init.normal_(projection.weight, mean=0.0, std=0.01)
            nn.init.zeros_(projection.bias)
            self.z_projection_layers[hidden_dim] = projection
            logger.debug(f"创建了 z 投影层: {self.latent_dim} -> {hidden_dim}")
        
        return self.z_projection_layers[hidden_dim]
    
    def register_attention_hooks(
        self,
        model: nn.Module,
        z: torch.Tensor,
        injection_mode: str = "add_to_last_token"
    ):
        """
        在模型的 attention 层注册 hook，注入 z
        
        Args:
            model: 目标 LLM 模型
            z: 潜在向量 [1, latent_dim]
            injection_mode: 注入模式
                - "add_to_last_token": 只在最后一个 token 加上 z
                - "add_to_all_tokens": 在所有 token 加上 z
        """
        # 清除之前的 hooks
        self.remove_hooks()
        
        # 找到所有 attention 层
        attention_layers = self._find_attention_layers(model)
        
        # 根据 injection_layers 配置决定注入哪些层
        layers_to_inject = self._select_layers_to_inject(attention_layers)
        
        logger.info(f"正在注册 attention hooks...")
        logger.info(f"  总 attention 层数: {len(attention_layers)}")
        logger.info(f"  注入层数: {len(layers_to_inject)}")
        logger.info(f"  注入模式: {injection_mode}")
        
        # 为每一层注册 hook
        for layer_idx, (layer_name, layer_module) in enumerate(layers_to_inject):
            hook_fn = self._create_hook_function(z, injection_mode, layer_name)
            handle = layer_module.register_forward_hook(hook_fn)
            self.hook_handles.append(handle)
        
        logger.info(f"✅ 已注册 {len(self.hook_handles)} 个 attention hooks")
    
    def _find_attention_layers(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """
        找到模型中所有的 attention 层
        
        Returns:
            [(layer_name, layer_module), ...]
        """
        attention_layers = []
        
        for name, module in model.named_modules():
            # 常见的 attention 层名称模式
            if any(keyword in name.lower() for keyword in [
                "attention", "attn", "self_attn", "self_attention"
            ]):
                # 排除子模块（如 attention.query, attention.key）
                if not any(sub in name.lower() for sub in ["query", "key", "value", "output"]):
                    attention_layers.append((name, module))
        
        logger.debug(f"找到 {len(attention_layers)} 个 attention 层")
        return attention_layers
    
    def _select_layers_to_inject(
        self,
        attention_layers: List[Tuple[str, nn.Module]]
    ) -> List[Tuple[str, nn.Module]]:
        """
        根据 injection_layers 配置选择要注入的层
        
        Args:
            attention_layers: 所有 attention 层
        Returns:
            要注入的层列表
        """
        if self.injection_layers == "all":
            return attention_layers
        
        elif isinstance(self.injection_layers, int):
            # 从最后一层开始往前数
            num_layers = self.injection_layers
            return attention_layers[-num_layers:]
        
        else:
            logger.warning(f"未知的 injection_layers 配置: {self.injection_layers}，使用 'all'")
            return attention_layers
    
    def _create_hook_function(
        self,
        z: torch.Tensor,
        injection_mode: str,
        layer_name: str
    ):
        """
        创建 hook 函数
        
        Args:
            z: 潜在向量 [1, latent_dim]
            injection_mode: 注入模式
            layer_name: 层名称（用于日志）
        Returns:
            hook 函数
        """
        def hook_fn(module, input, output):
            """
            Hook 函数：在 attention 层的输出上注入 z
            
            Args:
                module: attention 层模块
                input: 输入（通常是 tuple）
                output: 输出（通常是 tensor 或 tuple）
            Returns:
                修改后的 output
            """
            # 处理不同的输出格式
            if isinstance(output, tuple):
                # 有些模型返回 (hidden_states, attention_weights)
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None
            
            # 获取 hidden_dim
            hidden_dim = hidden_states.shape[-1]
            
            # 投影 z 到 hidden_dim
            projection_layer = self.create_z_projection_layer(hidden_dim)
            z_projected = projection_layer(z)  # [1, hidden_dim]
            
            # 根据注入模式修改 hidden_states
            if injection_mode == "add_to_last_token":
                # 只在最后一个 token 加上 z
                hidden_states[:, -1, :] = hidden_states[:, -1, :] + z_projected
            
            elif injection_mode == "add_to_all_tokens":
                # 在所有 token 加上 z（广播）
                hidden_states = hidden_states + z_projected.unsqueeze(1)
            
            else:
                logger.warning(f"未知的 injection_mode: {injection_mode}")
            
            # 返回修改后的输出
            if rest is not None:
                return (hidden_states,) + rest
            else:
                return hidden_states
        
        return hook_fn
    
    def register_fusion_hooks(
        self,
        model: nn.Module,
        z: torch.Tensor,
        fusion_mode: str = "input"
    ):
        """
        注册融合 hooks（支持 input/psa/softmax 三种方式）
        
        Args:
            model: LLM 模型
            z: 潜在向量 [1, latent_dim]
            fusion_mode: 融合方式
                - "input": INPUT fusion（第一层输入）
                - "psa": PSA fusion（指定层，近似版本）
                - "softmax": SOFTMAX fusion（lm_head 输出）
        """
        # 清除之前的 hooks
        self.remove_hooks()
        
        logger.info(f"正在注册 {fusion_mode} fusion hooks...")
        
        if fusion_mode == "input":
            self._register_input_fusion_hook(model, z)
        elif fusion_mode == "psa":
            self._register_psa_fusion_hooks(model, z)
        elif fusion_mode == "softmax":
            self._register_softmax_fusion_hook(model, z)
        else:
            raise ValueError(f"未知的 fusion_mode: {fusion_mode}，支持的模式: input, psa, softmax")
        
        logger.info(f"✅ 已注册 {len(self.hook_handles)} 个 {fusion_mode} fusion hooks")
    
    # 🆕 公共方法：便于直接调用
    def register_input_fusion_hook(self, model: nn.Module, z: torch.Tensor):
        """
        公共接口：注册 INPUT Fusion hook
        """
        self.remove_hooks()
        self._register_input_fusion_hook(model, z)
        logger.info(f"✅ INPUT fusion hook 已注册")
    
    def register_psa_fusion_hooks(self, model: nn.Module, z: torch.Tensor):
        """
        公共接口：注册 PSA Fusion hooks
        """
        logger.info(f"[PSA Fusion] 🔧 开始注册 PSA fusion hooks...")
        logger.info(f"  z 形状: {z.shape}")
        logger.info(f"  model 类型: {type(model)}")
        self.remove_hooks()
        self._register_psa_fusion_hooks(model, z)
        logger.info(f"✅ PSA fusion hooks 已注册 ({len(self.hook_handles)} 个)")
    
    def register_softmax_fusion_hook(self, model: nn.Module, z: torch.Tensor):
        """
        公共接口：注册 SOFTMAX Fusion hook
        """
        self.remove_hooks()
        self._register_softmax_fusion_hook(model, z)
        logger.info(f"✅ SOFTMAX fusion hook 已注册")
    
    def _register_input_fusion_hook(self, model: nn.Module, z: torch.Tensor):
        """
        INPUT Fusion: 在第一层 Transformer 前注入 z
        
        Args:
            model: LLM 模型
            z: 潜在向量 [1, latent_dim]
        """
        # 获取 hidden_dim
        try:
            hidden_dim = model.config.hidden_size
        except AttributeError:
            # 如果没有 config，尝试从第一层获取
            first_layer = model.model.layers[0]
            hidden_dim = first_layer.self_attn.hidden_size if hasattr(first_layer, 'self_attn') else 4096
            logger.warning(f"无法从 model.config 获取 hidden_size，使用默认值: {hidden_dim}")
        
        # 创建投影层
        z_proj_layer = self.create_z_projection_layer(hidden_dim)
        z_proj = z_proj_layer(z)  # [1, hidden_dim]
        
        logger.debug(f"INPUT fusion: z {z.shape} -> z_proj {z_proj.shape}")
        
        # 创建 hook 函数
        def input_fusion_hook(module, input_tuple):
            """
            在第一层前修改 hidden_states
            
            Args:
                module: 第一层 Transformer
                input_tuple: (hidden_states, ...) 或其他格式
            Returns:
                修改后的 input_tuple
            """
            # 处理不同的输入格式
            if isinstance(input_tuple, tuple):
                hidden_states = input_tuple[0]
                rest = input_tuple[1:]
            else:
                hidden_states = input_tuple
                rest = ()
            
            # hidden_states shape: [batch, seq_len, hidden_dim]
            # 将 z_proj 加到所有 token 上
            hidden_states = hidden_states + z_proj.unsqueeze(1)
            
            # 返回修改后的输入
            if rest:
                return (hidden_states,) + rest
            else:
                return (hidden_states,)
        
        # 注册到第一层
        first_layer = model.model.layers[0]
        handle = first_layer.register_forward_pre_hook(input_fusion_hook)
        self.hook_handles.append(handle)
        
        logger.debug(f"✅ INPUT fusion hook 已注册到第一层")
    
    def _register_psa_fusion_hooks(self, model: nn.Module, z: torch.Tensor):
        """
        PSA Fusion（近似）: 在指定层注入 z
        
        Args:
            model: LLM 模型
            z: 潜在向量 [1, latent_dim]
        """
        logger.info(f"[PSA Fusion] 进入 _register_psa_fusion_hooks...")
        
        # 获取 hidden_dim
        try:
            hidden_dim = model.config.hidden_size
            logger.info(f"[PSA Fusion] 从 model.config 获取 hidden_size: {hidden_dim}")
        except AttributeError:
            hidden_dim = 4096
            logger.warning(f"无法从 model.config 获取 hidden_size，使用默认值: {hidden_dim}")
        
        # 创建投影层
        z_proj_layer = self.create_z_projection_layer(hidden_dim)
        z_proj = z_proj_layer(z)  # [1, hidden_dim]
        
        logger.info(f"[PSA Fusion] z {z.shape} -> z_proj {z_proj.shape}")
        
        # 获取所有 Transformer 层
        all_layers = model.model.layers
        
        # 根据 injection_layers 配置选择要注入的层
        if self.injection_layers == "all":
            layers_to_inject = list(range(len(all_layers)))
        elif isinstance(self.injection_layers, int):
            # 最后 N 层
            num_layers = self.injection_layers
            layers_to_inject = list(range(len(all_layers) - num_layers, len(all_layers)))
        else:
            logger.warning(f"未知的 injection_layers 配置: {self.injection_layers}，使用最后 4 层")
            layers_to_inject = list(range(len(all_layers) - 4, len(all_layers)))
        
        logger.debug(f"PSA fusion: 总层数 {len(all_layers)}, 注入层: {layers_to_inject}")
        
        # 用于控制打印次数的计数器
        call_counter = {'count': 0}
        
        # 创建 hook 函数
        def psa_fusion_hook(module, input, output):
            """
            在指定层修改 hidden_states
            
            Args:
                module: Transformer 层
                input: 输入
                output: 输出 (hidden_states, ...) 或其他格式
            Returns:
                修改后的 output
            """
            # 🔧 关键修复：在 hook 函数内部重新获取 logger，确保在 worker 进程中也能正常工作
            import logging
            import os
            hook_logger = logging.getLogger(__name__)
            hook_logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
            
            call_counter['count'] += 1
            
            # 🔧 强制使用 print 输出，确保能看到 hook 被调用
            print(f"\n{'='*80}", flush=True)
            print(f"[PSA Fusion Hook] 🎯 第 {call_counter['count']} 次被调用", flush=True)
            print(f"{'='*80}", flush=True)
            
            # 只在前 5 次调用时打印详细日志（避免刷屏）
          
            
            hook_logger.info(f"\n{'='*80}")
            hook_logger.info(f"[PSA Fusion Hook] 🎯 第 {call_counter['count']} 次被调用")
            hook_logger.info(f"{'='*80}")
            
            # 处理不同的输出格式
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
                
               
                print(f"[PSA Fusion] 📊 输出格式分析:", flush=True)
                print(f"  output 是 tuple，长度: {len(output)}", flush=True)
                print(f"  hidden_states 形状: {hidden_states.shape}", flush=True)
                print(f"  hidden_states dtype: {hidden_states.dtype}", flush=True)
                    
                hook_logger.info(f"[PSA Fusion] 📊 输出格式分析:")
                hook_logger.info(f"  output 是 tuple，长度: {len(output)}")
                hook_logger.info(f"  hidden_states 形状: {hidden_states.shape}")
                hook_logger.info(f"  hidden_states dtype: {hidden_states.dtype}")
                
                # 只在第一次打印 rest 详情
                if call_counter['count'] == 1:
                    for i, item in enumerate(rest):
                        print(f"\n  rest[{i}] 的详细信息:", flush=True)
                        print(f"    类型: {type(item)}", flush=True)
                        hook_logger.info(f"\n  rest[{i}] 的详细信息:")
                        hook_logger.info(f"    类型: {type(item)}")
                        
                        if isinstance(item, torch.Tensor):
                            print(f"    形状: {item.shape}", flush=True)
                            print(f"    dtype: {item.dtype}", flush=True)
                            hook_logger.info(f"    形状: {item.shape}")
                            hook_logger.info(f"    dtype: {item.dtype}")
                        elif isinstance(item, tuple):
                            print(f"    tuple 长度: {len(item)}", flush=True)
                            hook_logger.info(f"    tuple 长度: {len(item)}")
                            for j, sub_item in enumerate(item):
                                if isinstance(sub_item, torch.Tensor):
                                    print(f"      [{j}] Tensor 形状: {sub_item.shape}, dtype: {sub_item.dtype}", flush=True)
                                    hook_logger.info(f"      [{j}] Tensor 形状: {sub_item.shape}, dtype: {sub_item.dtype}")
                                else:
                                    print(f"      [{j}] 类型: {type(sub_item)}", flush=True)
                                    hook_logger.info(f"      [{j}] 类型: {type(sub_item)}")
                        elif item is None:
                            print(f"    值: None", flush=True)
                            hook_logger.info(f"    值: None")
                        else:
                            print(f"    内容: {str(item)[:100]}", flush=True)
                            hook_logger.info(f"    内容: {str(item)[:100]}")
            else:
                hidden_states = output
                rest = ()
                
                print(f"[PSA Fusion] 📊 输出格式分析:", flush=True)
                print(f"  output 不是 tuple，直接是 Tensor", flush=True)
                print(f"  hidden_states 形状: {hidden_states.shape}", flush=True)
                print(f"  hidden_states dtype: {hidden_states.dtype}", flush=True)
                
                hook_logger.info(f"[PSA Fusion] 📊 输出格式分析:")
                hook_logger.info(f"  output 不是 tuple，直接是 Tensor")
                hook_logger.info(f"  hidden_states 形状: {hidden_states.shape}")
                hook_logger.info(f"  hidden_states dtype: {hidden_states.dtype}")
        
            # ============ 注入 z_proj 到 hidden_states ============
           
            print(f"\n[PSA Fusion] 🔧 准备注入 z_proj:", flush=True)
            print(f"  z_proj 形状: {z_proj.shape}", flush=True)
            print(f"  z_proj dtype: {z_proj.dtype}", flush=True)
            print(f"  z_proj 统计: 均值={z_proj.mean().item():.6f}, 标准差={z_proj.std().item():.6f}", flush=True)
            
            hook_logger.info(f"\n[PSA Fusion] 🔧 准备注入 z_proj:")
            hook_logger.info(f"  z_proj 形状: {z_proj.shape}")
            hook_logger.info(f"  z_proj dtype: {z_proj.dtype}")
            hook_logger.info(f"  z_proj 统计: 均值={z_proj.mean().item():.6f}, 标准差={z_proj.std().item():.6f}")
            
            # 注入前的 hidden_states 信息
            print(f"\n[PSA Fusion] 📊 注入前 hidden_states (最后一个 token):", flush=True)
            print(f"  形状: {hidden_states[:, -1, :].shape}", flush=True)
            print(f"  均值: {hidden_states[:, -1, :].mean().item():.6f}", flush=True)
            print(f"  标准差: {hidden_states[:, -1, :].std().item():.6f}", flush=True)
            print(f"  最小值: {hidden_states[:, -1, :].min().item():.6f}", flush=True)
            print(f"  最大值: {hidden_states[:, -1, :].max().item():.6f}", flush=True)
            
            hook_logger.info(f"\n[PSA Fusion] 📊 注入前 hidden_states (最后一个 token):")
            hook_logger.info(f"  形状: {hidden_states[:, -1, :].shape}")
            hook_logger.info(f"  均值: {hidden_states[:, -1, :].mean().item():.6f}")
            hook_logger.info(f"  标准差: {hidden_states[:, -1, :].std().item():.6f}")
            hook_logger.info(f"  最小值: {hidden_states[:, -1, :].min().item():.6f}")
            hook_logger.info(f"  最大值: {hidden_states[:, -1, :].max().item():.6f}")
            
            # 注入：将 z_proj 加到最后一个 token
            # hidden_states shape: [batch, seq_len, hidden_dim]
            # z_proj shape: [1, hidden_dim] 或 [hidden_dim]
            hidden_states_before = hidden_states[:, -1, :].clone()
            hidden_states[:, -1, :] = hidden_states[:, -1, :] + z_proj.squeeze(0)
            
            # 注入后的 hidden_states 信息
         
            print(f"\n[PSA Fusion] 📊 注入后 hidden_states (最后一个 token):", flush=True)
            print(f"  形状: {hidden_states[:, -1, :].shape}", flush=True)
            print(f"  均值: {hidden_states[:, -1, :].mean().item():.6f}", flush=True)
            print(f"  标准差: {hidden_states[:, -1, :].std().item():.6f}", flush=True)
            print(f"  最小值: {hidden_states[:, -1, :].min().item():.6f}", flush=True)
            print(f"  最大值: {hidden_states[:, -1, :].max().item():.6f}", flush=True)
            
            hook_logger.info(f"\n[PSA Fusion] 📊 注入后 hidden_states (最后一个 token):")
            hook_logger.info(f"  形状: {hidden_states[:, -1, :].shape}")
            hook_logger.info(f"  均值: {hidden_states[:, -1, :].mean().item():.6f}")
            hook_logger.info(f"  标准差: {hidden_states[:, -1, :].std().item():.6f}")
            hook_logger.info(f"  最小值: {hidden_states[:, -1, :].min().item():.6f}")
            hook_logger.info(f"  最大值: {hidden_states[:, -1, :].max().item():.6f}")
            
            # 计算变化量
            delta = hidden_states[:, -1, :] - hidden_states_before
            print(f"\n[PSA Fusion] 📈 注入前后的变化:", flush=True)
            print(f"  delta 均值: {delta.mean().item():.6f}", flush=True)
            print(f"  delta 标准差: {delta.std().item():.6f}", flush=True)
            print(f"  delta 最大绝对值: {delta.abs().max().item():.6f}", flush=True)
            print(f"  ✅ zi 已成功注入到 hidden_states！", flush=True)
            print(f"{'='*80}\n", flush=True)
            
            hook_logger.info(f"\n[PSA Fusion] 📈 注入前后的变化:")
            hook_logger.info(f"  delta 均值: {delta.mean().item():.6f}")
            hook_logger.info(f"  delta 标准差: {delta.std().item():.6f}")
            hook_logger.info(f"  delta 最大绝对值: {delta.abs().max().item():.6f}")
            hook_logger.info(f"  ✅ zi 已成功注入到 hidden_states！")
            hook_logger.info(f"{'='*80}\n")
            
            # 返回修改后的输出
            if rest:
                return (hidden_states,) + rest
            else:
                return hidden_states
        
        # 为每一层注册 hook
        logger.info(f"[PSA Fusion] 开始为 {len(layers_to_inject)} 层注册 hook...")
        for layer_idx in layers_to_inject:
            layer = all_layers[layer_idx]
            logger.info(f"[PSA Fusion] 正在为 layer {layer_idx} 注册 hook...")
            handle = layer.register_forward_hook(psa_fusion_hook)
            self.hook_handles.append(handle)
            logger.info(f"[PSA Fusion] Layer {layer_idx} hook 注册成功")
        
        logger.info(f"✅ PSA fusion hooks 已全部注册到 {len(layers_to_inject)} 层")
    
    
    def _register_softmax_fusion_hook(self, model: nn.Module, z: torch.Tensor):
        """
        SOFTMAX Fusion: 在 lm_head 输出后注入 z
        
        Args:
            model: LLM 模型
            z: 潜在向量 [1, latent_dim]
        """
        # 获取 vocab_size
        try:
            vocab_size = model.config.vocab_size
        except AttributeError:
            vocab_size = 151936  # Qwen2 的词表大小
            logger.warning(f"无法从 model.config 获取 vocab_size，使用默认值: {vocab_size}")
        
        # 创建投影层（投影到词表大小）
        if vocab_size not in self.z_projection_layers:
            projection = nn.Linear(self.latent_dim, vocab_size, bias=True).to(self.device)
            # 初始化为小值，避免初始时对模型影响太大
            nn.init.normal_(projection.weight, mean=0.0, std=0.01)
            nn.init.zeros_(projection.bias)
            self.z_projection_layers[vocab_size] = projection
            logger.debug(f"创建了 SOFTMAX 投影层: {self.latent_dim} -> {vocab_size}")
        
        z_proj_layer = self.z_projection_layers[vocab_size]
        z_proj_vocab = z_proj_layer(z)  # [1, vocab_size]
        
        logger.debug(f"SOFTMAX fusion: z {z.shape} -> z_proj_vocab {z_proj_vocab.shape}")
        
        # 创建 hook 函数
        def softmax_fusion_hook(module, input, output):
            """
            在 lm_head 后修改 logits
            
            Args:
                module: lm_head 层
                input: 输入
                output: logits [batch, seq_len, vocab_size]
            Returns:
                修改后的 logits
            """
            # output shape: [batch, seq_len, vocab_size]
            logits = output
            
            # 将 z_proj_vocab 加到所有位置的 logits 上
            # z_proj_vocab shape: [1, vocab_size]
            # 需要广播到 [batch, seq_len, vocab_size]
            logits = logits + z_proj_vocab.unsqueeze(1)
            
            return logits
        
        # 注册到 lm_head
        lm_head = model.lm_head
        handle = lm_head.register_forward_hook(softmax_fusion_hook)
        self.hook_handles.append(handle)
        
        logger.debug(f"✅ SOFTMAX fusion hook 已注册到 lm_head")
    
    def project_z_to_vocab(
        self, 
        z: torch.Tensor, 
        vocab_size: int = 151936
    ) -> torch.Tensor:
        """
        将 z 投影到词表空间（用于 SOFTMAX Fusion）
        
        这个方法用于 ForwardContext 方案，直接返回 z_proj_vocab tensor
        
        Args:
            z: 潜在向量 [1, latent_dim]
            vocab_size: 词表大小（默认 Qwen3 的 151936）
        
        Returns:
            z_proj_vocab: 投影后的向量 [vocab_size]
        """
        logger.info(f"[SOFTMAX Fusion] 投影 z 到词表空间...")
        logger.info(f"  z 形状: {z.shape}")
        logger.info(f"  vocab_size: {vocab_size}")
        
        # 创建投影层（投影到词表大小）
        if vocab_size not in self.z_projection_layers:
            projection = nn.Linear(self.latent_dim, vocab_size, bias=True).to(self.device)
            # 初始化为小值，避免初始时对模型影响太大
            nn.init.normal_(projection.weight, mean=0.0, std=0.01)
            nn.init.zeros_(projection.bias)
            self.z_projection_layers[vocab_size] = projection
            logger.info(f"  创建了 SOFTMAX 投影层: {self.latent_dim} -> {vocab_size}")
        
        # 投影 z 到词表空间
        z_proj_layer = self.z_projection_layers[vocab_size]
        z_proj_vocab = z_proj_layer(z)  # [1, vocab_size]
        z_proj_vocab = z_proj_vocab.squeeze(0)  # [vocab_size]
        
        logger.info(f"  z {z.shape} -> z_proj_vocab {z_proj_vocab.shape}")
        logger.info(f"  z_proj_vocab 统计: 均值={z_proj_vocab.mean().item():.6f}, "
                   f"标准差={z_proj_vocab.std().item():.6f}")
        logger.info(f"✅ SOFTMAX 投影完成")
        
        return z_proj_vocab
    
    def project_z_to_hidden(
        self, 
        z: torch.Tensor, 
        hidden_dim: int
    ) -> torch.Tensor:
        """
        将 z 投影到 hidden_dim 空间（用于 INPUT Fusion）
        
        这个方法用于 ForwardContext 方案，直接返回 z_proj_input tensor
        
        Args:
            z: 潜在向量 [1, latent_dim]
            hidden_dim: 模型的 hidden_size（如 4096）
        
        Returns:
            z_proj_input: 投影后的向量 [1, hidden_dim]
        """
        logger.info(f"[INPUT Fusion] 投影 z 到 hidden 空间...")
        logger.info(f"  z 形状: {z.shape}")
        logger.info(f"  hidden_dim: {hidden_dim}")
        
        # 创建投影层（投影到 hidden_dim）
        if hidden_dim not in self.z_projection_layers:
            projection = nn.Linear(self.latent_dim, hidden_dim, bias=False).to(self.device)
            # 初始化为小值，避免初始时对模型影响太大
            nn.init.normal_(projection.weight, mean=0.0, std=0.01)
            self.z_projection_layers[hidden_dim] = projection
            logger.info(f"  创建了 INPUT 投影层: {self.latent_dim} -> {hidden_dim}")
        
        # 投影 z 到 hidden 空间
        z_proj_layer = self.z_projection_layers[hidden_dim]
        z_proj_input = z_proj_layer(z)  # [1, hidden_dim]
        
        logger.info(f"  z {z.shape} -> z_proj_input {z_proj_input.shape}")
        logger.info(f"  z_proj_input 统计: 均值={z_proj_input.mean().item():.6f}, "
                   f"标准差={z_proj_input.std().item():.6f}")
        logger.info(f"✅ INPUT 投影完成")
        
        return z_proj_input
    
    def set_z_injection_for_input_fusion(
        self,
        z: torch.Tensor,
        hidden_dim: int
    ):
        """
        🆕 通过 vLLM ForwardContext 设置 INPUT Fusion 注入配置
        
        ⚠️ 互斥性保证：不能与 SOFTMAX Fusion 同时使用
        
        Args:
            z: 潜在向量 [1, latent_dim]
            hidden_dim: 模型的 hidden_size
        
        Returns:
            ZInjectionConfig: 用于设置到 vLLM engine 的配置
        """
        from vllm.forward_context import ZInjectionConfig, validate_z_injection_config
        
        logger.info(f"[INPUT Fusion] 🔧 准备设置 z_injection_config（ForwardContext 方案）")
        
        # 1. 投影 z 到 hidden_dim
        z_proj_input = self.project_z_to_hidden(z, hidden_dim)
        
        # 2. 创建 ZInjectionConfig（只设置 INPUT 相关字段）
        config = ZInjectionConfig(
            mode="input",
            z_proj_input=z_proj_input,
            z_proj_vocab=None  # 🔒 明确设置为 None，确保互斥
        )
        
        # 3. 验证互斥性
        validate_z_injection_config(config)
        
        logger.info(f"✅ [INPUT Fusion] z_injection_config 已创建（SOFTMAX 已禁用）")
        logger.info(f"  mode: {config.mode}")
        logger.info(f"  z_proj_input: {config.z_proj_input.shape if config.z_proj_input is not None else None}")
        logger.info(f"  z_proj_vocab: {config.z_proj_vocab}")
        
        return config
    
    def set_z_injection_for_softmax_fusion(
        self,
        z: torch.Tensor,
        vocab_size: int = 151936
    ):
        """
        🆕 通过 vLLM ForwardContext 设置 SOFTMAX Fusion 注入配置
        
        ⚠️ 互斥性保证：不能与 INPUT Fusion 同时使用
        
        Args:
            z: 潜在向量 [1, latent_dim]
            vocab_size: 词表大小
        
        Returns:
            ZInjectionConfig: 用于设置到 vLLM engine 的配置
        """
        from vllm.forward_context import ZInjectionConfig, validate_z_injection_config
        
        logger.info(f"[SOFTMAX Fusion] 🔧 准备设置 z_injection_config（ForwardContext 方案）")
        
        # 1. 投影 z 到 vocab_size
        z_proj_vocab = self.project_z_to_vocab(z, vocab_size)
        
        # 2. 创建 ZInjectionConfig（只设置 SOFTMAX 相关字段）
        config = ZInjectionConfig(
            mode="softmax",
            z_proj_input=None,  # 🔒 明确设置为 None，确保互斥
            z_proj_vocab=z_proj_vocab
        )
        
        # 3. 验证互斥性
        validate_z_injection_config(config)
        
        logger.info(f"✅ [SOFTMAX Fusion] z_injection_config 已创建（INPUT 已禁用）")
        logger.info(f"  mode: {config.mode}")
        logger.info(f"  z_proj_input: {config.z_proj_input}")
        logger.info(f"  z_proj_vocab: {config.z_proj_vocab.shape if config.z_proj_vocab is not None else None}")
        
        return config
    
    def create_softmax_logits_processor(
        self, 
        z: torch.Tensor, 
        vocab_size: int = 151936
    ) -> ZInjectionLogitsProcessor:
        """
        创建 SOFTMAX Fusion 的 LogitsProcessor（已弃用，使用 ForwardContext 方案）
        
        这个方法用于 vLLM 的 logits_processor 机制，但 vLLM V1 不支持
        保留此方法以便向后兼容
        
        Args:
            z: 潜在向量 [1, latent_dim]
            vocab_size: 词表大小（默认 Qwen3 的 151936）
        
        Returns:
            ZInjectionLogitsProcessor 实例
        """
        logger.warning(f"[SOFTMAX Fusion] create_softmax_logits_processor() 已弃用，"
                      f"建议使用 project_z_to_vocab()")
        
        # 使用新方法获取 z_proj_vocab
        z_proj_vocab = self.project_z_to_vocab(z, vocab_size)
        
        # 创建 LogitsProcessor（向后兼容）
        logits_processor = ZInjectionLogitsProcessor(z_proj_vocab)
        
        return logits_processor
    
    def remove_hooks(self):
        """
        移除所有注册的 hooks
        """
        if not hasattr(self, 'hook_handles'):
            return
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        logger.debug("已移除所有 hooks")
    
    def __del__(self):
        """
        析构函数：确保 hooks 被移除
        """
        try:
            self.remove_hooks()
        except:
            pass  # 忽略析构时的任何错误


# =================================================================================
# 便捷函数
# =================================================================================
def create_cvae_manager(
    cvae_model_path: str = "/nas/dhl/CVAE/models/GSM8K-MATH-trained/lars_selector_GSM8K-MATH.pth",
    embedding_model_path: str = "/nas/dhl/CVAE/models/deberta-v2-xlarge",
    injection_layers: Union[str, int] = "all",
    latent_dim: int = 128,
    embedding_dim: int = 1536,
    device: str = "cuda"
) -> CVAEBranchingManager:
    """
    便捷函数：创建 CVAE 分叉管理器
    
    Args:
        cvae_model_path: CVAE 模型路径
        embedding_model_path: Embedding 模型路径
        injection_layers: 注入层配置（"all" 或 int）
        latent_dim: 潜在向量维度（默认128）
        embedding_dim: 嵌入向量维度（默认1536，DeBERTa-v2-xlarge）
        device: 设备
    Returns:
        CVAEBranchingManager 实例
    """
    return CVAEBranchingManager(
        cvae_model_path=cvae_model_path,
        embedding_model_path=embedding_model_path,
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
        device=device,
        injection_layers=injection_layers
    )

