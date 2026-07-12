from transformers import PretrainedConfig


class TinyMindConfig(PretrainedConfig):
    model_type = "tinymind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
import math


# RMSNorm实现
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * self._norm(x)


# Yarn
def precompute_freqs_cis(
    dim: int, end: int, rope_base, rope_scaling: Optional[dict] = None
):
    # 初始化RoPE频率
    freqs, attn_factor = (1 / rope_base ** (torch.range(0, dim, 2).float() / dim), 1.0)

    if rope_scaling is not None:
        # 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界对应的波长比例，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界对应的波长比例，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 推断的长度大于训练的长度时，使用缩放
        if end > orig_max:
            # 波长比b到索引i的映射
            inv_dim = lambda b: (
                dim * math.log(orig_max / (b * 2 * math.pi)) / 2 * math.log(rope_base)
            )

            # 划分低频维度和高频维度
            # 0到low是高频，low到high是中间过渡，high到dim//2是低频
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 计算缩放因子
            # 在low之前，缩放因子为0，在high之后，缩放因子为1，在low和high之间线性过渡
            # clamp确保ramp在0到1之间，式子小于0时会被设置为0，大于1时会被设置为1，中间部分线性过渡到1
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            freqs = freqs * (1 - ramp + ramp / factor)

    # 根据end生成位置索引t
    t = torch.arange(end, device=freqs.device).float()

    # 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    # 将 Cos 和 Sin 分别重复两次，得到 [θ0, θ0, θ1, θ1, θ2, θ2, θ3, θ3, ... ] 这样的向量
    freqs_cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # 【seq_len,dim】
    freqs_sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # 【seq_len,dim】

    return freqs_cos, freqs_sin


# RoPE实现
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # 把 [x0, x1, x2, x3...] 变成 [-x1, x0, -x3, x2...]
    def rotate_every_two(x):
        # 将最后一个维度切分：一分为二（从偶数索引抽取x0, x2...；从奇数索引抽取x1, x3...）
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        # 把 x_odd 变负放前面，x_even 放后面，再利用 stack + flatten (或者 stack + view) 交错拼接回去
        x_rotated = torch.stack((-x_odd, x_even), dim=-1)
        return x_rotated.flatten(-2)  # 重新展开为原来的形状

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_every_two(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_every_two(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed
