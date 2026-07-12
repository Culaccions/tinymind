# TinyMind

从零开始实现的小参数量大语言模型，基于 minimind 项目学习。本项目旨在深入理解大语言模型的核心原理，通过手写实现每一个关键组件，帮助建立对 LLM 的直观理解。

## 特性

- **RMSNorm** — 高效的层归一化实现
- **RoPE** — 旋转位置编码，支持长序列外推
- **YARN** — 基于波长分析的位置编码缩放方法，实现上下文长度扩展
- **GQA** — 分组查询注意力，平衡性能与显存
- **Flash Attention** — 利用 PyTorch 原生 SDPA 加速注意力计算
- **MoE** — 混合专家架构（可选），提升模型容量
- **KV Cache** — 推理加速，避免重复计算

## 架构概览

### 模型结构

![](img/image.png)

### MoE 模块

![](img/LLM-structure-moe.jpg)

## 安装

### 环境要求

- Python >= 3.12
- CUDA 11.8+（推荐）

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/your-username/tinymind.git
cd tinymind

# 使用 uv 安装（推荐）
uv sync

# 或使用 pip
pip install -e .
```

## 项目结构

````
tinymind/
├── model/
│   └── model.py          # 模型定义：Config、RMSNorm、RoPE、Attention 等
├── trainer/
│   ├── train_pretrain.py  # 预训练脚本
│   └── trainer_utils.py   # 训练工具函数
├── dataset/
│   └── lm_dataset.py      # 数据集处理
├── img/                   # 文档图片资源
├── main.py                # 入口文件
└── pyproject.toml         # 项目配置
````

## 模型配置

默认配置参数如下：

| 参数                      | 默认值  | 说明             |
| ------------------------- | ------- | ---------------- |
| `hidden_size`             | 512     | 隐藏层维度       |
| `num_hidden_layers`       | 8       | Transformer 层数 |
| `num_attention_heads`     | 8       | 注意力头数       |
| `num_key_value_heads`     | 2       | KV 头数（GQA）   |
| `vocab_size`              | 6400    | 词表大小         |
| `max_position_embeddings` | 32768   | 最大序列长度     |
| `rope_theta`              | 1000000 | RoPE 基础频率    |
| `hidden_act`              | silu    | 激活函数         |
| `rms_norm_eps`            | 1e-5    | RMSNorm epsilon  |

### MoE 配置

| 参数                  | 默认值 | 说明                    |
| --------------------- | ------ | ----------------------- |
| `use_moe`             | False  | 是否启用 MoE            |
| `n_routed_experts`    | 4      | 路由专家数              |
| `n_shared_experts`    | 1      | 共享专家数              |
| `num_experts_per_tok` | 2      | 每个 token 激活的专家数 |
| `aux_loss_alpha`      | 0.01   | 辅助损失系数            |

## 技术细节

### RMSNorm

RMSNorm 相比 LayerNorm 去除了均值中心化步骤，计算更高效：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \cdot \gamma
$$

其中 $\gamma$ 是可训练的缩放参数，$\epsilon$ 用于数值稳定性。

### RoPE 旋转位置编码

RoPE 通过旋转矩阵将绝对位置信息编码为相对位置信息。对于位置 $m$ 和 $n$ 的向量，注意力分数仅依赖于相对位置 $(n-m)$：

$$
\text{Score} = q^T \cdot R((n-m)\theta) \cdot k
$$

分组策略：将高维向量两两分组，每组使用不同频率 $\theta_i = 10000^{-\frac{2(i-1)}{d_{\text{model}}}}$，低维组频率快（捕获局部信息），高维组频率慢（捕获全局信息）。

![](img/rope%E5%88%86%E6%B2%BB.png)

### YARN 位置编码缩放

YARN 通过波长分析实现上下文长度扩展：

- **波长** $\lambda_i = \frac{2\pi}{\theta_i}$：完成一次完整旋转所需的 token 距离
- **高频维度**（$\lambda_i \ll L$）：负责局部相对位置，不缩放
- **低频维度**（$\lambda_i \gg L$）：负责全局绝对位置，全量缩放
- **中间维度**：线性插值过渡

![](img/YARN%E5%A4%84%E7%90%86%E6%96%B9%E6%B3%95.png)

### GQA（分组查询注意力）

GQA 将注意力头分为若干组，每组共享一对 KV head，在保持模型质量的同时显著减少 KV Cache 的显存占用：

- Q head 数：`num_attention_heads`（默认 8）
- KV head 数：`num_key_value_heads`（默认 2）
- 重复次数：`rep_n = num_attention_heads / num_key_value_heads`

### Flash Attention

利用 PyTorch 的 `scaled_dot_product_attention` 实现硬件级注意力加速，在训练和预填充阶段自动启用。

## 使用方式

```bash
# 预训练（待实现）
python trainer/train_pretrain.py
```

## 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 架构
- [RoFormer](https://arxiv.org/abs/2104.09864) — 旋转位置编码
- [YaRN](https://arxiv.org/abs/2309.00071) — 上下文长度扩展
- [GQA](https://arxiv.org/abs/2305.13245) — 分组查询注意力
- [FlashAttention](https://arxiv.org/abs/2205.14135) — 高效注意力实现
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — 混合专家架构

## License

MIT

