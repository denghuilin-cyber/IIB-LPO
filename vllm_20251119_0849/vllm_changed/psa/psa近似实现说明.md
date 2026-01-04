非常好的问题！让我详细解释 PSA Fusion 的机制。

## 🎯 核心机制解析

### 1️⃣ **自回归生成的基本原理**

```python
# LLM 是自回归的，每次生成一个新 token

输入序列：["The", "cat", "is"]  (已生成的 3 个 token)
                                    ↓
                            [Transformer Layers]
                                    ↓
输出：预测第 4 个 token 是 "sleeping" 的概率最高
```

### 2️⃣ **hidden_states 的形状和含义**

```python
# 在 Transformer 的某一层：

hidden_states.shape = [batch_size, seq_len, hidden_dim]
                    = [1, 50, 4096]
                      ↑   ↑    ↑
                      |   |    └─ 特征维度（模型的 hidden size）
                      |   └────── 序列长度（当前已有 50 个 token）
                      └────────── batch size

# 索引含义：
hidden_states[:, 0, :]   # 第 1 个 token 的表征 [batch, 4096]
hidden_states[:, 1, :]   # 第 2 个 token 的表征 [batch, 4096]
...
hidden_states[:, -1, :]  # 最后一个 token 的表征 [batch, 4096] ⭐
```

### 3️⃣ **为什么只修改 `[:, -1, :]`（最后一个 token）？**

#### 场景理解：

```python
当前状态：已生成 ["The", "cat", "is"]
正在生成：第 4 个 token

Transformer 处理：
┌─────────────────────────────────────┐
│  Token 1: "The"    → hidden[0]      │ ← 已经生成，不需要修改
│  Token 2: "cat"    → hidden[1]      │ ← 已经生成，不需要修改
│  Token 3: "is"     → hidden[2]      │ ← 已经生成，不需要修改
│  Token 4: <生成中>  → hidden[-1]     │ ← ⭐ 当前正在预测，需要注入 z！
└─────────────────────────────────────┘
          ↓
    hidden[-1] 会经过 lm_head
          ↓
    预测词表概率：["sleeping": 0.8, "running": 0.15, ...]
```

#### PSA 注入的位置：

```python
# 在 Layer 24 (倒数第 4 层)

hidden_states[:, -1, :] = hidden_states[:, -1, :] + z_proj
                ↑
                └─ 只修改最后一个位置（正在生成的 token）

# 为什么不修改 [:, :, :]（所有位置）？
# 因为前面的 token 已经生成了，改它们没意义
# 只需要影响"当前正在生成"的 token
```

---

## 📊 完整的生成流程（带 PSA 注入）

### 场景：生成 "The cat is sleeping"

```python
Step 1: 输入 prompt ["The", "cat"]，生成 "is"
─────────────────────────────────────────────
Input: ["The", "cat"]
      ↓
[Layer 0]  hidden[:, 0] = "The"的表征
           hidden[:, 1] = "cat"的表征  ← 最后位置
      ↓
[Layer 1-23] ... (传播)
      ↓
[Layer 24] 🎯 PSA Hook 触发！
           hidden[:, 1, :] += z_proj  ← 注入 z 到最后位置
      ↓
[Layer 25-27] ... (继续传播，z 的影响被传递)
      ↓
[lm_head] 预测下一个 token → "is"


Step 2: 输入 ["The", "cat", "is"]，生成 "sleeping"
─────────────────────────────────────────────
Input: ["The", "cat", "is"]
      ↓
[Layer 0]  hidden[:, 0] = "The"
           hidden[:, 1] = "cat"
           hidden[:, 2] = "is"  ← 最后位置
      ↓
[Layer 24] 🎯 PSA Hook 触发！
           hidden[:, 2, :] += z_proj  ← 注入 z 到最后位置
      ↓
[Layer 25-27] ...
      ↓
[lm_head] 预测 → "sleeping"
```

---

## 🔍 为什么是 `[:, -1, :]` 而不是其他？

### 选项对比：

```python
# ❌ 选项 1：修改所有位置 [:, :, :]
hidden_states = hidden_states + z_proj
# 问题：会影响已生成的 token，可能破坏语义

# ❌ 选项 2：修改第一个位置 [:, 0, :]
hidden_states[:, 0, :] += z_proj
# 问题：第一个 token 是 prompt，已经固定，改了没意义

# ✅ 选项 3：修改最后一个位置 [:, -1, :]
hidden_states[:, -1, :] += z_proj
# 正确：影响"正在生成"的 token
```

---

## 🆚 三种 Fusion 模式对比

### INPUT Fusion（在第一层前）
```python
# 在 embedding 之后注入
┌─────────────────────────────────────┐
│  Embedding Layer                    │
│  所有 token: [seq_len, hidden_dim]  │
└─────────────────────────────────────┘
           ↓ 🎯 在这里注入 z
    hidden_states += z_proj  (全局影响)
           ↓
┌─────────────────────────────────────┐
│  Layer 0, 1, 2, ..., 27             │
│  所有层都受到 z 的影响               │
└─────────────────────────────────────┘
```

### PSA Fusion（在指定层后）
```python
┌─────────────────────────────────────┐
│  Layer 0-23 (不受 z 影响)            │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Layer 24                           │
└─────────────────────────────────────┘
           ↓ 🎯 在这里注入 z
    hidden[:, -1, :] += z_proj  (只影响最后一个 token)
           ↓
┌─────────────────────────────────────┐
│  Layer 25, 26, 27                   │
│  只有这几层受到 z 的影响             │
└─────────────────────────────────────┘
```

### SOFTMAX Fusion（在 lm_head 后）
```python
┌─────────────────────────────────────┐
│  Layer 0-27 (不受 z 影响)            │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  lm_head: hidden → logits           │
│  logits: [seq_len, vocab_size]      │
└─────────────────────────────────────┘
           ↓ 🎯 在这里注入 z
    logits += z_proj_vocab  (直接影响词表概率)
```

---

## 💡 核心总结

### ✅ **你的理解是对的！**

> "给每次生成的最后一个 token 时的 transformer block 最后一层 hidden state 上加上 zi 的投影"

**更精确的描述**：

1. **"每次生成"** ✅ - 在自回归生成的每一步
2. **"最后一个 token"** ✅ - `[:, -1, :]`（正在预测的位置）
3. **"指定层"** ✅ - 默认最后 4 层（Layer 24-27）
4. **"hidden state"** ✅ - 该层的输出表征
5. **"加上 z 的投影"** ✅ - `z_proj [1, 4096]`

### 🎯 影响范围

```python
影响时间：每次生成都注入
影响空间：只影响最后一个 token
影响层级：只影响指定层之后的层（默认最后 4 层）
```

### 🔑 关键区别

| 特性       | INPUT      | PSA           | SOFTMAX    |
| ---------- | ---------- | ------------- | ---------- |
| 注入层     | 第一层前   | 倒数第 4-1 层 | lm_head 后 |
| 影响 token | 所有 token | 只有最后一个  | 所有 token |
| 影响层数   | 所有后续层 | 最后几层      | 不经过层   |
|            |            |               |            |

---

