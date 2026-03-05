# DIN-V2: Enhanced Deep Interest Network

> Behavior-Type-Aware DIN with Transformer Encoder for CTR Prediction

## 项目概述

本项目是对阿里DIN（Deep Interest Network）模型的V2升级，针对DIN不建模行为间依赖关系且不区分行为类型的局限进行改进。

### 核心升级

| 升级点 | DIN V1 | DIN V2 |
|--------|--------|--------|
| 行为表征 | 仅Item Embedding | Item Embedding + **Behavior Type Embedding**（4种行为独立空间） |
| 序列建模 | 无序列依赖 | **2层4头Transformer Encoder**（因果Mask + 可学习位置编码） |
| 注意力 | 基础Target Attention | 增强版Target Attention（query*key, query-key交叉特征） |

### 模型架构

```
用户行为序列 → Item Embedding + Behavior Type Embedding (pv/fav/cart/buy)
                              ↓  + Learnable Position Embedding
                   Transformer Encoder (2层, 4头, Causal Mask)
                              ↓  F.scaled_dot_product_attention 优化
                   Target Attention (与候选商品交互)
                              ↓
                   MLP [96→128→64→1] → CTR预测
```

## 数据

### 数据生成
基于阿里巴巴UserBehavior数据集Schema，生成了包含**真实行为依赖模式**的数据：

- **1,829,350条**行为记录
- **50,000**用户 × **100,000**商品 × **5,000**类目
- 4种行为类型：pv(浏览)、fav(收藏)、cart(加购)、buy(购买)
- **Markov链行为转移**：模拟真实电商漏斗

### 行为转移矩阵

| From\To | pv | fav | cart | buy |
|---------|-----|-----|------|-----|
| **pv** | 77.9% | 9.0% | 8.0% | 5.0% |
| **fav** | 50.2% | 14.9% | 22.0% | 13.0% |
| **cart** | 34.9% | 10.1% | 25.1% | 30.0% |
| **buy** | 69.9% | 12.0% | 12.0% | 6.0% |

## 训练结果

### 训练曲线

| Epoch | Train Loss | Train AUC | Val AUC | Val Acc |
|-------|-----------|-----------|---------|---------|
| 1 | 0.7508 | 0.5046 | 0.5277 | 0.5165 |
| 2 | 0.6883 | 0.5905 | 0.6678 | 0.5240 |
| 3 | 0.6150 | 0.7324 | 0.8200 | 0.7105 |
| 4 | 0.4788 | 0.8879 | 0.9405 | 0.8860 |
| 5 | 0.3119 | 0.9705 | 0.9574 | 0.8940 |
| **Test** | - | - | **0.9648** | **0.8975** |

### 关键指标
- **Test AUC: 0.9648**
- **Best Val AUC: 0.9574**
- **参数量: 235,910**
- **训练时间: 271秒 (CPU)**

## 项目结构

```
din-v2-transformer/
├── src/
│   ├── model.py          # DIN-V2 模型（PyTorch Transformer实现）
│   ├── model_sdpa.py     # DIN-V2 优化版（F.scaled_dot_product_attention）
│   ├── model_v1.py       # DIN-V1 基线模型
│   ├── model_fast.py     # 手动Transformer实现
│   ├── dataset.py        # 数据预处理 & PyTorch Dataset
│   ├── train.py          # 完整训练脚本（支持GPU）
│   ├── run_epoch.py      # 分epoch训练（CPU友好）
│   ├── preprocess.py     # 数据预处理
│   └── utils.py          # 工具函数
├── data/
│   ├── gen_large.py      # 大规模数据生成（Markov行为依赖）
│   ├── analyze_data.py   # 数据分析报告生成
│   └── UserBehavior.csv  # 生成的行为数据（1.83M条）
├── logs/
│   ├── full_training.log # 完整5-epoch训练日志
│   ├── full_results.json # 训练结果JSON
│   └── data_analysis_report.md  # 数据分析报告
├── checkpoints/          # 模型检查点
├── requirements.txt
└── README.md
```

## 快速开始

### 环境

```bash
pip install torch numpy pandas scikit-learn tensorboard
```

### 生成数据
```bash
python data/gen_large.py
```

### 数据分析
```bash
python data/analyze_data.py
```

### 训练（GPU推荐）
```bash
# GPU训练（完整数据）
python src/train.py --model v2 --epochs 5 --batch_size 1024 --device cuda

# CPU分epoch训练
python src/run_epoch.py  # 运行5次完成5个epoch
```

## 技术细节

### Behavior Type Embedding
为每种行为类型（pv=0, fav=1, cart=2, buy=3）训练独立的Embedding向量，与商品Item Embedding相加：
```python
composite = item_embedding(item_id) + behavior_embedding(behavior_type)
```
使同一商品在不同行为语境下具有不同的表征。

### Transformer Encoder
- 2层4头Self-Attention
- **因果Mask (Causal Mask)**：确保每个位置只能关注之前的行为
- **可学习位置编码 (Learnable Position Encoding)**：适应不同长度的行为序列
- 使用`F.scaled_dot_product_attention`优化CPU/GPU性能

### Target Attention
增强版Target Attention，融合多种交互特征：
```python
attention_input = concat([query, key, query-key, query*key])
```
