# DIN-V2: Behavior-Type-Aware Deep Interest Network with Transformer Encoder

## 项目简介

本项目针对原始 DIN（Deep Interest Network）**不建模行为间依赖关系**且**不区分行为类型**的局限，进行了两大核心升级：

### 核心改进

1. **行为类型 Embedding**  
   为每种行为类型（点击/pv、长停留浏览/fav、询盘/cart、加购/buy）训练独立的行为类型 Embedding，与商品 Embedding 相加，使同一商品在不同行为语境下具有不同表征。

2. **Transformer Encoder 建模行为序列依赖**  
   在 Target Attention 前增加 Transformer Encoder（2层、4头、因果 Mask、可学习位置编码），先建模行为序列内部依赖关系，再将增强后的序列表征送入 Target Attention 与候选商品交互。

### 模型架构

```
用户行为序列 → Item Embedding + Behavior Type Embedding + Position Embedding
                              ↓
                   Transformer Encoder (2层, 4头, Causal Mask)
                              ↓
                   Target Attention (与候选商品交互)
                              ↓
                   Concat(用户兴趣向量, 候选商品特征, 用户画像特征)
                              ↓
                         MLP → CTR 预测
```

## 数据集

使用阿里巴巴 **UserBehavior** 数据集（约1亿条记录）：
- **来源**: [阿里云天池](https://tianchi.aliyun.com/dataset/649)
- **时间范围**: 2017-11-25 至 2017-12-03
- **行为类型**: pv(点击)、fav(收藏/长停留)、cart(加购)、buy(购买)
- **规模**: ~100万用户, ~400万商品, ~1亿行为记录

### 行为类型映射

| 原始行为 | 业务含义 | Behavior Type ID |
|---------|---------|-----------------|
| pv | 点击 | 0 |
| fav | 长停留浏览/收藏 | 1 |
| cart | 询盘/加购 | 2 |
| buy | 购买 | 3 |

## 项目结构

```
din-v2-transformer/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖
├── data/
│   └── download_data.py         # 数据下载脚本
├── src/
│   ├── __init__.py
│   ├── dataset.py               # 数据预处理与 Dataset
│   ├── model.py                 # DIN-V2 模型（Transformer + 行为类型Embedding）
│   ├── model_v1.py              # DIN-V1 基线模型
│   ├── train.py                 # 训练脚本
│   └── utils.py                 # 工具函数
├── logs/                        # 训练日志
├── checkpoints/                 # 模型权重
└── train.sh                     # 一键训练脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载并预处理数据

```bash
python data/download_data.py
```

### 3. 训练模型

```bash
# 使用 GPU 训练（推荐）
bash train.sh

# 或手动运行
python src/train.py --model v2 --epochs 3 --batch_size 1024 --device cuda
```

### 4. 查看训练日志

```bash
tensorboard --logdir logs/
```

## 模型对比 (V1 vs V2)

| 特性 | DIN-V1 (基线) | DIN-V2 (本项目) |
|-----|-------------|---------------|
| 行为类型区分 | ❌ 不区分 | ✅ 独立 Embedding |
| 行为间依赖 | ❌ 不建模 | ✅ Transformer Encoder |
| 注意力机制 | Target Attention | Transformer + Target Attention |
| 位置信息 | ❌ 无 | ✅ 可学习位置编码 |
| 因果关系 | ❌ 无 | ✅ Causal Mask |

## 训练参数

| 参数 | 默认值 |
|-----|-------|
| Embedding维度 | 64 |
| Transformer层数 | 2 |
| 注意力头数 | 4 |
| 最大序列长度 | 50 |
| 学习率 | 1e-3 |
| Batch Size | 1024 |
| Epochs | 3 |

## 引用

- Zhou, G., et al. "Deep Interest Network for Click-Through Rate Prediction." KDD 2018.
- Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
- UserBehavior Dataset: https://tianchi.aliyun.com/dataset/649

## License

MIT License
