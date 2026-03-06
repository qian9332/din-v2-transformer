# DIN-V2: Enhanced Deep Interest Network

基于淘宝真实用户行为数据集（318万条），对DIN模型进行升级迭代。

## 项目概述

DIN-V2 对标准DIN模型进行两大核心升级：
1. **行为类型Embedding**：为点击、收藏、加购、购买四种行为训练独立Embedding，与商品Embedding相加，使同一商品在不同行为语境下具有不同表征
2. **Transformer Encoder**：在Target Attention前增加Transformer Encoder（因果Mask + 可学习位置编码），建模行为序列内部依赖关系

## 🔍 重要：AUC≈1.0 问题诊断与修复

### 问题
第一版训练中Test AUC达到0.9998，明显不合理。

### 根因
经代码审计发现3个BUG：

| BUG | 严重程度 | 问题 | 影响 |
|-----|---------|------|------|
| **BUG-1** | 🔴致命 | 负样本`target_category`硬编码为0 | 信息泄露，AUC→1.0 |
| **BUG-2** | 🟡严重 | 均匀随机负采样 | popularity偏差，AUC虚高至0.92 |
| **BUG-3** | 🟡中等 | 5K样本训练236K参数 | 47:1参数比，严重过拟合 |

### 修复效果

| 模型 | 修复前 Test AUC | 修复后 Test AUC |
|------|----------------|----------------|
| DIN-V2 | 0.9998 ❌ | **0.8565** ✅ |
| DIN-V1 | ~0.9998 ❌ | **0.9412** ✅ |

📋 **完整诊断报告**: [docs/bugfix_report.md](docs/bugfix_report.md)

## 📊 数据集

**淘宝用户行为数据集**（来源: [GitCode](https://gitcode.com/Open-source-documentation-tutorial/6af95)）

| 指标 | 值 |
|------|-----|
| 总记录 | 3,182,261 条 |
| 用户数 | 3,156,958 |
| 商品数 | 945,601 |
| 类目数 | 7,227 |
| 行为类型 | 浏览94.23% / 收藏1.91% / 加购2.88% / 购买0.98% |
| 时间范围 | 2014-11-18 ~ 2014-12-22 |

### 数据特殊性
- **99.2%用户仅1条记录**（user_id更像行级ID，非真实用户标识）
- 需通过 **session-based方式**（按category+date分组）构建行为序列
- 最终生成: 训练573K + 验证90K + 测试98K 样本

## 🏗️ 模型架构

```
输入: 用户行为序列 [item_id, behavior_type, category_id] × N
  ↓
Item Embedding + Behavior Type Embedding (4种行为独立Embedding)  [V2独有]
  ↓ + Learnable Position Embedding
Transformer Encoder (1层2头, Causal Mask)  [V2独有]
  ↓
Target Attention (与候选商品交互)
  ↓
MLP [dim*3 → 128 → 64 → 1] → sigmoid → CTR预测
```

## 🏋️ 训练结果

### DIN-V2 (修复后)
```
Epoch | Train Loss | Train AUC | Val AUC
  1   |   0.5308   |  0.7860   |  0.8362
  2   |   0.4494   |  0.8558   |  0.8382  ← 最佳泛化
  3   |   0.3521   |  0.9189   |  0.8301  ← 开始过拟合
  4   |   0.2140   |  0.9715   |  0.8447
  5   |   0.1115   |  0.9923   |  0.8348
Test AUC = 0.8565
```

### DIN-V1 (基线对比)
```
Epoch | Train Loss | Train AUC | Val AUC
  1   |   0.5420   |  0.7902   |  0.8484
  2   |   0.3828   |  0.9091   |  0.9022  ← 最佳泛化
  3   |   0.2209   |  0.9700   |  0.8994
  4   |   0.1190   |  0.9912   |  0.8956
  5   |   0.0683   |  0.9970   |  0.8916
Test AUC = 0.9412
```

## 📁 项目结构

```
din-v2-transformer/
├── src/
│   ├── model.py              # DIN-V2 核心模型
│   ├── model_v1.py           # DIN-V1 基线模型
│   ├── preprocess.py         # ❌ 旧版预处理 (含BUG)
│   ├── preprocess_fixed.py   # ✅ 修复版预处理
│   ├── train.py              # 旧版训练脚本
│   ├── train_fixed.py        # ✅ 修复版训练脚本
│   ├── dataset.py            # 旧版数据集类
│   └── utils.py              # 工具函数
├── data/
│   ├── taobao_raw.txt        # 淘宝原始数据 (318万条)
│   ├── analyze_taobao.py     # 数据深度分析脚本
│   └── data_analysis_report.json
├── docs/
│   └── bugfix_report.md      # 🔍 完整问题诊断报告
├── logs/
│   ├── training_v2_fixed_full.log
│   ├── training_v1_fixed_full.log
│   └── bugfix_results.json
├── checkpoints/
├── README.md
└── requirements.txt
```

## 🚀 快速开始

```bash
# 克隆
git clone https://github.com/qian9332/din-v2-transformer.git
cd din-v2-transformer
pip install -r requirements.txt

# 数据分析
python data/analyze_taobao.py

# 预处理 (修复版)
python src/preprocess_fixed.py

# 训练V2
python src/train_fixed.py --model v2 --epochs 5 --batch_size 256

# 训练V1 (对比)
python src/train_fixed.py --model v1 --epochs 5 --batch_size 256

# GPU训练 (全部数据)
python src/train_fixed.py --model v2 --epochs 10 --batch_size 1024 --embed_dim 64 --hash_buckets 100000
```

## 关键教训

1. **负样本必须赋真实特征** — 任何特征字段的差异都可能造成信息泄露
2. **AUC > 0.95 要警惕** — 真实推荐场景中极少见，大概率存在问题
3. **负采样策略至关重要** — 均匀随机采样会引入popularity偏差，应使用freq^0.75加权
4. **参数/样本比要合理** — 建议 < 1:10
5. **数据特性决定模型选择** — session-based短序列不一定需要Transformer
