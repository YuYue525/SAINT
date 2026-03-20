# SAINT 快速开始指南 🦐

## 项目概览

这是一个基于 **SAINT (Self-Supervised Attention-based Tabular Transformer)** 的表格型数据自监督学习框架，专为黑产挖掘设计！

## 快速开始

### 1. 安装依赖

```bash
cd SAINT
pip install -r requirements.txt
```

### 2. 生成示例数据

```bash
python src/generate_sample_data.py --output_path data/sample_data.parquet --n_samples 10000
```

### 3. 训练模型

```bash
python src/train.py --data_path data/sample_data.parquet --config config.yaml --output_dir models
```

### 4. 生成 Embeddings

```bash
python src/inference.py \
  --data_path data/sample_data.parquet \
  --model_path models/saint_model_best.pth \
  --output_path data/embeddings.parquet
```

### 5. 聚类分析（挖掘黑产）

```bash
python src/cluster_analysis.py \
  --embeddings_path data/embeddings.parquet \
  --original_data_path data/sample_data.parquet \
  --output_dir data \
  --method kmeans \
  --n_clusters 8 \
  --visualize
```

## 项目结构

```
SAINT/
├── src/
│   ├── saint_model.py          # SAINT 模型核心实现
│   ├── data_loader.py          # 数据加载和预处理
│   ├── train.py                # 训练脚本
│   ├── inference.py            # 推理脚本（生成embeddings）
│   ├── cluster_analysis.py     # 聚类分析脚本
│   └── generate_sample_data.py # 生成示例数据
├── data/                       # 数据目录
├── models/                     # 模型保存目录
├── config.yaml                 # 配置文件
└── requirements.txt            # 依赖包
```

## 配置说明

在 `config.yaml` 中配置你的特征：

```yaml
features:
  categorical_features:    # 类别型特征 (string)
    - device_model
    - os_version
  numerical_features:      # 数值型特征 (float)
    - boot_time
    - battery_level
  boolean_features:        # 布尔型特征 ('true'/'false'/'null')
    - is_jailbroken
    - has_vpn
```

## 支持的特征类型

1. **类别型特征**：string 类型，如设备型号、操作系统版本等
2. **数值型特征**：float 类型，如开机时间、电池电量等
3. **布尔型特征**：string 类型，值为 `'true'`/`'false'`/`'null'`

## 黑产挖掘流程

1. **自监督预训练**：使用 SAINT 对大量无标签数据进行对比学习
2. **生成 Embeddings**：用训练好的模型将表格数据转换为高维向量
3. **聚类分析**：通过 K-Means 或 DBSCAN 发现异常聚类
4. **人工审核**：对小聚类进行重点分析，识别黑产样本

## GitHub 仓库

https://github.com/YuYue525/SAINT

## 问题反馈

有问题欢迎提 Issue！🦐
