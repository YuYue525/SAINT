# SAINT: Self-Attention and Intersample Attention Transformer

基于 SAINT (Self-Attention and Intersample Attention Transformer) 模型的表格型数据自监督学习框架，用于生成高质量的样本 embedding，支持后续聚类分析和黑产挖掘。

## 核心特性

### 1. 列间自注意力 (Intrasample / Feature-wise Attention)
在单条样本内部，让不同特征之间通过注意力交互，捕捉特征间高阶关系。

### 2. 行间注意力 (Intersample / Row-wise Attention)
在小批量内，不同样本之间也做注意力，使某个样本可以"参考"相近样本的信息。

### 3. 改进的连续特征嵌入
通过可学习的参数，更好地将数值特征与离散特征统一在同一嵌入空间内。

### 4. 对比自监督预训练
- 对表格样本进行扰动（随机掩码、值替换）构造"同一行的不同视图"
- 使用对比损失让同一行的不同视图在嵌入空间中靠近，不同样本远离

## 功能特性

- 支持三种特征类型：类别型、数值型、布尔型
- 自监督对比学习
- Transformer-based encoder
- 样本 embedding 生成
- 支持 Parquet 格式输入

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备数据

数据以 Parquet 格式存储，可以读取为 pandas DataFrame。支持三种特征类型：

- **类别型特征**：string 类型（如设备型号）
- **数值型特征**：float 类型（如设备开机时间）
- **布尔型特征**：string 类型，值为 'true'/'false'/'null'（如设备是否越狱）

### 2. 配置特征

在 `config.yaml` 中配置特征类型：

```yaml
categorical_features:
  - device_model
  - os_version
numerical_features:
  - boot_time
  - battery_level
boolean_features:
  - is_jailbroken
```

### 3. 训练模型

```bash
python src/train.py --data_path data/your_data.parquet --config config.yaml
```

### 4. 生成 Embeddings

```bash
python src/inference.py --data_path data/your_data.parquet --model_path models/saint_model.pth --output_path data/embeddings.parquet
```

## 项目结构

```
SAINT/
├── data/              # 数据目录
├── models/            # 模型保存目录
├── src/
│   ├── data_loader.py # 数据加载和预处理
│   ├── saint_model.py # SAINT 模型实现
│   ├── train.py       # 训练脚本
│   └── inference.py   # 推理脚本
├── config.yaml        # 配置文件
├── requirements.txt   # 依赖包
└── README.md
```

## 引用

SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-training
