# DHPA: 城市级停车位可用性预测模型

本项目实现了面向城市停车场网络的停车位可用性预测模型 DHPA（Decomposition-Enhanced Hybrid Spatiotemporal Modeling for City-Scale Parking Availability Forecasting）。模型以历史停车可用性序列和时间、节点相关辅助特征为输入，预测未来多个时间步内各停车场节点的可用车位变化情况。当前代码主要用于在 SINPA 数据集上进行训练、验证与测试。

> 说明：由于数据集体积较大，当前提交版本不包含完整数据文件。运行前需要手动补齐 `data/SINPA` 下的 `train.npz`、`val.npz`、`test.npz`，以及完整复现实验可能需要的 `aux_data/lots_location.csv`。

## 1. 项目结构

```text
.
├── experiments
│   └── DeepPA
│       └── main.py                 # 训练与测试入口
├── src
│   ├── base
│   │   ├── model.py                # 模型基类
│   │   ├── sampler.py              # 图采样相关工具
│   │   └── trainer.py              # 通用训练、验证、测试流程
│   ├── layers
│   │   └── embedding.py            # 时间、星期、利用率、地理区域等离散特征嵌入
│   ├── models
│   │   └── DeepPA.py               # DHPA/DeepPA 主体模型实现
│   ├── trainers
│   │   └── deeppa_trainer.py       # DeepPA 专用训练器
│   └── utils
│       ├── args.py                 # 命令行参数配置
│       ├── helper.py               # 数据加载、设备选择、节点数配置
│       ├── metrics.py              # MAE、RMSE 等评价指标
│       ├── scaler.py               # 标准化与反标准化
│       └── graph_algo.py           # 图相关计算工具
├── requirements.txt                # Python 依赖
├── data                            # 数据目录，当前未包含完整数据
│   └── SINPA
│       ├── train.npz               # 需手动补齐
│       ├── val.npz                 # 需手动补齐
│       └── test.npz                # 需手动补齐
└── aux_data
    └── lots_location.csv           # 需手动补齐；用于完整数据/空间辅助信息
```

## 2. 环境要求

建议使用 Python 3.8 环境运行。`requirements.txt` 中给出的主要依赖如下：

```text
numpy==1.19.2
pandas==1.4.1
scipy==1.8.0
torch==1.10.0
```

推荐创建独立环境：

```bash
conda create -n sinpa38 python=3.8 -y
conda activate sinpa38
```

安装依赖：
```bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -r requirements.txt
```

## 3. 数据准备

当前仓库未包含完整数据。运行前请将数据文件放置为以下结构：

```text
data/
└── SINPA/
    ├── train.npz
    ├── val.npz
    └── test.npz

aux_data/
└── lots_location.csv
```

其中 `train.npz`、`val.npz`、`test.npz` 需要包含以下两个数组：

```text
x: 输入序列，形状通常为 [num_samples, seq_len, num_nodes, input_dim]
y: 预测标签，形状通常为 [num_samples, horizon, num_nodes, output_dim]
```

当前默认配置为：

```text
seq_len    = 12
horizon    = 12
num_nodes  = 1687
input_dim  = 20
output_dim = 1
```

代码会在加载数据时，根据训练集目标通道计算标准化参数，并对训练集、验证集、测试集的输入与标签进行统一标准化；测试与评价阶段会再进行反标准化，最终输出原始尺度下的 MAE 与 RMSE。

注意：请在项目根目录运行命令。代码中数据路径为 `./data/SINPA`，如果在其他目录执行，可能出现找不到数据文件的问题。

## 4. 模型说明

本项目模型主体位于 `src/models/DeepPA.py`。整体结构包括：

1. **时间与节点特征嵌入**：通过 `TimeEmbedding` 对时间片、星期、利用率类别、地理区域等离散特征进行嵌入，并与停车可用性主变量结合。
2. **趋势-季节分解模块**：通过 `--ts_decompose true` 启用，对时间序列特征进行趋势平滑与多周期季节基建模；其中 `--decompose_kernel` 控制趋势平滑窗口，`--season_periods` 控制季节周期集合。
3. **空间学习模块**：通过 `--spatial_op` 选择空间算子，支持 `gco`、`msa`、`afno`、`gco_msa`。其中 `gco_msa` 表示先使用 GCO 捕获全局低频空间模式，再通过 summary tokens 上的多头注意力建模节点组级关系。
4. **时间学习模块**：使用时间 Transformer 建模历史序列依赖，默认启用因果注意力，避免未来信息泄漏。
5. **预测输出层**：使用卷积输出未来 `horizon` 个时间步中每个停车场节点的可用性预测结果。

## 5. 训练方法

在项目根目录下运行：

```bash
python experiments/DeepPA/main.py \
  --mode train --dataset SINPA --gpu 0 \
  --spatial_op gco_msa \
  --ts_decompose true --decompose_kernel 7 --season_periods 12,24,48 \
  --wandb False
```

单行命令版本如下：

```bash
python experiments/DeepPA/main.py --mode train --dataset SINPA --gpu 0 --spatial_op gco_msa --ts_decompose true --decompose_kernel 7 --season_periods 12,24,48 --wandb False
```

训练完成后，程序会自动加载验证集上保存的最优模型，并在测试集上输出整体 MAE、RMSE 以及各预测步的 MAE、RMSE。

## 6. 测试方法

如果已经训练并保存模型，可以使用相同的关键参数进行测试：

```bash
python experiments/DeepPA/main.py \
  --mode test --dataset SINPA --gpu 0 \
  --spatial_op gco_msa \
  --ts_decompose true --decompose_kernel 7 --season_periods 12,24,48 \
  --wandb False
```

需要注意，测试阶段会根据参数组合生成对应的日志目录，并从该目录下读取模型文件。因此测试时应尽量保持与训练时一致的关键参数，例如 `--spatial_op`、`--ts_decompose`、`--decompose_kernel`、`--season_periods`、`--n_hidden`、`--n_blocks`、`--n_heads`、`--batch_size`、`--base_lr`、`--n_exp` 等。

## 7. 常用参数说明

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--mode` | `train` | 运行模式，支持 `train` 或 `test` |
| `--dataset` | `base` | 数据集名称；运行 SINPA 时设置为 `SINPA` |
| `--gpu` | `6` | 使用的 GPU 编号 |
| `--seq_len` | `12` | 输入历史序列长度 |
| `--horizon` | `12` | 预测未来时间步数 |
| `--input_dim` | `20` | 输入特征维度 |
| `--output_dim` | `1` | 输出目标维度 |
| `--batch_size` | `8` | 批大小 |
| `--max_epochs` | `100` | 最大训练轮数 |
| `--patience` | `10` | Early stopping 的等待轮数 |
| `--base_lr` | `1e-3` | 初始学习率 |
| `--lr_step` | `3` | 学习率衰减间隔 |
| `--lr_decay_ratio` | `0.5` | 学习率衰减比例 |
| `--spatial_op` | `gco` | 空间算子，可选 `gco`、`msa`、`afno`、`gco_msa` |
| `--ts_decompose` | `False` | 是否启用趋势-季节分解模块 |
| `--decompose_kernel` | `7` | 趋势平滑窗口大小，代码中会自动调整为奇数 |
| `--season_periods` | `12,24,48` | 季节周期集合，逗号分隔 |
| `--temporal_causal` | `True` | 时间注意力是否采用因果掩码 |
| `--wandb` | `True` | 是否启用 Weights & Biases 记录；无账号或不需要记录时设置为 `False` |
| `--save_preds` | `False` | 是否保存训练集、验证集、测试集预测结果 |


## 8. 指标说明

项目主要使用以下指标评价预测效果：

- **MAE**：Mean Absolute Error，平均绝对误差，越小表示预测值与真实值越接近。
- **RMSE**：Root Mean Squared Error，均方根误差，对较大误差更敏感，越小表示整体预测误差越低。

代码会分别输出所有预测步的平均结果，以及每个预测步上的 MAE、RMSE。当 `horizon=12` 时，还会进一步输出 0-1h、1-2h、2-3h 三个预测区间的评价结果。


## 9. 备注

本仓库当前版本主要用于模型训练与实验复现。由于完整数据文件体积较大，提交版本中删除了以下文件：

```text
data/SINPA/train.npz
data/SINPA/val.npz
data/SINPA/test.npz
aux_data/lots_location.csv
```

因此，代码本身可以正常阅读和配置，但在未补齐数据文件之前无法直接完成训练和测试。
