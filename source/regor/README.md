# Regor

复现论文 **Progressive Correspondence Regenerator for Robust 3D Registration** 的工程目录。

官方仓库 `GuiyuZhao/Regor` 当前只发布了核心算法文件，缺少数据读取、评估、benchmark 配置等运行依赖。本目录已补齐这些基础模块，并修复了作者本机绝对路径和新版 NumPy 兼容问题。

## 环境

当前机器已验证：

```bash
python --version
python -m compileall -q .
```

依赖可用 `requirements.txt` 安装：

```bash
pip install -r requirements.txt
```

如果需要重新安装 CUDA 版 PyTorch，请按你的 CUDA 版本使用 PyTorch 官方安装命令。

## 数据目录

3DMatch/3DLoMatch 的点云与特征文件不在代码仓库内，需要单独下载数据。当前目录支持两种方式：

1. 下载已经带 `cloud_bin_*_fpfh.npz` / `cloud_bin_*_fcgf.npz` 的预处理包，直接放成下面结构。
2. 下载原始 `3dmatch/test/*/fragments/*.ply` 后，运行本项目的 FPFH 预处理脚本：

```bash
python scripts/prepare_3dmatch_fpfh.py
```

目标结构：

```text
data/
  3DMatch/
    fragments/
      7-scenes-redkitchen/
        cloud_bin_0_fcgf.npz
        cloud_bin_0_fpfh.npz
        ...
      ...
    gt_result/
      7-scenes-redkitchen-evaluation/
        gt.log
        gt.info
      ...
```

`benchmarks/3DMatch` 来自 PREDATOR benchmark 配置，`benchmarks/3DLoMatch` 和 `3DLoMatch.pkl` 来自 SC2-PCR++，已放在项目内。

## 运行

3DMatch + FCGF：

```bash
python test_3DMatch.py --config_path config_json/config_3DMatch_FCGF.json
```

3DMatch + FPFH：

```bash
python test_3DMatch.py --config_path config_json/config_3DMatch_FPFH.json
```

3DLoMatch + FCGF：

```bash
python test_3DLoMatch.py --config_path config_json/config_3DLoMatch_FCGF.json
```

3DLoMatch + FPFH：

```bash
python test_3DLoMatch.py --config_path config_json/config_3DLoMatch_FPFH.json
```

3DLoMatch + Predator：

```bash
python test_3DLoMatch.py --config_path config_json/config_3DLoMatch_Predator.json
```

当前 Predator 特征已由 `external/OverlapPredator` 导出到：

```text
external/OverlapPredator/snapshot/indoor/3DLoMatch/
```

共 `0.pth` 到 `1780.pth`，Regor 的 `config_json/config_3DLoMatch_Predator.json` 已指向该目录。

日志会写入 `logs/`。

## 已补齐内容

- `common.py`、`dataset.py`、`utils/`
- `evaluate_metric.py`
- `benchmark_utils_predator.py`
- `config.py`、`config_json/`
- `benchmarks/3DMatch`、`benchmarks/3DLoMatch`
- `3DLoMatch.pkl`
- `visualization.py`

## 主要来源

- Regor: https://github.com/GuiyuZhao/Regor
- SC2-PCR++: https://github.com/ZhiChen902/SC2-PCR-plusplus
- PREDATOR benchmark configs: https://github.com/prs-eth/OverlapPredator
