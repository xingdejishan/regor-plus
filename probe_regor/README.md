# ProbeRegor

ProbeRegor 是严格 pairwise、纯点云几何的端到端 REGOR 变体。每个样本只接收当前 source/target XYZ、冻结 PREDATOR 特征及其确定性几何导出量，输出唯一 4×4 刚体变换。

该目录是完整 ProbeRegor 实现，不是完整 ActiveRegor 或 Diagnosis-guided REGOR-S2；不会修改仓库现有 REGOR-S2 代码路径。

`baseline/` 保存本次实验实际调用的三个原生 REGOR 文件快照。目标仓库当前分支已修改同名接口，因此 ProbeRegor runner 会优先加载该快照，避免把后续 REGOR-S2 变化混入已报告结果。

## 方法模块

- `LineProbeBank`：从固定 Sobol 候选中选择穿过两片点云包围盒的人工直线，并构造近似交点 witness。
- `ProbeRegor._posterior_update`：以 line witness 更新 correspondence region 的 Beta posterior。
- `ProbeRegor._actions`：最大化 posterior 加权的 6DoF log-determinant 边际信息，并用错误增长风险修正 utility。
- `ProbeRegor._anisotropic_indices`：以局部协方差 Mahalanobis 距离构造各向异性再生邻域。
- `ProbeRegor._merge`：跨轮保留 correspondence 与 posterior 状态并去重。
- `ProbeRegor._joint_pose`：联合 line witness 与 regenerated correspondence 求解加权刚体变换。
- `ProbeRegor.run`：执行三轮 regeneration、增长率停止和官方 point refinement，输出唯一位姿。
- `baseline/`：本次实验所用 `Matcher_plus`、`Regenerator` 与 `Estimator` 的冻结快照。

## 近似与失败处理

- line intersection 是基于局部三点柱体和软聚合的几何近似，不是解析曲面交点。
- `local_matching` 发生 `RuntimeError` 或 `IndexError` 时，该轮新增 correspondence 置空并触发停止；不会回退到另一位姿专家。
- 加权位姿非有限时保留当前轮位姿，最终仍执行固定 20 次 point refinement。
- 这些处理属于 ProbeRegor 的冻结实现，不能用于声称完整 ActiveRegor 已被验证。

## 冻结参数

| 参数 | 作用 | 默认值 | 本次是否消融 | 位置 |
|---|---|---:|---|---|
| `line_count` | 每轮最大人工直线数 | 512 | 否 | `ProbeRegorConfig` |
| `line_candidates` | Sobol 候选直线数 | 4096 | 否 | `ProbeRegorConfig` |
| `nu0` | line gap 概率尺度系数 | 0.5 | 否 | `ProbeRegorConfig` |
| `action_count` | 每轮最大主动 seed 数 | 100 | 否 | `ProbeRegorConfig` |
| `local_covariance_k` | 局部协方差邻居数 | 16 | 否 | `ProbeRegorConfig` |
| `round_knn` | 三轮各向异性邻域大小 | `(100, 50, 20)` | 否 | `ProbeRegorConfig` |
| `inlier_scale` | correspondence Cauchy residual 尺度 | 0.1 m | 否 | `ProbeRegorConfig` |
| `covariance_floor_ratio` | 局部协方差特征值下限比例 | 0.05 | 否 | `ProbeRegorConfig` |
| `line_chunk` | line witness 计算分块 | 64 | 否 | `ProbeRegorConfig` |
| `point_chunk` | 点距离计算分块 | 1024 | 否 | `ProbeRegorConfig` |
| Beta prior | 每个初始 correspondence 的先验 | `Beta(1,1)` | 否 | `ProbeRegor.run` |
| information floor | 6DoF 信息矩阵初始对角值 | `1e-3` | 否 | `ProbeRegor._actions` |
| minimum actions | 继续 regeneration 的最少动作数 | 3 | 否 | `ProbeRegor.run` |
| growth stop | 增长停止条件 | `R+ <= 1` 或 `R- >= 1` | 否 | `ProbeRegor.run` |
| final refinement | 最终 point refinement 次数 | 20 | 否 | `ProbeRegor.run` |

本次结果来自上述单一冻结配置，没有进行参数扫描。

## 输入隔离

原始 PREDATOR `.pth` 同时含位姿标签，因此 online runner 不直接打开它。先生成只包含白名单字段的 compact artifact：

```bash
python probe_regor/create_pair_seed_csv.py --output artifacts/pair_seeds.csv

python probe_regor/prepare_online_artifacts.py \
  --raw-data-root external/OverlapPredator/snapshot/indoor/3DLoMatch \
  --source-pairs-path artifacts/pair_seeds.csv \
  --output-dir artifacts/probe_regor_head \
  --start-index 0 --limit 833 --overwrite

python probe_regor/prepare_online_artifacts.py \
  --raw-data-root external/OverlapPredator/snapshot/indoor/3DLoMatch \
  --source-pairs-path artifacts/pair_seeds.csv \
  --output-dir artifacts/probe_regor_tail \
  --start-index 833 --limit 948 --overwrite
```

compact artifact 严格排除 `rot`、`trans`、GT、scene metadata、overlap 和 saliency；online 阶段仅恢复已经冻结的采样点、特征与 NumPy RNG 状态。

## 标签盲随机247

```bash
python probe_regor/create_label_blind_random_split.py \
  --output outputs/probe_regor/random247.json

python probe_regor/run_probe_regor_natural1781.py \
  --head-root artifacts/probe_regor_head \
  --tail-root artifacts/probe_regor_tail \
  --output-dir outputs/probe_regor/random247_online \
  --index-file outputs/probe_regor/random247.json

python probe_regor/analyze_probe_regor_natural1781.py \
  --online-dir outputs/probe_regor/random247_online \
  --output-dir outputs/probe_regor/random247_analysis \
  --index-file outputs/probe_regor/random247.json
```

## 完整1781

```bash
python probe_regor/run_probe_regor_chunked_online.py \
  --head-root artifacts/probe_regor_head \
  --tail-root artifacts/probe_regor_tail \
  --output-dir outputs/probe_regor/full1781_online \
  --pair-count 1781 --chunk-size 250

python probe_regor/analyze_probe_regor_natural1781.py \
  --online-dir outputs/probe_regor/full1781_online \
  --output-dir outputs/probe_regor/full1781_analysis
```

## 测试

```bash
PYTHONPATH=probe_regor python probe_regor/test_probe_regor.py
python -m compileall -q probe_regor
```

## 结果

- [RESULTS.md](RESULTS.md)
- `results/random247_analysis_summary.json`
- `results/full1781_analysis_summary.json`
- `results/random247_online_summary.json`
- `results/full1781_online_summary.json`
- `results/label_blind_random247_seed20260824.json`
