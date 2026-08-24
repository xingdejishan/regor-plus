# ProbeRegor实验指标

实验日期：2026-08-24

数据集：3DLoMatch

任务：严格 pairwise registration

成功阈值：`RRE < 15°` 且 `RTE < 30 cm`

## 标签盲随机247

抽样：从自然1781索引中使用 NumPy `default_rng(seed=20260824)` 均匀无放回抽取247对。

split SHA256：`9c036629f6ec6a8b0e01935b44343440f12af08b6e0db357d11018f462210832`

### 完整性

| 指标 | 数值 |
|---|---:|
| Pair数 | 247 |
| Online error | 0 |
| Baseline finite pose | 247 |
| ProbeRegor finite pose | 247 |
| Online运行时间 | 535.968 s |
| 平均baseline时间/pair | 0.574 s |
| 平均ProbeRegor时间/pair | 1.496 s |
| 平均完成轮数 | 2.866 |

### 阈值指标

| 指标 | REGOR baseline | ProbeRegor | 变化 |
|---|---:|---:|---:|
| 成功数 | 183/247 | 187/247 | +4 |
| 成功率 | 74.089% | 75.709% | +1.619 pp |
| Median RRE | 3.3648° | 3.3286° | -0.0362° |
| Median RTE | 9.6317 cm | 8.8392 cm | -0.7924 cm |

| 配对统计 | 数值 |
|---|---:|
| Rescue | 6 |
| Damage | 2 |
| Net | +4 |
| Two-sided exact McNemar p | 0.2890625 |
| Rescue方向one-sided p | 0.14453125 |
| Paired bootstrap 95% CI | [-0.004049, 0.040486] |

### 分场景阈值成功数

| 场景 | Pair数 | Baseline | ProbeRegor | 变化 |
|---|---:|---:|---:|---:|
| 7-scenes-redkitchen | 68 | 54 | 53 | -1 |
| home_at_scan1 | 39 | 27 | 28 | +1 |
| home_md_scan9 | 32 | 20 | 21 | +1 |
| hotel_uc_scan3 | 32 | 29 | 30 | +1 |
| hotel_umd_hotel1 | 27 | 21 | 22 | +1 |
| hotel_umd_hotel3 | 7 | 3 | 3 | 0 |
| mit_76_studyroom | 36 | 25 | 26 | +1 |
| mit_lab_hj | 6 | 4 | 4 | 0 |

## 完整自然1781

### 完整性

| 指标 | 数值 |
|---|---:|
| Pair数 | 1781 |
| CUDA shard数 | 8 |
| Online error | 0 |
| Baseline finite pose | 1781 |
| ProbeRegor finite pose | 1781 |
| 平均baseline时间/pair | 0.584 s |
| 平均ProbeRegor时间/pair | 1.518 s |
| 平均完成轮数 | 2.851 |
| 合并pose文件SHA256 | `3dd92942be909a5a1201d749134bbf8076001da84dce68b2448f0a6c7afdfafa` |

### 阈值指标

| 指标 | REGOR baseline | ProbeRegor | 变化 |
|---|---:|---:|---:|
| 成功数 | 1283/1781 | 1293/1781 | +10 |
| 成功率 | 72.038% | 72.600% | +0.561 pp |
| Median RRE | 3.6248° | 3.4512° | -0.1737° |
| Median RTE | 11.1147 cm | 10.5644 cm | -0.5503 cm |

| 配对统计 | 数值 |
|---|---:|
| Rescue | 26 |
| Damage | 16 |
| Net | +10 |
| Two-sided exact McNemar p | 0.1641494 |
| Rescue方向one-sided p | 0.0820747 |
| Paired bootstrap 95% CI | [-0.001684, 0.012914] |

### 分场景阈值成功数

| 场景 | Pair数 | Baseline | ProbeRegor | 变化 |
|---|---:|---:|---:|---:|
| 7-scenes-redkitchen | 525 | 419 | 417 | -2 |
| home_at_scan1 | 289 | 198 | 200 | +2 |
| home_md_scan9 | 230 | 151 | 153 | +2 |
| hotel_uc_scan3 | 218 | 188 | 190 | +2 |
| hotel_umd_hotel1 | 158 | 112 | 114 | +2 |
| hotel_umd_hotel3 | 49 | 36 | 36 | 0 |
| mit_76_studyroom | 240 | 141 | 145 | +4 |
| mit_lab_hj | 72 | 38 | 38 | 0 |

### Corrected Redwood / 3DLoMatch covariance protocol

| 汇总指标 | REGOR baseline | ProbeRegor | 变化 |
|---|---:|---:|---:|
| Mean-scene recall | 67.294% | 68.041% | +0.747 pp |
| Weighted recall | 69.930% | 70.973% | +1.043 pp |

| 场景 | 有效非相邻pair | Baseline成功 | Probe成功 | Baseline recall | Probe recall |
|---|---:|---:|---:|---:|---:|
| 7-scenes-redkitchen | 524 | 407 | 408 | 77.672% | 77.863% |
| home_at_scan1 | 283 | 190 | 195 | 67.138% | 68.905% |
| home_md_scan9 | 222 | 142 | 147 | 63.964% | 66.216% |
| hotel_uc_scan3 | 210 | 177 | 180 | 84.286% | 85.714% |
| hotel_umd_hotel1 | 138 | 94 | 96 | 68.116% | 69.565% |
| hotel_umd_hotel3 | 42 | 29 | 28 | 69.048% | 66.667% |
| mit_76_studyroom | 237 | 131 | 134 | 55.274% | 56.540% |
| mit_lab_hj | 70 | 37 | 37 | 52.857% | 52.857% |

## 文件哈希

| 文件 | SHA256 |
|---|---|
| 测试时的`probe_regor.py` | `dfdea8550a72c2e5fd59f803661f71d77aacdbcd7a2071655ce8bd15679cd461` |
| `random247_analysis_summary.json` | `7b9173ad1a91f5d7a3bbac81e49ffdd9ae81a293554693088b429c087c02fa86` |
| `full1781_analysis_summary.json` | `dfc244d0c71c7732b578a9cc6f80d639cf31f0ccff6753320338f789362433f3` |
