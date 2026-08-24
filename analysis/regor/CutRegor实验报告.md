# CutRegor Frontier Separability Test 实验报告

日期：2026-08-24
轨道：严格 pairwise、纯几何、无新增训练
性质：GT-after-online 核心机制诊断，不是端到端 RR

## 结论

CutRegor 的 symmetry-frontier bridge 核心假设未通过预注册门槛，停止该方法族，不实施 branch-and-contract 完整后端。

## 输入与隔离

- 点云与冻结特征：`output/regor/2026-08-15_Hard247OnlineArtifacts/`；
- 候选模式：G17 的 GT-free 当前-pair SC2 模式，物理脱敏到 `output/regor/2026-08-24_CutRegorG17ModePool/`；
- 脱敏阶段只访问 `generated_trans` 和 G17 模式索引、成员、collision/Q3/Borda 量，不访问 `gt_trans` 或成功标签；输出 manifest 明确 `gt_fields_present_in_output=false`；
- 独立 analyzer 才读取 GT，用于离线选择 GT-near member 和不同的最高内部一致性错误 mode；GT 不进入模式出生、partial symmetry、frontier、anchor 或 bridge-domain 计算规则。

hard247 共 247 对；201 对同时具备可用正、负模式并进入诊断，46 对不可观测，0 error。G17 的 201/247 是候选成员容量，不是 CutRegor 最终结果。

## 冻结实现

- 384 个 FPS patch center；尺度 `4δ/8δ/16δ`；
- target 点到平面、法向、协方差谱和 patch 邻接保持共同定义 partial symmetry；
- 每种策略最多 32 个 query；
- 四锚点由模式内 10 cm residual correspondence 产生，并直接优化四面体矩阵的最小奇异值；
- 薄球壳容差 `2.5δ`，预测搜索半径 `8δ`，每个 domain 最多 8 个 target 点；
- 不扫描阈值、尺度、query 数、anchor 数或 hard247 样本规则。

## 正式指标

| Query | query 数 | positive survival | negative survival | separation rate |
|---|---:|---:|---:|---:|
| random | 6432 | 23.07% | 24.67% | 12.61% |
| high-curvature | 6432 | 21.24% | 22.20% | 12.73% |
| high-disagreement | 6432 | 6.87% | 5.67% | 5.94% |
| symmetry-frontier | 3705 | 75.28% | 75.36% | 16.60% |
| single-scale frontier | 3720 | 75.03% | 75.05% | 16.72% |
| 单点局部域 | 3705 | 98.25% | 95.09% | 3.54% |

预注册判定：

- frontier 相对 random：`+3.99` 个百分点，要求 `≥15`，失败；
- symmetry-frontier 正模式 survival：`75.28%`，要求 `>90%`，失败；
- 多尺度 `16.60%`，单尺度 `16.72%`，失败；
- 四锚点 `16.60%`，单点 `3.54%`，通过。

四项只通过一项，`continue_to_branch_and_contract=false`。

## 判断、原因、经验

判断：部分自对称边界不是当前 hard247 上足够强的新判别观测。四锚点球壳交集有排错能力，但 symmetry frontier 没有把这种能力集中到明显优于随机的位置。

原因：frontier query 下正确和错误模式的 bridge domain 大多同时非空；错误模式 negative survival 为 75.36%，正确模式 survival 也只有 75.28%。因此约束既不排他，也不能作为安全的 pose-cell contraction 前端。

经验：停止 frontier 阈值、patch scale、multiscale vote、query budget、anchor tolerance 和 branch-and-contract 参数扫描。AC-3、位姿区间分裂或 maximum rigid patch cover 只能传播已有排他约束，不能创造本测试缺失的判真信息。

## 可复核文件

- 正式汇总：`output/regor/2026-08-24_CutRegorFrontier_G17_Hard247_v2/frontier_summary.json`；
- 逐对结果：`output/regor/2026-08-24_CutRegorFrontier_G17_Hard247_v2/frontier_pairs.csv`；
- 候选脱敏 manifest：`output/regor/2026-08-24_CutRegorG17ModePool/manifest.json`；
- 实现：`code/regor/cut_regor/`；
- 单元测试：`code/regor/tests/test_cut_regor.py`。
