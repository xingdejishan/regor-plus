# CutRegor

CutRegor 是严格 pairwise、纯几何、无新增训练的部分自对称边界与四锚 bridge correspondence 实验实现。每个样本只使用当前 source/target 点云、冻结点特征及其确定性几何派生量。

## 当前结论

本目录实现了预注册 Frontier Separability Test 所需的完整核心机制，但正式 hard247 结果只通过四项继续条件中的一项，因此 `continue_to_branch_and_contract=false`。本版本不是端到端 CutRegor，也没有最终 RR；按计划没有继续实现 AC-3、位姿区间分裂和 rigid patch cover。

正式指标：

- 201/247 对可评估，0 error；
- symmetry-frontier separation `16.60%`，random `12.61%`，增益 `+3.99` 个百分点；
- 正模式 bridge survival `75.28%`；
- 单尺度 frontier `16.72%`，多尺度 `16.60%`；
- 四锚点 `16.60%`，单点局部域 `3.54%`。

## 文件

- `config.py`：冻结配置；
- `geometry.py`：分辨率、FPS、多尺度 patch graph 与位姿运算；
- `frontier.py`：mode-induced partial symmetry、图边界和 query 选择；
- `bridge.py`：描述子对应、最小奇异值条件化四锚点与 set-valued bridge domain；
- `prepare_frontier_mode_pool.py`：从冻结 G17 工件生成物理去 GT 的候选池；
- `analyze_frontier_separability.py`：GT-after-online Frontier Test；
- `run_frontier_candidates.py`：早期自然 refined Top-50 工程 runner，不是正式 G17 实验入口。

正式实验使用工作区中的 `code/regor/diagnostics/einfo_online_artifact.py` 读取物理脱敏 point artifact。`prepare_frontier_mode_pool.py` 的输出不包含 `gt_trans` 或成功标签；GT 只由独立 analyzer 读取，用于离线构造正/负 mode pair。

## 测试

在仓库根目录执行：

```bash
PYTHONPATH=code/regor:code/regor/tests python -c 'import test_cut_regor as t; [getattr(t,n)() for n in sorted(dir(t)) if n.startswith("test_")]'
```

完整报告位于 `analysis/regor/CutRegor实验报告.md`，正式汇总位于 `output/regor/2026-08-24_CutRegorFrontier_G17_Hard247_v2/frontier_summary.json`。
