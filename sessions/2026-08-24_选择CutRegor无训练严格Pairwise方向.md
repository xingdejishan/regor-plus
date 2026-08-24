# 选择 CutRegor 无训练严格 Pairwise 方向

日期：2026-08-24

用户在授权学习型严格 pairwise 轨道后，进一步提供 CutRegor 完整方案并明确要求按该方案实现。当前执行主线切换为无新增训练的严格 pairwise CutRegor；学习型 D²-Regor 授权保留，但其实现计划暂停，不与 CutRegor 混合。

CutRegor 首先按方案原文执行 Frontier Separability Test。该测试是 GT-after-online 的离线容量诊断，不是完整方法结果；只有预注册机制门槛通过，才继续同一计划的 branch-and-contract 完整实现。

详细计划：`plan/2026-08-24_CutRegor渐进约束再生成完整计划.md`。

执行结果：正式 Frontier Test 在 201 个可观测 hard pair 上只满足“四锚点优于单点”一项门槛；frontier 相对 random 仅 `+3.99` 个百分点、正模式 survival `75.28%`、多尺度略低于单尺度。按用户给出的方案和预注册规则，CutRegor symmetry-frontier bridge family 已证伪并停止，没有继续实现 branch-and-contract，也没有报告端到端 RR。
