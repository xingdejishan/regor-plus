# OrbitRegor 完整实现与负结果冻结

用户授权按 OrbitRegor 方案执行：在严格 pairwise、PLY-only、无新增训练边界内，把 correspondence 基本单元改为几何自同构 orbit，执行 quotient growth、delayed instance lift、atlas coalescence，最后接原生 REGOR 输出唯一位姿。

已完成完整方法、六个端到端消融、无身份在线 runner、独立 evaluator 和四项结构测试。Hotel3 49-pair 在线 `49/49`、0 error；baseline/full 为 `35/14`，full 相对 baseline `2 rescue/23 damage/net -21`。full 相对 immediate 与 no-commuting 均为 `0/0`，singleton-only 比 full 多成功2对，触发预注册停止条件。

Review更正：首个v1把全部lift atoms一次性送入域收缩，缺少计划要求的progressive quotient growth，因此原冻结决定撤回，v1只作实现审计。只补齐tight-domain birth与交换关系驱动的逐轮atom regeneration，不改参数和其余链路，然后重跑Hotel3。

v2正式复验完成：Hotel3在线`49/49`、六分支、0 error、GT隔离与全部hash通过；baseline/singleton/immediate/no-commuting/no-coalescence/no-point/full为`35/17/14/14/15/14/15`。Full相对baseline为`2 rescue/22 damage/net -20`，McNemar `p=3.5882e-5`，paired bootstrap 95% RR差值CI `[-57.14pp,-24.49pp]`。

Full平均从`5.06`个birth atoms出发，经`2.24`轮新增`4.31`个atoms，确认progressive机制实际执行。Full相对immediate与no-commuting均仅净增1对，相对no-coalescence为0，相对singleton净损2对。按预注册停止条件冻结当前PLY-only orbit-to-lift方法族，不扩大到hard247/1781，不扫描等价参数。
