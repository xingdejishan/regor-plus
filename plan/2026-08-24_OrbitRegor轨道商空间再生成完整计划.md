# OrbitRegor 轨道商空间再生成完整计划

状态：已完成并按预注册停止条件冻结为负结果。轨道：无新增训练、PLY-only、严格 pairwise REGOR；端到端输出唯一刚体位姿。首个Hotel3 v1在review中发现lift atoms一次性进入、缺少真正的逐轮quotient growth，只作为实现审计；补齐progressive orbit regeneration后的v2为正式结果。

## 目标

实现并验证完整链路：

`当前 source/target XYZ 与冻结特征 → 多尺度 patch atlas → 云内近似自同构 orbit cover → orbit-valued correspondence → quotient growth → delayed instance lift → 点级 REGOR densification → 唯一位姿`

OrbitRegor 不把重复实例立即压成确定点对，而以轨道对应和实例提升域作为中间状态；只有提升域被跨轨道关系收缩后才物化点对应。

## 继承事实

- `analysis/regor/experiment-summary.md`：正确 mode 经常能够出生，但同源表面内点数、残差、PPF、Chamfer、稳定性、拓扑、最优传输和公共见证不能稳定判真。
- `analysis/regor/CutRegor实验报告.md` 与 `output/regor/2026-08-24_CutRegorFrontier_G17_Hard247_v2/frontier_summary.json`：symmetry frontier separation 仅 `16.60%`，不能依赖 mode 自生 bridge evidence。
- `analysis/regor/DomainRegor联合对应域可行性实验.md` 与 `output/regor/2026-08-24_DomainRegorJointDomainFeasibility_hard247/joint_feasibility_summary.json`：正确刚体保留 `94.95%`，错误收缩仅 `7.25%`，非平局正例胜率 `51.41%`；重复错误可形成一对一、距离一致和 proper tetrahedron 的刚体部分同构。
- `papers/3D Registration with Maximal Cliques.md`：多个 maximal clique 有利于 mode birth，但仍缺少 truth selector。
- `papers/Progressive Correspondence Regenerator for Robust 3D Registration.md`：正确局部先验存在时，REGOR 能扩大 correspondence；它在本方法中只承担实例提升后的点级 densification。
- `papers/H8联合生成池并入V9文献综述.md`：H8 是条件容量适配，历史 Oracle region 接口不得进入本轮在线链路；本轮仅复用当前 pair 的原始 correspondence compatibility clique 原理。

## 禁用方法族

- 禁止相机、深度射线、free-space、RGB/RGB-D、语义、第三点云、跨 pair 状态和 GT 派生在线输入。
- 禁止重新扫描内点数、残差、PPF、Chamfer、稳定性、Top-K、温度、权重、阈值和轮数。
- 禁止 baseline/challenger gate、逐 pair fallback、best-of-K、Oracle candidate selection。
- 不把 orbit detection、singleton cover、兼容图加深、共同刚体 cover 或更强 all-different 单独称为创新。
- 不复用 H8 历史 Oracle target-region action；候选出生只能来自当前 pair 的冻结 descriptor correspondence 与兼容 clique。

## 新增机制

1. `orbit-valued correspondence`：对应基本单元由确定点对改为 source/target 自相似轨道之间的实例映射域。
2. `quotient relation complex`：保存轨道间的相对布局关系束，而非把重复实例当成独立票数。
3. `delayed instance lift`：通过共享位姿、交换关系和 partial bijection 收缩实例域，域收缩后才物化点对应。
4. `atlas coalescence`：兼容局部轨道图谱按共同轨道映射、位姿邻近和提升域兼容性合并。
5. 两级再生成：轨道层先扩张结构并收缩身份，点层最后调用原生 REGOR 扩张。

## 冻结工程协议

- patch atlas 使用确定性 FPS；尺度由当前 pair 的 median-NN 分辨率确定。
- patch signature 仅由多尺度邻域的 PCA 谱、径向距离分布、邻域距离谱和法向无符号关系构成。
- 每个 patch 始终保留 singleton orbit；多实例 orbit 由同云互惠 signature 邻居、空间分离和局部谱一致性构成重叠 cover。
- 当前 pair 的冻结 descriptor correspondence 生成 cross-patch lift atoms；兼容 clique 和官方 SC² candidate 只用于 atlas particle birth。
- 轨道交换约束以 source/target 轨道实例相对位移在当前旋转下的一致性实现；收缩同时执行 partial one-to-one。
- 最终粒子按预注册词典序选择：已解释非 singleton orbit 数、交换约束独立秩、唯一提升实例数、空间覆盖、最后才是几何残差。
- 物化后的 patch 对与当前粒子支持的原始 correspondence 联合进入未修改 REGOR regeneration 和 Estimator，输出唯一位姿。

## 完整消融

- Frozen REGOR baseline。
- Singleton-only：保留图谱增长但禁止多实例 orbit。
- Immediate lift：orbit cover 建立后立即按当前位姿选实例。
- Delayed lift without commuting relations。
- Delayed lift without atlas coalescence。
- Full OrbitRegor without point-level REGOR。
- Full OrbitRegor。

所有消融必须运行至唯一最终位姿；orbit 数、压缩率、域收缩率和 correspondence 增长量仅作机制诊断。

## 数据与评价

- smoke 只检查接口、数值、内存和协议。
- 自然 1781 与 hard247 均为开发数据；正式结论只能在结构和参数冻结后的 scene-disjoint 数据上获得。
- 开发评价统一报告完整分母、baseline/OrbitRegor、逐场景 RRE/RTE、rescue、damage、net、McNemar、paired bootstrap CI、运行时间和失败数。
- 在线 runner 只接收内容哈希 token、当前 pair XYZ 和冻结官方特征；不得接收 scene/pair/file identity。输出密封并记录 SHA256 后，独立 evaluator 才能读取 GT。

## 停止条件

满足任一核心机制反证则冻结整个 orbit-to-lift 方法族，不扫描等价参数：

1. delayed lift 与 immediate lift 的端到端结果无差异；
2. commuting relation 只提高内部一致性，不改变最终位姿；
3. singleton-only 与 full orbit 的结果相同；
4. 正确和错误模式在 quotient atlas 中仍保持同构；
5. 增益只来自候选 clique birth，而 orbit-to-lift 主体无独立贡献；
6. 完整开发集没有稳定净增；
7. scene-disjoint 冻结测试置信区间跨零。

## Review 顺序

1. 输入白名单、GT 隔离、唯一输出和完整分母。
2. orbit cover、域收缩、交换约束、partial bijection、粒子合并和刚体有限性单测。
3. immediate/delayed、singleton/full、commuting/no-commuting 的端到端机制归因。
4. 实现 hash、在线账本、trace hash 和独立 evaluator 复核。

## 首次实现审计结果（v1，不作为最终方法结论）

完整实现、六个端到端消融、无身份在线 runner 和独立 evaluator 均已完成。Hotel3 在线 `49/49`、0 error、所有分支唯一有限位姿、`gt_loaded=false`；账本和49条trace全部通过hash检查。

baseline/singleton/immediate/no-commuting/no-coalescence/no-point/full成功数为：

`35/16/14/14/15/12/14`。

Full 相对 baseline 为 `2 rescue / 23 damage / net -21 / p=1.9431e-5`，paired bootstrap 95% RR difference CI `[-59.18pp,-26.53pp]`。Full 相对 immediate 与 no-commuting 均为 `0 rescue / 0 damage`；singleton-only 相对 full 多成功2对；full相对no-point只净救2对。

数值同时触发停止条件1、2、3和6的场景级强反证，但review发现v1所有lift atoms在首轮一次性进入，只实现了domain contraction，没有实现计划要求的progressive quotient correspondence growth。因此v1不用于冻结方法族；仅补齐缺失机制，不改变输入、atlas、orbit、candidate、阈值、预算、词典序目标或REGOR下游，随后重跑同一Hotel3协议。

## 正式结果（v2）

review修正后的实现从tight lift domain出生，仅由当前已激活的跨orbit commuting relation逐轮激活coarse-feasible atom，再执行关系收缩与实例物化。Hotel3完整场景在线`49/49`、六分支、0 error、`gt_loaded=false`；实现hash、在线账本和49条trace全部通过独立复核。

baseline/singleton/immediate/no-commuting/no-coalescence/no-point/full成功数为：

`35/17/14/14/15/14/15`。

Full相对baseline为`2 rescue / 22 damage / net -20 / p=3.5882e-5`，paired bootstrap 95% RR difference CI `[-57.14pp,-24.49pp]`。Full相对singleton为`0/2`，相对immediate、no-commuting、no-point均为`1/0`，相对no-coalescence为`0/0`。

progressive机制确实执行：Full平均从`5.06`个birth atoms出发，经过`2.24`轮再生成新增`4.31`个atoms，lift domain最终`9.47→9.37`。因此正式负结果不再能归因于“缺少quotient growth”。delayed lift与commuting各只改变1对成功，coalescence不改变任何成功标签，多实例orbit反而比singleton净损2对；完整方法相对冻结baseline显著退化，触发停止条件3和6，并为停止条件1、2、5提供强机制证据。

决策：不扩大到hard247或自然1781，不扫描orbit signature、patch尺度、relation tolerance、lift轮数、词典序或particle budget。冻结的是当前纯PLY、自同构orbit→commuting contraction→delayed lift→REGOR这一具体方法族；不宣称所有抽象quotient-to-lift方法不可能。若未来出现合法的新identity观测，可复用orbit-valued state，但必须提出新的truth mechanism并重新预注册。
