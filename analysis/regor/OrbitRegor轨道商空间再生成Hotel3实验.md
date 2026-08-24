# OrbitRegor 轨道商空间再生成 Hotel3 实验

## 正式判断（v2）

完整progressive OrbitRegor在Hotel3全场景被证伪，按预注册规则停止，不扩大到hard247或自然1781，也不扫描等价参数。在线阶段`49/49`完成、六个分支均输出唯一有限刚体位姿、`0 error`、`gt_loaded=false`；实现hash、在线账本和全部trace hash通过独立复核。

| 方法 | 成功/49 | RR |
|---|---:|---:|
| Frozen REGOR baseline | 35 | 71.43% |
| Singleton-only OrbitRegor | 17 | 34.69% |
| Immediate instance lift | 14 | 28.57% |
| Delayed lift without commuting | 14 | 28.57% |
| Delayed lift without coalescence | 15 | 30.61% |
| Full without point-level REGOR | 14 | 28.57% |
| Full OrbitRegor | 15 | 30.61% |

Full相对baseline为`2 rescue / 22 damage / net -20`，two-sided exact McNemar `p=3.5882e-5`，paired bootstrap 95% RR difference CI为`[-57.14pp,-24.49pp]`。

以Full为第二项比较：

| 比较 | Full rescue | Full damage | net |
|---|---:|---:|---:|
| Full vs singleton-only | 0 | 2 | -2 |
| Full vs immediate lift | 1 | 0 | +1 |
| Full vs no commuting | 1 | 0 | +1 |
| Full vs no coalescence | 0 | 0 | 0 |
| Full vs no point-level REGOR | 1 | 0 | +1 |

Full平均从`5.06`个tight-domain birth atoms出发，经`2.24`轮progressive regeneration新增`4.31`个atoms；lift domain为`9.47→9.37`，non-singleton orbit支持`8.71`，relation rank `2.96`，物化`8.61`对，点级REGOR两轮后为`44.43`条对应。progressive quotient growth、commuting contraction、delayed lift和点级再生成均实际执行，但没有形成可靠truth mechanism。

判断原因：delayed lift与commuting各只净改变1对成功，coalescence不改变成功标签，多实例orbit相对singleton净损2对；Full相对baseline的置信区间完全低于零。重复错误结构可以与正确结构一样形成自同构orbit、交换关系和partial bijection，内部自洽不能提供物理实例身份。

经验：保留orbit-valued state作为未来新identity观测下的歧义表示；停止当前PLY-only的orbit signature、patch尺度、relation tolerance、lift轮数、词典序和particle budget变体。冻结的是本次具体方法族，不外推为所有quotient-to-lift表述均不可能。

正式证据：

- 在线输出：`output/regor/2026-08-24_OrbitRegor_v2_Hotel3_49/`
- 独立评价：`output/regor/2026-08-24_OrbitRegor_v2_Hotel3_49_eval/summary.json`

## v1 实现审计

首个Hotel3 v1得到强负数值，但review发现所有lift atoms在首轮一次性进入，只有域收缩，没有实现计划要求的progressive quotient correspondence growth。因此本节数值只作为实现审计，不作为完整OrbitRegor方法结论，也暂不据此冻结方法族。

在线阶段 `49/49` 完成、六个 OrbitRegor 分支全部输出唯一有限刚体位姿、`0 error`、`gt_loaded=false`。在线输出密封并记录账本与 trace SHA256 后，独立 evaluator 才读取 GT。

## v1 端到端数值

| 方法 | 成功/49 | RR |
|---|---:|---:|
| Frozen REGOR baseline | 35 | 71.43% |
| Singleton-only OrbitRegor | 16 | 32.65% |
| Immediate instance lift | 14 | 28.57% |
| Delayed lift without commuting | 14 | 28.57% |
| Delayed lift without coalescence | 15 | 30.61% |
| Full without point-level REGOR | 12 | 24.49% |
| Full OrbitRegor | 14 | 28.57% |

Full 相对 baseline 为 `2 rescue / 23 damage / net -21`，two-sided exact McNemar `p=1.9431e-5`，paired bootstrap 95% RR difference CI 为 `[-59.18pp,-26.53pp]`。

## v1 核心机制归因

以 Full 为第二项比较：

| 比较 | Full rescue | Full damage | net |
|---|---:|---:|---:|
| Full vs singleton-only | 0 | 2 | -2 |
| Full vs immediate lift | 0 | 0 | 0 |
| Full vs no commuting | 0 | 0 | 0 |
| Full vs no coalescence | 0 | 1 | -1 |
| Full vs no point-level REGOR | 2 | 0 | +2 |

Full、immediate、no-commuting、singleton 和 no-coalescence 的最终 pose 在 `49/49` 上并不数值相同，因此 `0/0` 不是代码分支未生效，而是不同 orbit 推断没有转化为阈值成功差异。

Full 平均机制量：

- source/target 非 singleton orbit：`33.61/34.39`；
- lift atoms：`192`；pose births：`50`；
- lift domain：`9.84 → 9.73`，平均只收缩约 `0.10` 个 atom；
- non-singleton orbit 支持：`9.04`；relation rank：`2.96`；
- materialized pairs：`8.96`；
- 两轮 REGOR 后 correspondence：`44.73`。

## v1 暴露的问题

Orbit cover、关系与实例域均实际进入求解，但quotient correspondence没有逐轮出生与扩张。review修正限定为：从tight pose-local lift domain出生，仅由当前已激活的跨orbit交换关系逐轮激活coarse feasible新atom，最后再做关系收缩。该修正补齐缺失状态转移，不调整任何结果驱动参数。

点级 REGOR 相对 no-point 分支净救回 2 对，符合“REGOR 能在已有局部先验附近扩张”的历史事实；但它无法修复 quotient-to-lift 已经选择的错误实例图谱。

## v1 暂定经验（已由v2复验）

- “延迟身份决定”作为表示是成立的，但相同 XYZ 表面内的自同构关系没有产生新的物理身份信息。
- orbit 数、交换约束秩和 atlas coverage 仍属于内部自洽量；重复错误结构同样能够满足它们。
- 不再实现 orbit signature、patch 尺度、orbit size、relation tolerance、lift round、lexicographic order 或 particle budget 变体。
- 如果未来获得新的合法身份观测，orbit-valued state 可以作为承载歧义的表示复用；在当前 PLY-only 无训练边界内，它不能作为 truth mechanism。

## 证据

- 在线输出：`output/regor/2026-08-24_OrbitRegor_Hotel3_49/`
- 独立评价：`output/regor/2026-08-24_OrbitRegor_Hotel3_49_eval/summary.json`
- 计划：`plan/2026-08-24_OrbitRegor轨道商空间再生成完整计划.md`
- 实现：`code/regor/orbit_regor/`
