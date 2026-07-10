# IRIS-Reg：迭代射线矛盾约束搜索

## 1. 方法定义

IRIS-Reg（Iterative Ray-contradiction-constrained Search）是一个可重复调用的搜索算子，不是固定的 Round2。REGOR 的 R1 只负责初始化；之后每一轮根据当前位姿暴露出的 RGB-D 射线矛盾，构造局部逃逸约束，生成多个 SE(3) 候选，并将历史失败位姿累积为禁域。

推理阶段没有 GT，因此不能保证每轮真实 RE/TE 都下降。IRIS-Reg 保证的是：候选池中历史上已验证的最优位姿不会被删除，输出不会在独立 validation 射线能力上退化。真实姿态误差、Oracle Recall 和实际选择 Recall 必须分开报告。

## 2. 总体流程

```text
REGOR R1 初始化
→ 构建双向 RGB-D 射线证据
→ 评估当前位姿的射线矛盾
→ 构造局部逃逸约束
→ 累积历史失败假设的 SE(3) 禁域
→ 约束对应生成并产生多个候选位姿
→ 候选局部细化与 SE(3) NMS
→ search / validation 射线集合独立评分
→ 保留历史最优并选择下一轮 incumbent
→ 继续下一轮或停止
```

IRIS-Reg 通过 `mode="iterative_ray"` 接入 `test_3DLoMatch.py`，不再把新方法实现为一次性的 `if enter_round2` 分支。

## 3. 射线证据

每个 fragment 的 `RayBundle` 保存：

```python
frame_ids: Tensor       # [R]
origins: Tensor         # [R, 3]
directions: Tensor      # [R, 3]
observed_depth: Tensor  # [R]
pixels: Tensor          # [R, 2]
camera_poses: Tensor    # [R, 4, 4]
confidence: Tensor      # [R]
split: Tensor           # search / validation
```

每条有效深度射线保存三类区间：

- 自由空间：`[0, d - mu]`
- 表面支持：`[d - mu, d + mu]`
- 遮挡之后：不直接惩罚

source 点投影到 target RGB-D 帧；target 点通过 `T^-1` 投影到 source RGB-D 帧。search/validation 按固定 frame hash 在实验开始前拆分，默认 70%/30%，两组永不交叉。

## 4. 射线矛盾

对 source 点 `p_i` 在当前位姿 `T_t` 下投影到 target 相机 `k`：

```math
x_{ik}=T_t p_i, \qquad z_{ik}=\operatorname{camera\text{-}depth}(x_{ik})
```

若观测深度为 `d_ik`，自由空间违背量为：

```math
v_{ik}(T_t)=\max(0,d_{ik}-\mu-z_{ik})
```

只有预测点落入传感器已观测自由空间时才产生惩罚。表面支持为：

```math
s_{ik}(T_t)=\exp\left(-\frac{|z_{ik}-d_{ik}|}{\sigma}\right)
```

表面支持需要像素有效、投影在视锥内并且没有明显前景遮挡。先按帧聚合，再在 frame 之间取鲁棒中位数：

```math
E_{free}(T)=\operatorname{median}_k
\frac{\sum_i w_{ik}\rho(v_{ik})}{\sum_i w_{ik}+\epsilon}
```

```math
S_{surface}(T)=\operatorname{median}_k
\frac{\sum_i w_{ik}s_{ik}}{\sum_i w_{ik}+\epsilon}
```

## 5. 局部逃逸约束

采用左乘李代数扰动：

```math
T(\delta\xi)=\exp(\delta\xi^\wedge)T_t
```

对深度的一阶近似为：

```math
z_{ik}(\delta\xi)\approx z_{ik}(0)+J_{ik}\delta\xi
```

其中：

```math
J_{ik}=e_3^\mathsf{T}R_{cw,k}
\begin{bmatrix}-[T_tp_i]_\times & I\end{bmatrix}
```

对高置信矛盾射线构造：

```math
G_t\delta\xi\ge b_t
```

使用带单轮扰动边界的鲁棒约束优化：

```math
\delta\xi_t^*=\arg\min_{\delta\xi}
\sum_k w_k\rho\left(\max(0,b_k-G_k\delta\xi)\right)
 +\lambda\|\delta\xi\|^2
```

输出逃逸方向、`G^T W G` 信息矩阵、已充分约束的旋转/平移方向和不确定方向。

## 6. 历史约束记忆

每个拒绝候选保存：

```python
pose
ray_signature
active_ray_ids
local_G
local_b
radius_rotation
radius_translation
```

新候选只有同时满足以下条件，才可重新进入历史失败 basin：

```math
d_{SE(3)}(T,T_j)<r_j
```

```math
\cos(c(T),c(T_j))>\eta
```

```math
E_{search}(T)>E_{search}(T_j)-\delta
```

这避免错误地删除“位姿接近但已经解除冲突”的候选。

## 7. 多候选生成

`RayGuidedRegenerator.generate()` 每轮：

1. 为每个 source 点保留 target 描述子 top-L 候选；
2. 使用射线逃逸约束重排对应分数；
3. 从高分对应中构造三点或多点几何一致 seed；
4. 用加权 SVD 生成原始候选位姿；
5. 调用现有 Regor paired-local GMM 做局部细化；
6. 执行 SE(3) NMS；
7. 输出多个不同候选。

候选分数为：

```math
S_{ij}=S_{desc}(i,j)+\lambda_eS_{escape}(i,j)-\lambda_hS_{history}(i,j)
```

对候选对应 `q_j-T_tp_i`，使用局部 Jacobian 最小范数修正估计 `\widehat{\delta\xi}_{ij}`，并计算：

```math
S_{escape}(i,j)=-\frac1M\sum_{k=1}^M
\max(0,b_k-G_k\widehat{\delta\xi}_{ij})
```

## 8. 候选池与 SE(3) NMS

每轮候选池必须包含当前 incumbent 和历史候选。SE(3) 距离为：

```math
d(T_a,T_b)=\sqrt{
\left(\frac{\theta(R_a^TR_b)}{\sigma_R}\right)^2+
\left(\frac{\|t_a-t_b\|}{\sigma_t}\right)^2}
```

距离小于阈值时只保留证据更好的候选，并记录 `unique_pose_count`。默认保留约 20% 的独立探索候选；探索候选不使用当前逃逸方向，但仍受历史重复位姿 NMS 约束。

## 9. 主动射线选择

候选池为 `H_t` 时，射线的信息量定义为：

```math
I(r)=\operatorname{Var}_{T\in H_t}[v_r(T)]
```

优先选择：

1. 当前 incumbent 矛盾严重的射线；
2. 不同候选之间预测差异最大的射线；
3. 历史轮次中使用频率低的射线；
4. 来自不同相机帧和视角的射线。

每轮只激活固定数量的有效射线，默认 2,000 条，避免重复扫描完整 RGB-D 数据。

## 10. Search / validation 选择

验证分数为：

```math
Q_{val}(T)=\widetilde E_{free}(T)+
\lambda_s[1-\widetilde S_{surface}(T)]
```

search 和 validation 使用固定、互斥的射线集合。候选池：

```math
P_t=\{T_{best}\}\cup H_t
```

只有：

```math
Q_{val}(T_{new})<Q_{val}(T_{best})-\delta_Q
```

且表面支持没有明显崩溃时，才替换 incumbent。推理阶段不能使用 GT；GT 只在实验审计阶段计算。

## 11. 迭代搜索器

```python
archive = HypothesisArchive(r1_pose)
memory = ConstraintMemory()

for round_id in range(max_rounds):
    incumbent = archive.best_pose
    active_rays = ray_selector.select(incumbent, archive.hypotheses, memory)
    constraints = constraint_builder.build(incumbent, active_rays)
    proposals = regenerator.generate(
        current_pose=incumbent,
        escape_constraints=constraints,
        history_memory=memory,
        candidate_count=candidates_per_round,
    )
    proposals = pose_nms(proposals)
    evaluator.score_search(proposals)
    evaluator.score_validation(proposals)
    archive.add(proposals)
    archive.update_best()
    memory.add_rejected(archive.rejected_this_round)
    if should_stop(archive, time_budget):
        break
```

当前 REGOR 接口中只保留：

```python
if mode == "iterative_ray":
    pred_trans, archive = iterative_search.run(
        r1_pose=r1_trans,
        inputs=pair_inputs,
        time_budget=config.ray_time_budget,
    )
```

IRIS-Reg 模块不能接收 `gt_trans`。

## 12. 默认参数

```text
max_rounds = 4
candidates_per_round = 16
active_ray_count = 2000
ray_search_fraction = 0.70
ray_surface_mu = 0.05 m
ray_surface_sigma = 0.03 m
ray_top_l = 5
escape_lambda = 1.0
history_lambda = 1.0
history_rotation_radius = 10 degrees
history_translation_radius = 0.20 m
pose_nms_rotation_sigma = 5 degrees
pose_nms_translation_sigma = 0.10 m
pose_nms_threshold = 1.0
validation_delta = 0.01
```

所有参数必须在配置和日志中显式报告，并支持 `max_rounds`、`candidates_per_round`、ray budget、top-L、escape/history 权重和 NMS 阈值消融。

## 13. 实验协议

第一层是生成能力实验：固定相同 R1、轮数、候选预算、描述子、paired-local GMM 和运行时间；关闭 early stop、门控、CTC 和 final merge，只比较候选池。

必须报告：

- `OracleRR@round1` 到 `OracleRR@round4`
- 每轮首次出现正确位姿的样本数
- R1 failure 子集上的 Oracle Recall
- 每轮独立位姿数量和 `unique_pose_count`
- 历史拒绝 basin 数量
- candidate generation time

对照组为等算力 repeated REGOR：相同 R1、相同轮数和候选数，但不使用射线逃逸约束和历史冲突禁域。

还需运行 shuffled-ray 安慰剂：保留所有代码不变，仅将 pair-specific 射线约束随机替换为另一 pair 的约束。只有真实射线显著优于 shuffled-ray，才能说明提升来自物理证据。

第二层固定候选池后独立评估选择器；第三层再单独评估最终 estimator。必须保存 `T_raw`、`T_local_refined`、`T_merge_refined`、`T_final`，防止把后处理影响错误归因于 IRIS-Reg。

## 14. 日志

每个 pair、每轮一行：

```text
pair_id, round_id, elapsed_time
incumbent_pose, incumbent_search_energy, incumbent_validation_energy
active_ray_count, new_active_ray_count, violation_frame_count, mean_violation_depth
constraint_rank, constraint_condition_number
escape_rotation_norm, escape_translation_norm
generated_candidate_count, unique_candidate_count
rejected_by_history_count, independent_explore_count
round_oracle_success, cumulative_oracle_success
cumulative_best_re, cumulative_best_te, first_success_round
selected_candidate_id, selected_success, selection_oracle_gap
```

每个 candidate 单独保存：

```text
candidate_id, parent_candidate_id, round_id, pose, seed_ids
correspondence_count, descriptor_score, predicted_escape_score
actual_search_energy, validation_energy, surface_support
nearest_history_pose_distance, history_signature_similarity
gt_re, gt_te, gt_success
```

`gt_*` 只在审计阶段填充，不能进入搜索类输入。

## 15. 主要指标

```math
OracleRR(t)=\frac1N\sum_{n=1}^N
\mathbf1\left[\exists T\in\bigcup_{\tau=0}^tH_\tau^{(n)}:
RE(T)<15^\circ, TE(T)<30\text{ cm}\right]
```

同时报告：

- 实际选择 RR；
- `SelectionEfficiency = 实际选择成功数 / Oracle 可成功数`；
- `GateRecall_repairable`；
- 候选池平均 unique pose 数；
- 首次成功轮次；
- search/validation 分数和 GT RE/TE；
- IRIS-Reg、Repeated-Regor、shuffled-ray 三组对照。

## 16. 实现边界

第一版只验证射线约束是否在相同候选预算下提升累计 Oracle RR。门控、动态停止和最终融合不参与核心模块结论；只有候选生成能力成立后，才接入实际选择器和最终 estimator。
