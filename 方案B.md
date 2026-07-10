下面是基于当前 `codex/diagnosis-guided-regor-s2` 分支的完整修改清单。目标不是继续修补一次性 R2，而是把现有代码改成一个统一的**迭代射线约束搜索模块**。

## 一、必须先修复的现有代码问题

### 1. 修复 `weakness_guided` seed 实际未被使用

当前 `test_3DLoMatch.py` 先生成：

```python
r2_seed_src, r2_seed_tgt = sample_guided_seed_correspondences(...)
```

随后却设置：

```python
regenerate_mode = "guided_global"
```

而 `Regenerator.regenerate()` 遇到 `guided_global` 后直接调用：

```python
guided_global_matching(
    src_point,
    tgt_point,
    src_feature,
    tgt_feature,
    guide,
    sampling_num,
)
```

这条路径不读取 `src_key_corr` 和 `tgt_key_corr`，因此前面计算的 guided seed 被丢弃。([GitHub][1])

需要修改为：

```python
regenerator.generate_hypotheses(
    seed_src=r2_seed_src,
    seed_tgt=r2_seed_tgt,
    ...
)
```

并确保 seed 参与：

* 局部 KNN 搜索；
* 局部对应再生；
* 位姿估计；
* 候选所属 seed 的审计记录。

不要继续使用当前 `guided_global` 的提前返回路径。

---

### 2. 删除“一次 R2”的硬编码结构

当前 `active_max_rounds` 只参与：

```python
active_max_rounds >= 2
```

随后代码仍只执行一次 `if enter_round2:`，并不存在循环。([GitHub][1])

改为：

```python
archive = HypothesisArchive(r1_pose)

for round_id in range(config.search_max_rounds):
    result = iterative_search.step(...)
    archive.add(result.hypotheses)

    if result.should_stop:
        break
```

不再使用 `R2/R3/R4` 分支命名，统一称为 `round_id=1,2,...`。

---

### 3. 禁止新一轮位姿无条件覆盖历史最优位姿

当前代码执行：

```python
pred_trans = r2_trans
```

然后直接把 R1、R2 对应拼接。R2 即使更差，也会先覆盖 R1。([GitHub][1])

需要改为：

```python
candidate_pool = [incumbent_pose] + new_hypotheses
best_pose = validator.select(candidate_pool)
```

且只有满足：

```python
new_score < incumbent_score - min_improvement
```

时才更新 incumbent。

每轮必须保留历史最优位姿，不能原地替换。

---

### 4. 禁止默认拼接 R1 和新一轮对应

当前流程将：

```python
src_final = cat([r1_src_corr, r2_src_corr])
tgt_final = cat([r1_tgt_corr, r2_tgt_corr])
```

之后再运行 estimator 或 robust refinement。([GitHub][1])

这会使错误的 R1 对应重新污染新候选。

改为每个候选独立保存：

```python
PoseHypothesis(
    pose_raw,
    src_corr,
    tgt_corr,
    match_weights,
    parent_id,
    seed_ids,
)
```

默认局部细化只能使用该候选自己的对应。`R1+candidate` 拼接只能作为单独的后处理消融，不能作为核心算法默认路径。

---

### 5. 从推理接口中移除 `gt_trans`

当前 `Regenerator.regenerate()` 接收并保存：

```python
self.gt_trans = gt_trans
```

虽然目前主要用于可视化和审计，但推理模块不应持有 GT。([GitHub][2])

接口改为：

```python
def generate_hypotheses(
    seed_src,
    seed_tgt,
    src_points,
    tgt_points,
    src_features,
    tgt_features,
    constraints,
    memory,
):
```

GT 只允许在外围 evaluator 中使用：

```python
audit_hypotheses(hypotheses, gt_trans)
```

---

### 6. 暂时关闭现有 early-stop 对核心模块的影响

当前早停为：

```python
fsv_value < tau_fsv and roc_bar > tau_rho
```

并且配置中 `active_tau_fsv=0.65`、`active_tau_rho=0.75`。([GitHub][1])

模块开发阶段应改为：

```json
"use_search_gate": false
```

所有样本固定运行相同候选预算，以免门控影响模块有效性判断。

端到端阶段再单独接入 gate，且不能继续把 `ROC_bar` 与 S2 混在一起后称为“S2 效果”。

---

### 7. 拆分四模式消融运行

当前：

```json
"run_round2_ablation_suite": true
```

会依次执行：

```python
none
original
random_uniform
weakness_guided
```

因此一次命令实际进行了四次完整测试。([GitHub][1])

修改为：

```json
"run_ablation_suite": false,
"experiment_mode": "iterative_ray"
```

每种方法独立运行、独立输出文件。

需要批量实验时，由外部脚本调用四次，并在每次运行开始前重新设置完全相同的随机种子。

---

## 二、重做自由空间数据表示

### 8. 保留旧标量 FSV，但只作为 baseline

当前 `compute_rgbd_fsv()` 仅进行：

```python
warped_src = transform(src_points, trans)
violation = target_free_space.contains(warped_src)
return violation.mean()
```

它没有射线、帧、深度差和冲突方向。([GitHub][3])

将其重命名为：

```python
compute_binary_voxel_fsv()
```

仅供旧方法对照使用，不能再承担新模块的约束构造。

---

### 9. 新增 `ray_evidence.py`

定义：

```python
@dataclass
class RayBundle:
    frame_ids: Tensor
    origins: Tensor
    directions: Tensor
    observed_depths: Tensor
    camera_poses: Tensor
    intrinsics: Tensor
    confidences: Tensor
    split_ids: Tensor
```

必须保存：

* 每条射线属于哪个 RGB-D 帧；
* 相机中心；
* 世界坐标方向；
* 实际观测深度；
* 有效深度掩码；
* 内参与相机位姿；
* search/validation 划分；
* 需要时保存像素位置。

不能再只保存 `free_voxels、origin、voxel_size`。当前 `FreeSpaceVolume` 的 NPZ 格式只包含这三个字段。([GitHub][3])

---

### 10. 数据加载必须同时返回 source 和 target 射线

当前数据接口的 11 项版本只额外返回了：

```python
target_free_space
```

没有 source 射线证据。([GitHub][1])

改为结构化返回，避免继续依赖 tuple 长度：

```python
@dataclass
class RegistrationPair:
    src_keypoints: Tensor
    tgt_keypoints: Tensor
    src_features: Tensor
    tgt_features: Tensor
    src_overlap: Tensor
    tgt_overlap: Tensor
    src_ray_bundle: RayBundle
    tgt_ray_bundle: RayBundle
    gt_transform: Tensor | None
    pair_id: str
```

推理模块接收不含 GT 的视图：

```python
pair.inference_inputs()
```

---

### 11. 双向射线检查

每个位姿 (T) 都计算：

* source 点经 (T) 投影到 target RGB-D 帧；
* target 点经 (T^{-1}) 投影到 source RGB-D 帧。

输出：

```python
RayEvaluation(
    free_violation,
    surface_support,
    valid_observation_count,
    per_frame_scores,
    per_ray_residuals,
)
```

不能继续只做 source→target 单向检查。

---

### 12. 区分自由空间、表面和未知区域

对观测深度 (d)：

```text
z < d - margin       自由空间冲突
|z-d| <= margin      表面支持
z > d + margin       遮挡或未知，不直接处罚
无有效深度            未知，不计入分母
```

分母必须是“有效可观测点数”，不能是全部 source 点数。

---

### 13. 固定 search/validation 射线划分

按帧划分，不按单条像素随机划分：

```python
search_frames = deterministic_hash(frame_id) < 0.7
validation_frames = ~search_frames
```

* search rays：构造约束、生成候选；
* validation rays：选择候选；
* 两者不可重叠；
* 所有比较方法使用同一划分。

---

## 三、新增逐射线约束模块

### 14. 新增 `ray_constraint_builder.py`

输入当前位姿和 search rays：

```python
constraints = builder.build(
    pose=current_pose,
    src_points=src_points,
    tgt_points=tgt_points,
    src_rays=src_rays,
    tgt_rays=tgt_rays,
)
```

每条冲突射线输出：

```python
RayConstraint(
    ray_id,
    point_id,
    violation_depth,
    pose_jacobian,   # [6]
    confidence,
    frame_id,
)
```

局部形式：

[
G_t\delta\xi \ge b_t
]

需要加入：

* Huber 或截断鲁棒权重；
* 每帧权重归一化；
* 最大单帧约束数量；
* Jacobian rank；
* 条件数；
* 旋转和平移可观测性。

---

### 15. 约束只能作为软引导，不能直接永久裁掉半空间

由于位姿变化后像素投影会改变，一阶 Jacobian 只在局部有效。

因此对应评分使用软处罚：

[
S_{\text{escape}}
=================

-\frac{1}{M}
\sum_m
w_m
\max(0,b_m-G_m\delta\xi)
]

而不是简单：

```python
if violates_constraint:
    remove_correspondence()
```

永久历史禁域必须通过实际重新投影验证，不能仅依赖一阶近似。

---

## 四、重做候选对应和 seed 生成

### 16. 新增 `ray_guided_regenerator.py`

它必须输出多个候选，不再直接压缩为一个位姿：

```python
list[PoseHypothesis]
```

每个候选包含：

```python
@dataclass
class PoseHypothesis:
    hypothesis_id: int
    parent_id: int
    round_id: int
    pose_raw: Tensor
    pose_local_refined: Tensor
    src_corr: Tensor
    tgt_corr: Tensor
    correspondence_scores: Tensor
    seed_ids: Tensor
    generation_mode: str
```

---

### 17. 描述子候选只计算一次并缓存

初始化时为每个 source 点计算 target top-(L)：

```python
descriptor_topk_indices
descriptor_topk_scores
```

后续每轮只重新计算射线引导分数：

[
S_{ij}
======

S_{\mathrm{desc}}
+
\lambda_eS_{\mathrm{escape}}
----------------------------

\lambda_hS_{\mathrm{history}}
]

不能每轮重新计算完整 source-target 描述子矩阵。

---

### 18. seed 必须由对应组合产生，而不是只给 source 点一个 prior

当前方法主要构造：

```python
source_prior
source_under_support
target_under_support
```

然后重新选择对应。它利用的是点级权重，不是对应级物理约束。([GitHub][2])

新方法必须给每个 ((p_i,q_j)) 单独评分，因为同一个 source 点匹配到不同 target 区域，产生的位姿方向完全不同。

---

### 19. 保留独立探索，但与 R1 真正去耦

每轮候选分成：

```text
约束引导候选：80%
独立探索候选：20%
```

独立探索候选：

* 不使用当前逃逸方向；
* 不使用 `o_best/o_broad`；
* 仍使用描述子和几何一致性；
* 仍受历史重复位姿 NMS 约束。

这是保证搜索完整性，不是退回旧方法。

---

### 20. 一个 seed 必须生成一个局部候选

禁止将所有 seed 的对应全部混合后只估计一个位姿。

正确结构：

```python
for seed_group in seed_groups:
    local_corr = regenerate_around(seed_group)
    pose = estimate(local_corr)
    hypotheses.append(pose)
```

随后再做候选级筛选。

---

### 21. 添加 SE(3) 候选去重

当前数组中的 top-k 数量不能代表真实候选多样性。

定义：

[
d(T_a,T_b)
==========

\sqrt{
(\theta/\sigma_R)^2+
(|t_a-t_b|/\sigma_t)^2
}
]

当两个候选距离低于阈值时，只保留验证分更好的一个。

必须记录：

```python
generated_count
unique_pose_count
duplicate_count
```

---

## 五、新增历史约束记忆

### 22. 新增 `ray_constraint_memory.py`

每个被拒绝的候选保存：

```python
@dataclass
class RejectedBasin:
    pose: Tensor
    ray_ids: Tensor
    ray_signature: Tensor
    search_energy: float
    local_information_matrix: Tensor
    rotation_radius: float
    translation_radius: float
```

---

### 23. 历史禁域不能只按位姿距离判断

新候选被认定为重复历史错误盆地，必须同时满足：

```text
SE(3) 距离接近
射线冲突签名相似
自由空间能量没有明显改善
```

即：

[
d_{\mathrm{SE(3)}}(T,T_j)<r
]

[
\cos(c(T),c(T_j))>\eta
]

[
E(T)\ge E(T_j)-\delta
]

否则，位姿接近但已经解除冲突的候选会被误删。

---

### 24. 历史记忆应保存候选，不保存轮次名称

不使用：

```text
R1 memory
R2 memory
R3 memory
```

统一按 `hypothesis_id` 和 `parent_id` 建立搜索树。

---

## 六、实现真正的迭代搜索器

### 25. 新增 `iterative_ray_search.py`

核心接口：

```python
class IterativeRaySearch:
    def run(
        self,
        initial_pose,
        pair_inputs,
        time_budget_seconds,
    ) -> SearchResult:
        ...
```

内部状态：

```python
archive
constraint_memory
descriptor_cache
ray_cache
current_incumbent
elapsed_time
```

---

### 26. 每轮执行顺序固定为

```text
评价 incumbent 的新射线冲突
→ 构造逃逸约束
→ 对对应候选重新评分
→ 生成多个 seed group
→ 生成多个位姿候选
→ SE(3) 去重
→ search rays 预筛
→ validation rays 独立选择
→ 更新 incumbent
→ 保存被拒绝盆地
```

不能在验证前就把所有候选对应和 incumbent 对应混合。

---

### 27. 主动选择增量射线

每轮不重新扫描全部射线。

射线优先级：

[
I(r)
====

\operatorname{Var}_{T\in\mathcal H}
e_r(T)
]

综合：

* incumbent 上冲突程度；
* 候选间分歧；
* 该射线历史使用次数；
* 相机视角多样性。

每轮只新增固定数量射线，并缓存历史计算结果。

---

### 28. 设置明确停止条件

停止条件只能来自：

```text
达到时间预算
达到最大轮数
连续若干轮无新 SE(3) 盆地
validation score 改善不足
候选生成数量不足
最优与次优候选分差足够大
```

第一阶段实验建议关闭自适应停止，固定轮数和候选数；最终推理再开启。

---

## 七、重做候选验证器

### 29. 新增 `ray_pose_validator.py`

使用 validation rays 评价：

[
Q(T)
====

E_{\mathrm{free}}(T)
+
\lambda_s(1-S_{\mathrm{surface}}(T))
]

还必须记录：

```python
free_violation
surface_support
valid_observation_ratio
frame_coverage
bidirectional_consistency
```

避免一个仅仅落在“未观测区域”的错误位姿得到低冲突分数。

---

### 30. 拒绝低可观测候选

当：

```python
valid_observation_count < min_valid_observations
```

候选不能因为没有被射线看见而获得高置信度。

它应被标记为：

```text
insufficient_evidence
```

而不是“低冲突正确”。

---

### 31. 验证器第一版不要训练

先使用固定的物理评分，避免候选生成模块的效果被另一个学习型判别器掩盖。

验证器参数只允许在开发集确定一次，测试集冻结。

---

## 八、修改 `test_3DLoMatch.py`

### 32. 将文件恢复为实验入口，不再承载算法主体

当前文件约 1100 行，并混合了：

* R1；
* FSV；
* ROC；
* prior；
* R2；
* 最终估计；
* 日志；
* 消融调度。([GitHub][1])

最终只应保留：

```python
pair = loader.get_pair(i)
r1_result = run_r1(pair)
search_result = iterative_search.run(
    initial_pose=r1_result.pose,
    pair_inputs=pair.inference_inputs(),
    time_budget_seconds=config.search_time_budget,
)
evaluator.update(search_result, pair.gt_transform)
```

---

### 33. 将现有辅助函数移出测试脚本

下列函数不应继续位于 `test_3DLoMatch.py`：

```text
build_guided_prior
sample_guided_seed_correspondences
attach_guided_candidate_pools
generate_r1_topk_transforms
select_best_r1_transform
robust_weighted_estimate
mini_ransac_transform
```

分别放入：

```text
legacy_guidance.py
hypothesis_generation.py
pose_refinement.py
```

旧方法保留用于 baseline，但不能和新方法共享可变状态。

---

### 34. R1 只运行一次

多模式或多轮运行时，缓存：

```python
R1Result(
    pose,
    correspondences,
    descriptors,
    topk_hypotheses,
)
```

不同方法读取完全相同的 R1 缓存，避免 R1 随机采样差异影响对比。

---

## 九、配置文件必须重构

### 35. 删除或移入 legacy 区域的参数

这些参数属于旧的点级 prior 方法：

```text
active_alpha
active_prior_lambda
active_eta_l
active_reset_gamma
active_reset_topk
active_reset_tau_fsv
active_broad_radius
active_support_radius
active_support_r1_weight
active_support_local_weight
active_tau_rho
```

保留时必须放入：

```json
"legacy_weakness_guided": {...}
```

不能继续混在新方法主配置中。

---

### 36. 新增统一配置

```json
{
  "method": "iterative_ray",
  "search_max_rounds": 4,
  "search_time_budget_seconds": 0,
  "candidates_per_round": 16,
  "independent_explore_fraction": 0.2,

  "descriptor_topk": 16,
  "seed_group_count": 32,
  "local_corr_max_points": 64,

  "active_rays_per_round": 3000,
  "search_frame_fraction": 0.7,
  "ray_trunc_margin": 0.05,
  "min_valid_ray_count": 200,

  "pose_nms_rotation_deg": 5.0,
  "pose_nms_translation": 0.10,

  "history_rotation_radius_deg": 10.0,
  "history_translation_radius": 0.20,
  "history_signature_similarity": 0.9,

  "validation_min_improvement": 0.01,
  "stagnation_rounds": 2,

  "enable_gate": false,
  "enable_candidate_merge": false,
  "enable_ctc": false
}
```

具体数值需要开发集确定，但字段结构应固定。

---

### 37. 配置校验必须覆盖所有实际使用字段

当前配置中存在多项 `active_round2_*` 参数，但现有 required-key 列表并未完整覆盖全部新增字段。([GitHub][4])

应采用 dataclass 或 schema：

```python
@dataclass
class IterativeRayConfig:
    ...
```

启动时一次性校验：

* 正数范围；
* 比例范围；
* 候选数；
* 时间预算；
* search/validation 是否互斥；
* 不允许推理模块启用 GT。

---

## 十、增加缓存和性能控制

### 38. 缓存以下固定结果

每个 pair 只计算一次：

```text
描述子 top-L 候选
source/target RayBundle
search/validation 帧划分
source/target 点的体素索引
R1 输出
相机投影常量
```

---

### 39. 候选批量射线评分

不要逐候选、逐射线 Python 循环。

接口：

```python
scores = ray_evaluator.evaluate_batch(
    poses=[K, 4, 4],
    points=[N, 3],
    rays=active_rays,
)
```

主要张量操作应在 GPU 上批量完成。

---

### 40. 运行时间分项统计

每对样本记录：

```text
r1_time
ray_load_time
constraint_time
candidate_generation_time
candidate_validation_time
refinement_time
total_time
```

否则无法判断耗时来自新射线模块还是原 Regor 对应再生。

---

## 十一、专门验证核心模块是否有效

### 41. 第一主实验完全关闭 gate 和最终融合

所有样本：

```text
固定 R1
固定轮数
固定候选预算
不 early-stop
不 CTC
不拼接 R1 对应
不使用最终判别器选择
```

仅保存所有原始候选，之后用 GT 计算累计 Oracle RR：

[
\mathrm{OracleRR@round}\ t
]

这直接衡量新模块是否生成了正确位姿，不受判别器影响。

---

### 42. 唯一主对照为等算力重复 Regor

对照方法必须拥有：

* 相同 R1；
* 相同候选总数；
* 相同轮数；
* 相同时间预算；
* 相同局部再生；
* 相同位姿估计。

唯一差异是：

```text
Repeated-Regor：不使用逐射线约束和历史禁域
Iterative-Ray：使用逐射线约束和历史禁域
```

---

### 43. 加入 shuffled-ray 安慰剂实验

将某一 pair 的射线约束替换为另一 pair 的射线约束，其余完全相同。

预期：

```text
正确射线 > shuffled rays ≈ 无射线约束
```

否则提升可能只是增加随机扰动或候选多样性造成的，而非物理射线信息。

---

### 44. 候选生成和候选选择必须分两次实验

第一次只测：

```text
archive 中是否出现正确位姿
```

第二次固定 archive 到磁盘，再比较选择器。

定义：

[
\text{Selection Efficiency}
===========================

\frac{\text{选择成功数}}
{\text{Oracle 可成功数}}
]

这样能明确区分：

* 候选生成失败；
* 验证器排序失败；
* 最终 refinement 破坏正确位姿。

---

### 45. 分别保存四种位姿结果

每个候选保存：

```text
pose_raw
pose_local_refined
pose_merge_refined
pose_system_final
```

核心模块主结果使用 `pose_raw` 和 `pose_local_refined`。

`pose_merge_refined` 或 `pose_system_final` 下降时，不能归因于射线搜索模块。

---

## 十二、必须增加的审计日志

### 46. pair-round 日志

```text
pair_id
round_id
elapsed_time
incumbent_id
incumbent_search_score
incumbent_validation_score
active_ray_count
new_ray_count
conflict_ray_count
valid_observation_count
constraint_rank
constraint_condition_number
generated_candidate_count
unique_candidate_count
history_rejected_count
independent_candidate_count
round_oracle_success
cumulative_oracle_success
best_re
best_te
first_success_round
```

---

### 47. candidate 日志

```text
pair_id
round_id
candidate_id
parent_id
generation_mode
seed_ids
correspondence_count
descriptor_score
escape_score
history_penalty
search_ray_score
validation_ray_score
surface_support
valid_observation_ratio
nearest_history_distance
signature_similarity
raw_re
raw_te
local_refined_re
local_refined_te
success
```

GT 字段只能由外围 audit 代码追加。

---

## 十三、单元测试和集成测试

### 48. 必须增加以下测试

```text
test_guided_seed_is_consumed
test_regenerator_returns_multiple_hypotheses
test_pose_nms_removes_duplicates
test_incumbent_is_never_dropped
test_gt_not_in_inference_signature
test_search_validation_rays_are_disjoint
test_unknown_space_is_not_free_space
test_bidirectional_ray_evaluation
test_history_basin_requires_signature_match
test_active_max_rounds_executes_real_loop
test_equal_seed_reproducibility
test_time_budget_stops_search
```

其中最重要的是：

```python
assert changing_guided_seeds_changes_generated_hypotheses
```

这能防止再次出现“seed 计算了但没有进入执行路径”的问题。

---