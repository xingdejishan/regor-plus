# AGENTS.md

## 目标

本项目是研究型点云配准方法实现，不是普通工程功能补丁。

实现时必须严格按照研究方案的语义定义完成，而不是在现有代码路径上做最小改动、保守补丁或近似替代。代码能跑通不等于方法被正确实现。

核心原则：

> 方案中的模块定义、输入来源、候选生成方式、搜索区域、诊断信号和早停逻辑，均属于方法本身的一部分，不能擅自简化或替换。

---

## 1. 实现前必须先输出实现计划

在写代码前，必须先给出实现计划，包含：

1. 将修改哪些文件；
2. 每个方案模块对应哪个函数或类；
3. 哪些旧逻辑会被替换；
4. 哪些变量是新定义；
5. 哪些地方可能只是 proxy 或 fallback；
6. 哪些参数需要新增；
7. 如何验证实现确实符合方案。

禁止直接开始写代码。

---

## 2. 重要参数必须显式报告

凡是定义训练参数、推理参数、阈值、采样数、温度系数、体素大小、过滤规则、迭代次数、候选数量、早停条件等重要内容时，必须在实现前或实现后明确报告。

报告格式：

```text
参数名：
作用：
默认值：
取值范围：
为什么设成这个值：
是否需要消融：
所在文件 / 函数：
```

必须报告的参数包括但不限于：

```text
K_top
tau_fsv
tau_rho
tau_overlap
alpha
lambda_prior
eta_L
gamma_reset
epsilon_log
n_min
n_max
voxel_size
ransac_iters
robust_kernel_c
max_rounds
early_stop_enabled
density_filter_enabled
```

如果新增了任何未列出的参数，也必须报告。

禁止在代码中静默写入 magic number。

---

## 3. 不允许把方案降级为工程需求

本项目不是“尽量少改代码跑通”。必须完整实现方案定义。

禁止以下行为：

1. 用已有变量冒充方案变量；
2. 用现有代码路径绕过核心模块；
3. 用 proxy 替代真实定义却不说明；
4. 因为接口方便就复用错误阶段的候选；
5. 只加 scalar gate，但不改变搜索域；
6. 保留旧逻辑导致方案核心模块没有生效；
7. 代码能跑但方法语义不一致。

如果某个模块因为数据、接口或算力限制无法完整实现，必须明确说明：

```text
无法完整实现的模块：
原因：
当前临时替代方案：
该替代方案与原方案的差异：
它会影响哪些实验结论：
```

---

## 4. Round2 主动生成的硬约束

Round2 是本方法的核心创新。必须实现为“按诊断结果改变搜索区域的主动再生成”，不能仍然只围绕 Round1 correspondence 做局部 KNN。

硬约束：

1. Round2 source seed 必须从 `O(p)` 高的 source 点中采样；
2. 禁止只使用 `r1_src_corr / r1_tgt_corr` 作为 Round2 seed；
3. Round2 必须允许访问 `O_best`、`O_broad`、`O_reset` 指定的新搜索区域；
4. 当进入 reset prior 时，Round2 必须能回到 Top-K survivor union 或 global matchability 区域；
5. 如果真实重叠区没有被 Round1 seed 覆盖，Round2 仍应有机会到达该区域。

如果代码仍然只在 Round1 seed 附近再生成，则该实现视为无效实现。

---

## 5. Top-K 候选的硬约束

Top-K 候选必须来自 Round1 GMM 之后的匹配集合 `C_R1`。

正确流程：

```text
C0
→ Round1 GMM
→ C_R1
→ generate {T1, ..., TK}
→ refine
→ select T_best
```

禁止用 initial matcher 阶段的 `seedwise_trans` 冒充方案中的 Top-K。

如果为了调试临时使用 initial matcher 的候选，必须命名为：

```text
initial_topk_proxy
```

并明确它不是方案定义中的 Top-K。

---

## 6. FSV 的命名与实现约束

严格 FSV 指：

> 使用 RGB-D frames、camera intrinsics、camera poses 或 free-space volume，检查候选位姿是否把 source 点放入 target 传感器已观测自由空间。

如果代码只是使用 bbox、最近邻距离、Chamfer distance、目标点云外包围盒或其他几何近似，则不能命名为 `FSV`。

必须命名为：

```text
fsv_proxy
```

或者：

```text
geometry_violation_proxy
```

并在日志和实验表格中明确标注。

禁止用 proxy 支撑“物理自洽”结论。

---

## 7. 早停逻辑硬约束

方案中的早停是：

```text
if FSV(T_best) < tau_fsv and ROC_bar(T_best) > tau_rho:
    return T_best
```

如果满足早停条件，就必须直接返回 Round1 refined pose `T_best`。

禁止早停后继续进入 final robust estimator 并改变位姿，除非方案被明确改写为：

```text
early stop + mandatory refinement
```

如果采用 mandatory refinement，必须在方案文档和实验命名中说明，不得仍称为原始 early stop。

---

## 8. Search prior 的实现约束

必须实现包容式 overlap prior：

```math
O(p)
=
v(T) \cdot NoisyOR(
    O_best(p),
    (1 - ROC_bar(T)) \cdot O_broad(p)
)
+
(1 - v(T)) \cdot O_reset(p)
```

其中：

```math
v(T) = exp(-alpha \cdot FSV(T))
```

或在使用 proxy 时：

```math
v(T) = exp(-alpha \cdot fsv_proxy(T))
```

约束：

1. `O_broad = dilate(O_best)`；
2. `O_broad` 不允许包含 Top-K union；
3. Top-K survivor union 只允许进入 `O_reset`；
4. `O_reset` 必须包含 dilation；
5. `O_reset` 应包含弱 global matchability fallback：

```math
O_reset = NoisyOR(dilate(O_surv), gamma \cdot O_global_matchable)
```

---

## 9. 几何力臂的数值安全约束

几何力臂必须使用无惩罚基线定义，禁止使用 `[0,1]` 距离直接取 log。

正确形式：

```math
d(p|c_ref) = clip(||p - c_ref|| / r_src, 0, 1)
L(p|c_ref) = 1 + eta_L \cdot d(p|c_ref)
```

含义：

1. 中心点 `L=1`，`log L=0`，不惩罚；
2. 远端点 `L>1`，`log L>0`，只奖励；
3. 避免点云中心出现 “log hole / donut effect”。

禁止使用：

```math
L = clip(||p - c_ref|| / r_src, 0, 1)
```

然后再计算：

```math
log(L)
```

---

## 10. GMM similarity 调制约束

必须使用带温度的 log-prior 调制：

```math
S'(p,q)
=
S_feat(p,q)
+
lambda_prior \cdot [
    log O_tilde(p)
    + log U_s_tilde(p)
    + log U_t_tilde(q)
    + log L_tilde(p|c_ref)
]
```

数值边界：

```math
x_tilde = clip(x, epsilon, 1.0), x in {O, U_s, U_t}
L_tilde = max(L, epsilon)
```

注意：

1. `O, U_s, U_t` 是抑制项，取值在 `[0,1]`，可以 clip 到 1；
2. `L = 1 + eta_L * d >= 1` 是奖励项，禁止 clip 到 1；
3. 如果把 `L` clip 到 1，会导致远端奖励完全失效；
4. `lambda_prior` 必须报告并消融；
5. 禁止无温度直接累加 log-prior。

---

## 11. Source / Target 欠支持权重约束

欠支持权重必须受当前位姿可信度 `v(T)` 调制。

正确形式：

```math
U_s(p) = 1 / (1 + v(T) \cdot rho_R1_source(p))
U_t(q) = 1 / (1 + v(T) \cdot rho_R1_target(q))
```

原因：

1. 当前位姿可信时，Round1 覆盖统计可信，可以抑制已覆盖区域；
2. 当前位姿不可信时，Round1 内点可能是错的，必须释放 source / target 抑制；
3. 禁止在 `v(T) -> 0` 时仍然用 Round1 错误覆盖统计误杀真实重叠区。

---

## 12. Final estimation 约束

合并 `C_R1 ∪ C_R2` 后，禁止直接做普通 SVD。

必须采用：

```text
density filtering
→ voxel-balanced weighting
→ robust weighted SVD 或 mini-RANSAC
```

顺序不能改变。

原因：

1. 先过滤孤立外点；
2. 再做空间均衡；
3. 最后用 robust estimator 抑制残余外点。

体素均衡不能直接使用裸：

```math
1 / n_voxel
```

否则孤立外点会因 `n_voxel=1` 获得高权重。

必须至少包含：

```math
w_density
w_balance
w_robust
```

推荐：

```math
w_final = w_match \cdot w_density \cdot w_balance \cdot w_robust
```

---

## 13. 实验与日志要求

每次运行必须输出以下信息：

```text
Pairs
Mean Reg Recall
Mean Re
Mean Te
Mean FMR
Round2 trigger rate
Early stop rate
Mean model time
Output precision / recall
R1 inlier count
R2 inlier count
R2 precision
Novel inlier coverage
Final used correspondence count
Density filtering drop ratio
```

如果目标是验证主动生成模块，必须单独报告：

```text
Round1 only
Round1 + original Round2
Round1 + random / uniform Round2
Round1 + weakness-guided Round2
```

禁止只报告 full pipeline 总 RR，然后直接判断主动生成模块是否有效。

---

## 14. 消融实验必须保留

至少需要支持以下消融开关：

```text
use_ctc
use_early_stop
use_round2
use_o_prior
use_o_reset
use_target_ut
use_adaptive_leverage
use_density_filter
use_robust_final
use_fsv_proxy
use_rgbd_fsv
```

并支持以下参数消融：

```text
lambda_prior
eta_L
gamma_reset
alpha
tau_fsv
tau_rho
n_min
n_max
K_top
```

---

## 15. 实现后必须自检

实现完成后，必须给出 checklist：

```text
[ ] Round2 seed 是否来自 O(p) top-N，而不是只来自 r1_corr？
[ ] Top-K 是否由 C_R1 重新估计，而不是 initial matcher？
[ ] FSV 是否是真 RGB-D free-space？如果不是，是否命名为 fsv_proxy？
[ ] early stop 是否直接 return T_best？
[ ] O_broad 是否只等于 dilate(O_best)，不含 Top-K？
[ ] O_reset 是否包含 dilate(Top-K survivor union) 和 global fallback？
[ ] L 是否为 1 + eta_L * d，而不是 [0,1] 后取 log？
[ ] L 是否没有被 clip 到 1？
[ ] U_s / U_t 是否被 v(T) 调制？
[ ] final estimation 是否先 density filtering 再 voxel balance？
[ ] 是否报告所有新增参数？
[ ] 是否单独验证 weakness-guided Round2？
```

如果任一项未满足，必须说明原因，且不能声称方案已完整实现。

---

## 16. 开发风格

代码应优先保证方法语义正确，其次才是最小改动。

允许重构接口。

如果现有函数接口无法表达方案，应修改接口，而不是把方案压缩进旧接口。

禁止为了少改代码牺牲方案定义。

---

## 17. 默认结论规范

实验结果解释必须谨慎。

如果实现中使用 proxy、fallback 或不完整模块，结论必须写成：

```text
当前结果验证的是 proxy / partial implementation，不能证明完整方案有效或无效。
```

只有当所有硬约束满足时，才能声称：

```text
该实验验证了 Diagnosis-guided REGOR-S2 的完整实现。
```
