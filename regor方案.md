# Diagnosis-guided REGOR-S2：最终方案

## 0. 核心目标

本方案把原始 REGOR 的固定 progressive regeneration 改成 **诊断驱动的主动再生成**。

核心思想：

> Round 1 先用 GMM 生成候选匹配与候选位姿；  
> 再用 FSV 判断当前位姿是否物理可信，用 pose-conditioned supported ROC 判断当前重叠区支持是否充分；  
> 若当前解已经可信，则早停；若不可信或支持不足，则进入 Round 2 主动再生成。

最终主张：

> REGOR 负责生成候选匹配；FSV 与 pose-conditioned supported ROC 负责判断是否需要继续生成；包容式 overlap prior 负责决定第二轮去哪生成。

---

## 1. 方法动机

原始 REGOR 的流程可以概括为：

```text
seed correspondences
→ local grouping
→ GMM regeneration
→ CTC local correction
→ global refinement
→ next round
```

但本地 3DLoMatch 中间结果显示：

1. **GMM 有明显作用**：它能从 seed 中生成更多候选内点。
2. **CTC 作用很小**：从 GMM 到 CTC，Count、Precision、Inliers 几乎不变。
3. **固定两轮缺乏充分理论依据**：它不像 RANSAC 那样有概率迭代公式，也不像 one-shot 方法那样结构极简。

因此，本方案删除 CTC，保留 GMM，并引入 diagnosis-guided active regeneration。

---

## 2. 输入

每个点云对需要：

- 源点云：`P`
- 目标点云：`Q`
- 初始对应关系：`C0`
- 点特征：如 FPFH / FCGF / Predator / GeoTransformer feature
- RGB-D 相机轨迹或可构建 free-space map 的传感器信息
- 前端预测重叠度：`Overlap_pred`
- 可选：深度图、相机内参、TSDF / occupancy / free-space volume

其中，`Overlap_pred` 可以来自 Predator 或其他 overlap prediction module。

---

## 3. Round 1：基础 GMM 生成

### 3.1 删除 CTC

原流程：

```text
GMM → CTC → global
```

改为：

```text
GMM → candidate pose generation → diagnosis
```

删除 CTC 的原因：

- 本地结果显示 CTC 对匹配质量提升极小。
- CTC 增加复杂度，但没有稳定收益。
- 后续由 FSV 与覆盖诊断控制是否继续生成。

### 3.2 GMM regeneration

从初始对应关系 `C0` 出发：

```text
C0
→ local grouping
→ GMM regeneration
→ C_R1
```

其中 `C_R1` 是 Round 1 生成的候选匹配集合。

### 3.3 Top-K 候选位姿

从 `C_R1` 生成多个候选位姿：

```text
C_R1
→ generate {T1, T2, ..., TK}
→ lightweight refinement
→ select T_best
```

说明：

- `Top-K` 不是第二轮结果，而是 Round 1 的候选池备份。
- `T_best` 是 Round 1 中分数最高的 refined candidate。
- Top-K 用于在当前 `T_best` 不可信时构建 reset search prior。

---

## 4. 诊断信号

### 4.1 FSV：自由空间违背率

记：

```math
FSV(T)
```

含义：

> 将源点云通过位姿 `T` 变换到目标坐标系后，落入目标传感器已观测自由空间的比例。

性质：

- 越高越差。
- 高 FSV 表示当前位姿物理上不自洽。
- 低 FSV 只说明“没有明显穿模”，不保证位姿一定正确。

FSV 主要回答：

```text
当前 T 是否物理上可疑？
```

### 4.2 AOC：绝对覆盖率

先定义绝对覆盖率：

```math
AOC(T)
=
\frac{
|\mathcal{V}_{inlier}(T)|
}{
|\mathcal{V}_{source}|
}
```

其中：

- `V_inlier(T)`：在当前位姿下被内点覆盖的 source voxel 集合。
- `V_source`：source 点云的全部 voxel 集合。

AOC 的问题：

> 在 3DLoMatch 中真实 overlap 只有 10%–30%，所以 AOC 天然偏低，不能直接用于早停判断。

### 4.3 ROC_bar：pose-conditioned supported overlap

`ROC_bar` 不应该只是 AOC 的归一化覆盖率。
在 redkitchen 这类重复室内结构里，错误位姿也可能把 source 点投到墙面、柜面、桌面附近；如果只问“投过去附近有没有 target 点”，会把错误位姿误判为覆盖充分。

因此 `ROC_bar` 必须定义为 pose-conditioned supported overlap：

```math
\bar{ROC}(T)
=
\operatorname{clip}
\left(
\frac{
\sum_i
O_{\text{pred}}(p_i)\cdot
\mathbb{I}
\left[
d(Tp_i, P_t)<\tau_d
\right]\cdot
s_i(T)
}{
\sum_i O_{\text{pred}}(p_i)+\epsilon
},
0,
1
\right)
```

其中：

- `O_pred(p_i)`：外部 overlap predictor 对 source 点的预测重叠先验。
- `d(Tp_i, P_t)`：source 点经候选位姿变换后到 target 点云的最近距离。
- `tau_d`：target surface proximity 阈值，默认等于 `tau_overlap`。
- `s_i(T)`：位姿条件正向几何支持，取值 `[0,1]`。

`s_i(T)` 不能恒等于 1。它至少应包含一种正向几何支持，例如：

```text
对应点残差是否小
局部邻域是否有 target surface support
法向是否一致
是否属于 Round1 高置信 correspondence 支持区域
是否被多个局部点一致支持
```

当前实现采用：

```math
s_i(T)
=
\operatorname{NoisyOR}
\left(
w_{r1}\cdot S_{r1}(p_i),
w_{local}\cdot S_{local}(p_i)
\right)
```

其中：

- `S_r1(p_i)`：`p_i` 是否落在 Round1 refined pose 的高置信 correspondence source 支持区域附近。
- `S_local(p_i)`：`p_i` 附近是否有多个 source 点也在当前位姿下获得 target surface support。
- 当前版本没有使用 normal consistency；如果加入可靠法向，应作为额外支持项进入 `s_i(T)`。

旧的：

```math
\frac{AOC(T)}
{Overlap_{\text{pred}}+\epsilon}
```

只能作为 `coverage_ratio_proxy` 诊断量，不能再用于 early stop 或 `O(p)` 的扩张系数。

性质：

- 越高越好。
- 它不是绝对覆盖率，也不是简单归一化覆盖率。
- 它用于判断当前位姿下 predicted-overlap source 区域是否被 target surface 和当前 correspondence / 局部几何共同支持。

---

## 5. 早停判别式

早停条件：

```math
FSV(T_{\text{best}}) < \tau_{fsv}
\quad \text{and} \quad
\bar{ROC}(T_{\text{best}}) > \tau_{\rho}
```

若满足：

```text
skip Round 2
enter mandatory final refinement with C_R1
```

否则：

```text
enter Round 2
```

本方案采用：

```text
early stop + mandatory refinement
```

即 early stop 只表示“停止主动再生成 / 不进入 Round 2”，不表示直接输出 `T_best`。  
满足 early stop 后仍必须进入最终估计模块：

```text
density filtering
→ voxel-balanced weighting
→ robust weighted SVD 或 mini-RANSAC
→ return T_refined
```

解释：

- FSV 低：当前位姿没有明显违反自由空间。
- `ROC_bar` 高：当前位姿下 predicted-overlap 区域已经获得 target surface 与 correspondence / 局部几何支持。
- 两者同时满足，说明没有必要进入第二轮主动生成，但仍需要最终 refinement 稳定位姿。

这把原始 REGOR 的固定两轮改成了按需迭代。

---

## 6. Round 2 搜索先验：包容式 overlap prior

Round 2 的关键不是盲目扩大搜索，而是构建一个 soft overlap prior：

```math
O(p) \in [0,1]
```

含义：

> source 点 `p` 在 Round 2 中参与主动再生成的软权重。

### 6.1 位姿可信度

定义当前位姿可信度：

```math
v(T)=\exp(-\alpha \cdot FSV(T))
```

性质：

- `FSV` 低时，`v(T)` 接近 1，说明当前位姿物理可信。
- `FSV` 高时，`v(T)` 接近 0，说明当前位姿不可信。

`v(T)` 决定是否信任 `T_best`。

### 6.2 局部扩张区域

由 `T_best` 得到当前估计重叠区：

```math
O_{\text{best}}(p)
```

高效实现方式：

```math
O_{\text{best}}(p)
=
\mathbb{I}
\left(
\min_{q\in Q}
\|T_{\text{best}}p-q\|
<
\tau_{\text{overlap}}
\right)
```

然后定义纯净的局部扩张区域：

```math
O_{\text{broad}}(p)
=
\operatorname{dilate}
\left(
O_{\text{best}}(p)
\right)
```

注意：

> `O_broad` 只负责在当前可信位姿周围做局部扩张，不包含 Top-K union。  
> 这样可以避免把其他候选中的错误区域走私进来。

### 6.3 重置搜索区域

当 `FSV` 高、当前 `T_best` 不可信时，使用重置搜索区域：

```math
O_{\text{reset}}(p)
```

推荐定义：

1. 优先使用 Top-K 中 FSV 较低候选的 overlap union；
2. 对候选 union 做空间膨胀，避免只押注在单个幸存候选的狭窄区域；
3. 如果 Top-K 都不可信，则退回全局高 matchability 区域；
4. 即使存在 Top-K 幸存候选，也保留一个弱全局探索项，防止 reset mask 过窄。

可写为：

```math
O_{\text{surv}}(p)
=
\operatorname{Union}
\left(
\{O_k(p)\mid FSV(T_k)<\tau_{k}\}
\right)
```

```math
O_{\text{reset}}(p)
=
\operatorname{NoisyOR}
\left(
\operatorname{dilate}(O_{\text{surv}}(p)),
\gamma\cdot O_{\text{global-matchable}}(p)
\right)
```

其中：

```math
\gamma \in [0,1]
```

是弱全局探索权重。建议第一版取：

```text
gamma ∈ {0.05, 0.10, 0.15}
```

`gamma` 不宜过大，否则 reset 会重新退化成全局乱搜。若 `O_surv` 为空，则直接使用：

```math
O_{\text{reset}}(p)=O_{\text{global-matchable}}(p)
```

这样可以避免当 Top-K 中只有一个候选通过 FSV 检查时，Round 2 被锁死在单一候选的狭窄 overlap 区域内。

### 6.4 包容式 overlap prior

最终：

```math
O(p)
=
v(T)\cdot
\operatorname{NoisyOR}
\Big(
O_{\text{best}}(p),
(1-\bar{ROC}(T))\cdot O_{\text{broad}}(p)
\Big)
+
(1-v(T))\cdot O_{\text{reset}}(p)
```

其中：

```math
\operatorname{NoisyOR}(a,b)=1-(1-a)(1-b)
```

为什么用 Noisy-OR，而不是凸组合？

因为当前位姿可信但覆盖不足时，我们的目标是：

> 保留已知正确区域，同时向外扩张。

凸组合会让 `O_best` 和 `O_broad` 竞争权重，导致覆盖率低时削弱已知正确区域。  
Noisy-OR 是包容式合并，不会压低原有可信区域。

行为解释：

- `FSV` 低、`ROC_bar` 高：主要使用 `O_best`，可早停。
- `FSV` 低、`ROC_bar` 低：保留 `O_best`，同时向 `dilate(O_best)` 扩张。
- `FSV` 高：不信任 `O_best`，由 `O_reset` 接管。

---

## 7. 自适应几何力臂

### 7.1 问题

原先几何力臂可定义为：

```math
L(p)=\|p-c_1\|
```

其中 `c1` 是 Round 1 内点集中心。

但当 `FSV` 高、`v(T)→0` 时，Round 1 内点集可能本身就是错误匹配。  
此时继续围绕 `c1` 计算力臂，会产生伪力臂陷阱。

### 7.2 自适应参考中心

定义：

```math
c_{\text{ref}}
=
v(T)\cdot c_1
+
(1-v(T))\cdot c_{\text{global}}
```

其中：

- `c1`：Round 1 内点中心。
- `c_global`：source 点云全局几何中心。

性质：

- `v(T)→1`：当前位姿可信，围绕当前内点中心计算力臂。
- `v(T)→0`：当前位姿不可信，退回全局几何中心。

### 7.3 几何力臂定义：只奖励远端点，不惩罚中心点

原始距离型定义：

```math
L(p|c_{\text{ref}})
=
\operatorname{clip}
\left(
\frac{\|p-c_{\text{ref}}\|}{r_{\text{src}}},
0,
1
\right)
```

会在 `p` 接近 `c_ref` 时得到 `L≈0`。  
由于后续 GMM similarity 中使用 `log L`，这会在点云中心产生巨大负惩罚，造成“对数空洞”或“甜甜圈效应”：中心区域的优质匹配会被错误压制。

因此，最终采用无惩罚基线的几何力臂：

```math
d(p|c_{\text{ref}})
=
\operatorname{clip}
\left(
\frac{\|p-c_{\text{ref}}\|}{r_{\text{src}}},
0,
1
\right)
```

```math
L(p|c_{\text{ref}})
=
1+\eta_L\cdot d(p|c_{\text{ref}})
```

其中：

```math
\eta_L \ge 0
```

`r_src` 是 source 点云尺度归一化项。

这个定义的含义是：

- 中心点：`L=1`，`log L=0`，不受惩罚；
- 远端点：`L=1+eta_L`，`log L>0`，获得温和奖励。

也就是说，几何力臂只用于奖励能提供更大旋转约束的远端点，而不会误杀对平移估计有价值的中心匹配。

建议第一版测试：

```text
eta_L ∈ {0, 0.2, 0.5, 1.0}
```

其中 `eta_L=0` 表示关闭几何力臂项。

注意：

> `L` 只表示几何约束价值，不单独决定 seed。  
> 它必须和 `O(p)`、`M(p,q)`、`U_s(p)`、`U_t(q)` 一起使用。

---

## 8. Round 2：双向主动 GMM

最终 Round 2 的匹配权重：

```math
w(p,q)
=
O(p)
\cdot
M(p,q)
\cdot
U_s(p)
\cdot
U_t(q)
\cdot
L(p|c_{\text{ref}})
```

各项含义：

- `O(p)`：source 点是否位于 Round 2 应该探索的 soft overlap prior 内。
- `M(p,q)`：匹配可行性，来自 feature similarity 或 GMM confidence。
- `U_s(p)`：source 端欠支持程度。
- `U_t(q)`：target 端欠支持程度。
- `L(p|c_ref)`：几何力臂价值。

### 8.1 Source 端欠支持

```math
U_s(p)=\frac{1}{1+v(T)\cdot\rho^{source}_{R1}(p)}
```

其中 `rho_R1^source(p)` 是 Round 1 source 内点在 `p` 附近的密度。

作用：

> 只有当当前位姿可信时，才相信 Round 1 的 source 已覆盖区域统计。  
> 当 `FSV` 高、`v(T)→0` 时，Round 1 的“内点区域”可能本身是错误的，此时自动释放 source 端抑制，避免误杀真实重叠区。

### 8.2 Target 端欠支持

```math
U_t(q)=\frac{1}{1+v(T)\cdot\rho^{target}_{R1}(q)}
```

作用：

> 只有当当前位姿可信时，才抑制 Round 1 已经过度使用的 target 区域。  
> 当 `FSV` 高、`v(T)→0` 时，Round 1 的 target 连线可能是错误的，此时自动释放 target 端抑制，避免把真实重叠区误杀。

这是双向主动生成的关键。

### 8.3 GMM similarity 调制

如果原始 GMM 相似度矩阵为：

```math
S_{\text{feat}}(p,q)
```

则加入多维自适应诊断与几何先验后的调制公式为：

```math
S'(p,q)
=
S_{\text{feat}}(p,q)
+
\lambda
\left[
\log \tilde{O}(p)
+
\log \tilde{U}_s(p)
+
\log \tilde{U}_t(q)
+
\log \tilde{L}(p|c_{\text{ref}})
\right]
```

**【数值安全与边界定义（核心修正）】**

为了保证在 PyTorch 矩阵并行运行时，各物理量在对数映射（Log-domain）下尺度自洽，且不发生特征吞噬或机制失效，必须采用**分类域有界截断**：

```math
\tilde{x} = \operatorname{clip}(x, \epsilon, 1.0) \quad \text{for } x \in \{O, U_s, U_t\}
```

```math
\tilde{L}(p|c_{\text{ref}}) = \max(L(p|c_{\text{ref}}), \epsilon)
```

**设计原理与避坑说明：**

1. **温度调节因子 $\lambda$**：用于平衡对数先验项与特征相似度 $S_{\text{feat}}$ 的数值尺度（建议第一版实验取 0.1 左右）。如果不加该系数，$\log(1e-4) \approx -9.2$ 的庞大变幅会直接把神经网络特征抹杀，导致软调制退化为硬截断（Hard Mask）。

2. **力臂截断放开（规避甜甜圈效应）**：由于 $O, U_s, U_t$ 的理论取值域在 $[0, 1]$ 之间，使用上界为 1 的 clip 是完全正确的；**但由于我们在第 7.3 节中将几何力臂重构为了 $L = 1 + \eta_L \cdot d \ge 1$，其取值天然大于等于 1**。

   - **如果错误地对 $L$ 沿用上界为 1 的 clip 算子，会导致所有远端高价值点的奖励信息全部被截断扼杀（恒等于 1，$\log 1 = 0$），导致几何力臂项在运行时完全失效。**
   - 因此，$\tilde{L}$ 必须独立出来使用 max(L, epsilon) 确保对数安全，从而完美保留远端点的正向奖励，且在中心点处保持 $\log(1.0)=0$ 的零惩罚基线。

---

## 9. 合并与最终估计

Round 2 得到：

```text
C_R2
```

合并：

```text
C = C_R1 ∪ C_R2
```

不要直接做普通 SVD。  
因为 Round 2 仍可能包含 outliers。

推荐：

```text
C
→ density filtering
→ voxel-balanced weighting
→ robust weighted SVD 或 mini-RANSAC
→ T_final
```

注意顺序：

> 先过滤孤立外点，再做体素均衡。  
> 不能直接使用 `1 / n_voxel`，否则孤立外点所在体素 `n_voxel=1` 会获得满权重，反而被放大。

最终权重可写为：

```math
w_i^{final}
=
w_{\text{match},i}
\cdot
w_{\text{density},i}
\cdot
w_{\text{balance},i}
\cdot
w_{\text{robust},i}
```

其中密度门控可以写成：

```math
w_{\text{density},i}
=
\mathbb{I}
\left(
n_{\text{voxel}(i)}\ge n_{\min}
\right)
```

或者使用邻域半径密度：

```math
w_{\text{density},i}
=
\mathbb{I}
\left(
\rho_{\text{neighbor}}(i)\ge\tau_{\rho}
\right)
```

通过密度检查后，再做空间均衡：

```math
w_{\text{balance},i}
=
\frac{1}
{\min(n_{\text{voxel}(i)},n_{\max})}
```

其中 `n_max` 用于避免极密集区域被过度压低。  
在 3DLoMatch 的常见体素设置下，建议第一版取：

```text
n_max ∈ {5, 10}
```

这样既能打破超密集匹配块的支配地位，也不会让高纯度密集内点完全失去话语权。

鲁棒核可使用 Geman-McClure 或类似形式：

```math
w_{\text{robust},i}
=
\frac{c^2}{(r_i^2+c^2)^2}
```

因此最终估计阶段的原则是：

> Density filter removes isolated tail outliers;  
> voxel balancing prevents dense regions from dominating;  
> robust kernel reduces residual outlier influence.

---

## 10. 最终完整流程

```text
Input:
P, Q, C0, features, RGB-D trajectory/free-space map, Overlap_pred

Round 1:
1. C0 → local grouping
2. GMM regeneration → C_R1
3. Generate Top-K candidates {T1, ..., TK}
4. Lightweight refinement
5. Select T_best

Diagnosis:
6. Compute FSV(T_best)
7. Compute AOC(T_best) as diagnostic coverage_ratio_proxy only
8. Compute pose-conditioned supported ROC_bar(T_best)

Early Stop:
9. If FSV(T_best) < τ_fsv and ROC_bar(T_best) > τ_ρ:
       skip Round 2
       go to Final with C_R1 and T_best

Round 2:
10. Compute v(T)=exp(-α·FSV(T))
11. Compute O_best
12. Compute O_broad=dilate(O_best)
13. Compute O_reset
14. Compute O(p) using Noisy-OR prior
15. Compute c_ref
16. Compute w(p,q)=O·M·U_s·U_t·L
17. Guided GMM regeneration → C_R2

Final:
18. Merge C_R1 ∪ C_R2
    If early stop: use C_R1 only
19. Density filtering
20. Voxel-balanced weighting
21. Robust weighted SVD / mini-RANSAC
22. Output T_refined / T_final
23. Optional: final FSV-based failure detection
```

---

## 11. 与原始 REGOR 的区别

| 模块 | 原始 REGOR | 本方案 |
|---|---|---|
| 迭代方式 | 固定 progressive rounds | Diagnosis-guided 按需触发 |
| CTC | 保留 | 删除 |
| 第二轮生成 | 固定局部再生成 | FSV + 覆盖诊断引导的主动再生成 |
| 搜索区域 | 由上一轮对应关系决定 | 包容式 overlap prior 控制 |
| 低重叠处理 | 隐式处理 | 使用 `ROC_bar` 避免低重叠误判 |
| 错误位姿处理 | 可能继续沿错误区域扩增 | FSV 高时进入 reset prior |
| 目标端约束 | 主要 source-side | source-target 双向欠支持 |
| 最终估计 | 常规估计 | density filtering + voxel-balanced robust estimation |

---

## 12. 需要验证的关键实验

### 12.1 CTC 删除实验

对比：

```text
REGOR 原版
REGOR w/o CTC
本方案 w/o Round 2
本方案 full
```

目标：

> 验证 CTC 是否确实可删除，以及新的主动再生成是否带来提升。

### 12.2 早停有效性实验

对比：

```text
固定两轮
只用 FSV 早停
FSV + ROC_bar 早停
```

指标：

- RR
- RE / TE
- 平均运行时间
- 进入 Round 2 的样本比例
- 误早停率

### 12.3 overlap prior 消融

对比：

```text
Convex combination
Max union
Noisy-OR union
Noisy-OR + reset
```

目标：

> 验证包容式 overlap prior 是否优于零和凸组合。

### 12.4 target-side 欠支持消融

对比：

```text
source-only U_s
source + target U_s·U_t
```

目标：

> 验证 target 端软抑制是否能减少目标端匹配扎堆。

### 12.5 自适应力臂消融

对比参考中心：

```text
L(p|c1)
L(p|c_global)
L(p|c_ref)
```

目标：

> 验证当前位姿不可信时，自适应力臂是否比固定 c1 更稳。

同时消融力臂强度：

```text
eta_L = 0
eta_L = 0.2
eta_L = 0.5
eta_L = 1.0
```

其中 `eta_L=0` 表示关闭几何力臂项。  
该实验用于验证“只奖励远端点、不惩罚中心点”的设计是否稳定，并避免原始 `L∈[0,1]` 带来的对数空洞。


### 12.6 先验调制温度消融

对比：

```text
lambda = 0
lambda = 0.05
lambda = 0.1
lambda = 0.2
lambda = 0.5
```

目标：

> 验证 log-prior 是否需要温度控制，并避免先验项吞噬特征相似度。

### 12.7 可信度调制欠支持消融

对比：

```text
U_s = 1 / (1 + rho_s)
U_t = 1 / (1 + rho_t)
U_s = 1 / (1 + v·rho_s)
U_t = 1 / (1 + v·rho_t)
```

目标：

> 验证当当前位姿不可信时，释放 source / target 欠支持抑制是否能避免误杀真实重叠区。

### 12.8 Reset prior 消融

对比：

```text
Top-K survivor union
dilate(Top-K survivor union)
NoisyOR(dilate(Top-K survivor union), gamma·global-matchable)
```

目标：

> 验证 reset 搜索区域是否需要膨胀与弱全局探索，避免押注单个幸存候选。

### 12.9 最终估计鲁棒性消融

对比：

```text
voxel-balanced only
density filtering + voxel-balanced
density filtering + voxel-balanced + robust kernel
mini-RANSAC
```

目标：

> 验证密度过滤是否能抑制孤立外点被 voxel balance 放大的问题。

---

## 13. 可能失败场景

1. **FSV 地图不可靠**  
   相机 pose 错误、深度空洞、动态物体会影响自由空间判断。

2. **Overlap_pred 误差过大**  
   如果 predicted overlap 明显错误，`ROC_bar` 会失真。

3. **Top-K 候选整体失败**  
   此时 `O_reset` 只能退回全局高 matchability 区域，提升有限。

4. **重复结构不违反自由空间**  
   某些错误位姿可能 FSV 低，但仍然是错误对齐。

5. **Round 2 GMM 生成外点过多**  
   需要最终 robust weighted SVD 或 mini-RANSAC 兜底。

6. **先验调制温度不合适**  
   如果 `lambda` 过大，log-prior 会吞噬特征相似度，退化成 hard mask；如果过小，几何先验几乎不起作用。

7. **Reset prior 过窄**  
   如果 Top-K 中只有一个候选通过 FSV，且没有 dilation 或 global fallback，Round 2 可能被锁死在错误区域。

8. **体素均衡放大孤立外点**  
   如果没有 density filtering，孤立 outlier 会因为 `n_voxel=1` 获得过高权重，拉偏最终 SVD。


9. **几何力臂过强**  
   如果 `eta_L` 过大，远端点会被过度奖励，可能放大远端外点的旋转力矩。需要与 `O(p)`、`M(p,q)` 和 robust final estimation 一起约束。

---

## 14. 一句话总结

本方案将 REGOR 的固定两轮 correspondence regeneration 改为 diagnosis-guided active regeneration：用 FSV 判断当前位姿是否物理可信，用 pose-conditioned supported ROC 判断当前重叠区支持是否充分，并通过包容式 overlap prior、无惩罚基线的自适应几何力臂和 source-target 双向欠支持权重，引导第二轮 GMM 在可靠区域中主动补充高价值匹配。
