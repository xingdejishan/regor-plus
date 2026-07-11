# 历史条件对应搜索（方案 C）

`memory_graph` 是独立的纯点云配准入口，不读取射线、RGB-D 或 R1 输出。输入只包括源/目标关键点及其描述子；GT 仅用于实验审计，不进入候选表、子图搜索、位姿估计或记忆更新。

配置位于每个实验 JSON 的顶层 `memory_graph` 节，由 `memory_graph_config.py:MemoryGraphConfig` 严格校验；它与 `iterative_ray` 配置完全分离，缺失或未知字段都会立即报错。

## 方法对应

| 方案模块 | 实现 |
| --- | --- |
| top-K 候选表与 Beta 后验 `R_t` | `correspondence_memory.py:CorrespondenceMemory` 的 `src_indices/tgt_indices/alpha/beta` |
| 几何骨架与共成/共败图 `G_t` | 同文件的 CSR `row_ptr/edge_cols/static_geometry/edge_success/edge_failure` |
| SE(3) 盆地 `B_t` | 同文件的 LRU 哈希 `basins` |
| PROSAC 种子和 `F_t(S)` 子图扩展 | `memory_guided_registration.py:MemoryGuidedRegistration._prosac_seed` 与 `CorrespondenceMemory.expand_support` |
| 加权 SVD 与 TLS | `MemoryGuidedRegistration._weighted_rigid/_robust_refine` |
| `J_t(T)`、单轨记忆更新、联合停止 | `MemoryGuidedRegistration._pose_score/run` |
| 纯点云实验和审计 | `test_3DLoMatch.py:run_memory_graph_experiment` |

支持集采用贪心最大化：

`sum(u_a) + lambda_edge * mean(omega_ab) - lambda_degeneracy * D_phi(S)`。

`D_phi` 使用源支持点协方差特征值的归一化熵和体素覆盖率；低于相应阈值的支持集不会进入 SVD。候选位姿用全候选表上的截断最小二乘细化；只有每轮模型分数最高的一个候选写入三层记忆。

当前没有 proxy 或 fallback。输出中的 `topk_candidate_fmr`、`topk_candidate_inlier_ratio`、`topk_candidate_inlier_count` 分别对应固定 top-K 表的 FMR、IP、IN；方案 C 不再生对应，因此没有把任何值伪称为 INR。

## 参数报告

| 参数 | 作用 | 默认值 | 取值范围 | 设定依据 / 消融 | 文件 / 函数 |
| --- | --- | --- | --- | --- | --- |
| `memory_topk` | 每个源点保留的目标候选数 K | 20 | >=3 | 方案默认量级；消融 K | `memory_graph_config.py`, `_build_candidates` |
| `memory_graph_neighbors` | 每个候选的静态 CSR 邻边数 L | 32 | >=1 | 控制图稀疏性；消融 L | `memory_graph_config.py`, `_build_relation_graph` |
| `memory_sigma_g`, `memory_tau_g` | 几何相容核带宽和建边阈值 | 0.10, 0.60 | >0, (0,1] | 与关键点尺度和内点阈值匹配；消融两者 | 配置, `_build_relation_graph` |
| `memory_alpha0`, `memory_beta0`, `memory_descriptor_prior` | Beta 初始先验 | 1, 1, 2 | >0, >0, >=0 | 弱描述子先验，不替代后续证据 | 配置, `__init__` |
| `memory_forgetting` | 后验和边计数遗忘因子 rho | 0.95 | (0,1] | 防止早期误判永久锁死；必须消融 | 配置, `update_pair_posterior/update_relation_graph` |
| `memory_eta_positive`, `memory_eta_negative` | 单对应正/负更新步长 | 1, 1 | >=0 | 令一次强证据约等于一个 Beta 计数 | 配置, `update_pair_posterior` |
| `memory_eta_edge_positive`, `memory_eta_edge_negative` | 共成/共败边更新步长 | 1, 1 | >=0 | 对称的边证据累积；消融 | 配置, `update_relation_graph` |
| `memory_evidence_cap`, `memory_compress_epsilon` | 证据饱和和弱边剪枝阈值 | 100, 0.01 | >0, >=0 | 保证记忆有界并清除近抵消边 | 配置, `compress` |
| `memory_lambda_descriptor`, `memory_lambda_reliability`, `memory_lambda_graph` | 种子分数的描述子、后验、图中心性权重 | 1, 1, 1 | >=0 | 对应 `u_a`; 三层消融会关闭后两项 | 配置, `rank` |
| `memory_lambda_edge`, `memory_lambda_degeneracy` | `F_t(S)` 的边一致性和退化惩罚权重 | 1, 1 | >=0 | 显式控制子图而非事后过滤；消融 | 配置, `support_objective` |
| `memory_lambda_edge_success`, `memory_lambda_edge_failure` | 动态边正/负计数权重 | 1, 1 | >=0 | 对应 `omega_ab` 的成功与失败项 | 配置, `edge_weight` |
| `memory_support_min`, `memory_support_max`, `memory_support_trial_count` | 最小/最大支持集与每步精评候选数 | 3, 32, 8 | >=3, >=最小值, >=1 | 3 点保证 6DoF 最小求解；试探数控制运行时间 | 配置, `expand_support` |
| `memory_min_eigen_entropy`, `memory_min_coverage`, `memory_coverage_voxel_size` | 结构熵、覆盖率阈值及体素尺度 | 0.20, 0.20, 0.10 m | [0,1], [0,1], >0 | 拒绝线/面退化和过度局部支持；消融 | 配置, `support_signature/degeneracy_penalty` |
| `memory_hypotheses_per_round`, `memory_max_rounds` | 每轮假设数 H 和最大轮数 | 24, 8 | >=1, >=1 | 与方案默认预算一致；必须消融轮数 | 配置, `run` |
| `memory_inlier_threshold`, `memory_tls_threshold`, `memory_tls_iters` | 验证阈值、TLS 截断和迭代次数 | 0.10 m, 0.10 m, 3 | >0, >0, >=0 | 与 3DMatch 评价尺度对齐；消融 TLS | 配置, `_robust_refine/_verify` |
| `memory_lambda_inlier`, `memory_lambda_error`, `memory_lambda_coverage` | `J_t(T)` 的一致集、残差、覆盖权重 | 1, 1, 1 | >=0 | 避免只按内点数选姿态 | 配置, `_pose_score` |
| `memory_lambda_basin_negative`, `memory_lambda_basin_positive` | 失败盆地惩罚和成功盆地奖励 | 1, 1 | >=0 | 仅对结构签名相近的同桶候选生效 | 配置, `basin_bonus` |
| `memory_prosac_initial_fraction`, `memory_prosac_growth` | 渐进采样前缀初值和扩张率 | 0.20, 0.15 | (0,1], >=0 | 先利用高分候选，停滞时扩展搜索域 | 配置, `_prosac_seed` |
| `memory_basin_rotation_deg`, `memory_basin_translation`, `memory_basin_max` | SE(3) 桶分辨率与 LRU 容量 | 5 deg, 0.10 m, 256 | >0, >0, >=1 | 平衡重复检测和误合并 | 配置, `basin_key/update_basin` |
| `memory_basin_signature_momentum`, `memory_basin_signature_similarity` | 桶签名 EMA 与重复判断阈值 | 0.90, 0.90 | [0,1), [0,1] | 防止仅因位姿接近而过度抑制 | 配置, `update_basin/basin_bonus` |
| `memory_patience`, `memory_score_epsilon`, `memory_pose_epsilon_rotation_deg`, `memory_pose_epsilon_translation_multiplier`, `memory_novelty_threshold`, `memory_delta_threshold` | 联合停机规则 | 3, 0.005, 0.2 deg, 0.5, 0.1, 0.01 | 正数或 [0,1] | 对应方案 C 的分数、位姿、新颖性、记忆增量判据 | 配置, `run` |
| `memory_strong_stop_min_inliers`, `memory_strong_stop_inlier_fraction`, `memory_strong_stop_error_ratio`, `memory_strong_stop_coverage` | 强停止的数量、比例、误差、覆盖阈值 | 30, 0.02, 0.75, 0.20 | >=3, (0,1], >0, [0,1] | 避免已高置信时无意义迭代 | 配置, `_strong_stop` |
| `memory_use_reliability`, `memory_use_relation_history`, `memory_use_basin` | 三层记忆消融开关 | true | bool | `memory_use_relation_history` 只关闭历史边计数；静态几何骨架始终保留以生成可解的支持集 | 配置, `scripts/run_memory_graph_ablations.py` |

## 验证清单

- [x] 种子来自固定 top-K 候选表的 PROSAC 前缀，而非 R1 correspondence。
- [x] 子图扩展使用动态共成/共败边和结构退化项。
- [x] 姿态用加权 SVD 后在完整候选表做 TLS 验证。
- [x] 每轮只有当前最优候选更新三层记忆。
- [x] 位姿负记忆同时要求相同 SE(3) 桶和相近结构签名。
- [x] 输出始终是 best-so-far，分数单调不降。
- [x] 停止同时考虑 patience、分数、位姿变化、新颖性和记忆增量。
- [x] `tests/test_correspondence_memory.py` 覆盖后验、图、盆地和完整 6DoF 合成配准。
