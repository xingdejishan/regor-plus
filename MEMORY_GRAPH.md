# Historical Conditional Correspondence Search

`memory_graph` uses point clouds, descriptors, and a required fixed R1 pose cache. It does not read RGB-D or ray evidence. Ground truth is used only after inference for audits.

Each JSON configuration has a strict top-level `memory_graph` section. Its schema is `memory_graph_config.py:MemoryGraphConfig`; ray and memory settings are intentionally separate.

## Archives and evaluation

- The cached R1 pose is a permanent round-0 raw parent.
- Its cached correspondences are mapped to fixed top-K candidate IDs before round 1; their posterior, relation edges, and non-improvement basin state initialize the search memory.
- Each cache record carries and validates pair ID, descriptor, seed, R1 settings, inlier threshold, and a deterministic configuration fingerprint.
- Every SVD pose is independently verified and permanently retained as a raw parent.
- TLS only creates a refinement child. A child enters the post-refinement archive only if it stays in the configured SE(3) trust region and improves evidence score over its parent.
- Selection is over the union of raw parents and accepted children.
- `round_logs.csv` reports prefix-limited raw Oracle, post-refinement Oracle, and selected metrics. It separately reports the R1-failure subset and newly repaired failures. The post-refinement Oracle is asserted to be no worse than the raw Oracle.

## Method mapping

| Scheme module | Implementation |
| --- | --- |
| Candidate Beta posterior `R_t` | `correspondence_memory.py:CorrespondenceMemory.alpha/beta` |
| Static compatibility plus dynamic co-success/co-failure graph `G_t` | CSR `row_ptr/edge_cols/static_geometry/edge_success/edge_failure` |
| SE(3) basin cache `B_t` | LRU `basins`, failure-signature pre-search cache |
| PROSAC and greedy `F_t(S)` | `MemoryGuidedRegistration._prosac_seed`, `CorrespondenceMemory.expand_support` |
| Raw parent / refinement child archives | `MemoryGuidedRegistration.run` |
| Fixed-R1 experiment audit | `test_3DLoMatch.py:run_memory_graph_experiment` |

The structural signature is

`[mean TopK-source entropy, lambda1/lambda2, lambda1/lambda3, |V(S)|/|V(P)|, cross-group compatibility]`.

The coverage term is deliberately `|V(S)| / |V(P)|`. The inverse would grow for small local supports and is not a coverage ratio. Cross-group compatibility is the mean static geometric compatibility over support-pair correspondences assigned to different source spatial groups.

## Parameter report

| Parameters | Purpose | Defaults | Valid range | Rationale / ablation | Location |
| --- | --- | --- | --- | --- | --- |
| `memory_topk`, `memory_graph_neighbors`, `memory_sigma_g`, `memory_tau_g` | Candidate table K and sparse geometric graph L | 20, 32, 0.10, 0.60 | K>=3, L>=1, sigma>0, tau in (0,1] | Fixed across all ablations; ablate K and L | config, `CorrespondenceMemory` |
| `memory_alpha0`, `memory_beta0`, `memory_descriptor_prior`, `memory_forgetting` | Beta initialization and forgetting | 1, 1, 2, 0.95 | positive, positive, >=0, (0,1] | Allows early descriptor bias to be corrected; ablate forgetting | config, posterior update |
| `memory_eta_positive`, `memory_eta_negative`, `memory_eta_edge_positive`, `memory_eta_edge_negative`, `memory_evidence_cap`, `memory_compress_epsilon` | Posterior/edge evidence update, saturation, compression | 1, 1, 1, 1, 100, 0.01 | steps>=0, cap>0, epsilon>=0 | Bounded dynamic memory | config, update/compress |
| `memory_lambda_descriptor`, `memory_lambda_reliability`, `memory_lambda_graph`, `memory_lambda_edge`, `memory_lambda_degeneracy` | Seed score and `F_t(S)` weights | all 1 | >=0 | Separates node ranking, compatibility, and structural penalty | config, rank/support objective |
| `memory_lambda_edge_success`, `memory_lambda_edge_failure` | Dynamic edge reward/penalty | 1, 1 | >=0 | Implements co-success/co-failure `omega_ab` | config, edge weight |
| `memory_support_min`, `memory_support_max`, `memory_support_trial_count` | SVD minimum, support cap, greedy trial budget | 3, 32, 8 | >=3, >=min, >=1 | 3 is the minimum 6DoF sample; trial count controls cost | config, expansion |
| `memory_signature_entropy_temperature` | Per-source Top-K entropy temperature | 0.10 | >0 | Measures matching ambiguity, not cross-correspondence score spread | config, signature |
| `memory_min_lambda12_ratio`, `memory_min_lambda13_ratio` | Line and plane degeneracy thresholds | 0.10, 0.05 | [0,1] | `lambda1/lambda3` explicitly rejects planar support | config, degeneracy |
| `memory_min_coverage`, `memory_coverage_voxel_size` | Global source coverage threshold and voxel size | 0.002, 0.10 m | [0,1], >0 | 32-point support cannot satisfy a local-normalized 0.2 threshold; ablate voxel and threshold | config, signature |
| `memory_cross_group_voxel_size`, `memory_min_cross_group_agreement` | Source grouping and cross-group agreement | 0.30 m, 0.50 | >0, [0,1] | Rejects one-local-region supports without cross-region rigidity evidence | config, signature |
| `memory_hypotheses_per_round`, `memory_max_rounds`, `memory_prosac_initial_fraction`, `memory_prosac_growth` | Search budget and prefix schedule | 24, 8, 0.20, 0.15 | >=1, >=1, (0,1], >=0 | Required iteration/budget ablation | config, search |
| `memory_max_sampling_attempts`, `memory_fixed_budget_mode` | Uniform valid-raw sampling-attempt cap and fixed-round ablation mode | 96, true | >=H, boolean | Main ablations run H valid raw targets for every T round and disable early stop; exhausted attempts are logged | config, search |
| `memory_inlier_threshold`, `memory_tls_threshold`, `memory_tls_iters` | Verification and TLS | 0.10 m, 0.10 m, 3 | >0, >0, >=0 | Must match evaluation scale | config, refinement |
| `memory_refine_trust_rotation_deg`, `memory_refine_trust_translation`, `memory_refine_min_score_improvement` | Child acceptance trust region and evidence gain | 15 deg, 0.30 m, 0.01 | >0, >0, >=0 | Prevents refinement from replacing a valid raw parent | config, child acceptance |
| `memory_lambda_inlier`, `memory_lambda_error`, `memory_lambda_coverage` | Pose score `J_t` | all 1 | >=0 | Avoids selecting on inlier count alone | config, pose score |
| `memory_lambda_basin_nonimproving`, `memory_lambda_basin_positive`, `memory_basin_presearch_penalty`, `memory_basin_presearch_min_support_overlap` | Post-pose non-improvement score and pre-SVD overlap-constrained penalty | 1, 1, 1, 0.50 | >=0, >=0, >=0, [0,1] | This is non-improvement memory, not failure memory; pre-SVD suppression requires both signature and candidate-support overlap | config, basin/support objective |
| `memory_basin_rotation_deg`, `memory_basin_translation`, `memory_basin_max`, `memory_basin_signature_momentum`, `memory_basin_signature_similarity` | Pose hashing and signature cache | 5 deg, 0.10 m, 256, 0.90, 0.90 | >0, >0, >=1, [0,1), [0,1] | Balances duplicate suppression and false merges | config, basin cache |
| `memory_patience`, `memory_score_epsilon`, `memory_pose_epsilon_rotation_deg`, `memory_pose_epsilon_translation_multiplier`, `memory_novelty_threshold`, `memory_delta_threshold` | Joint stopping | 3, 0.005, 0.2 deg, 0.5, 0.1, 0.01 | non-negative; novelty in [0,1] | Basin-disabled ablations omit novelty from stopping | config, search |
| `memory_strong_stop_min_inliers`, `memory_strong_stop_inlier_fraction`, `memory_strong_stop_error_ratio`, `memory_strong_stop_coverage` | Strong-stop confidence | 30, 0.02, 0.75, 0.20 | >=3, (0,1], >0, [0,1] | Stops only on wide, accurate consensus | config, strong stop |
| `memory_use_reliability`, `memory_use_relation_history`, `memory_use_basin` | Clean three-layer ablations | true, true, true | boolean | Disabled layers neither update nor affect their downstream weights/query/stop paths; static geometric graph remains common | config, ablation script |
| `memory_require_r1_cache` | Fair fixed-R1 repair evaluation | true | must be true | Missing cache is an error; no R1 fallback is allowed | config, experiment entry |
| `memory_r1_mapping_radius`, `memory_r1_min_mapping_ratio`, `memory_r1_low_confidence_inlier_ratio`, `memory_r1_low_confidence_error_ratio` | Cache-correspondence identity mapping and round-0 memory initialization | 1e-5 m, 0.80, 0.50, 1.00 | >0, [0,1], [0,1], >0 | R1 support must map to the fixed top-K table before round 1; cache mismatch or insufficient mapping stops the run | config, R1 initialization |

## Verification checklist

- [x] Raw parents are independently scored and permanently archived.
- [x] Rejected children cannot remove or replace raw parents.
- [x] Post-refinement Oracle is prefix-limited and cannot be below raw Oracle.
- [x] R1 is a permanent cached round-0 parent; R1-failure metrics are separate.
- [x] Reliability and basin ablations disable their update and every downstream use.
- [x] Signature tests cover global coverage, plane degeneracy, and pre-search basin suppression.
- [x] `tests/test_correspondence_memory.py` covers raw-parent survival and future-round audit leakage.
- [x] `scripts/run_memory_graph_ablations.py` also runs `repeated_regor_equal_budget` with the same cached R1 and H/T settings; pass `--no-include-repeated-regor` only when that control is intentionally excluded.
