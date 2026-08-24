from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_ROOT = REPO_ROOT / "source" / "regor"
CODE_ROOT = REPO_ROOT / "code" / "regor"
sys.path.insert(0, str(BASELINE_ROOT))
sys.path.insert(0, str(CODE_ROOT))

from Correspondence_regenerate_v2 import Regenerator
from initial_matching_plus import Matcher_plus
from Tranformation_estimaton import Estimator

from .config import OrbitRegorConfig
from .runtime import OrbitRegorResult, OrbitRegorRuntime


def create_matcher() -> Matcher_plus:
    return Matcher_plus(
        inlier_threshold=0.1,
        num_node="all",
        use_mutual=False,
        d_thre=0.1,
        num_iterations=10,
        ratio=0.2,
        nms_radius=0.1,
        max_points=8000,
        k1=60,
        k2=50,
        FS_TCD_thre=0.05,
        relax_match_num=100,
        NS_by_IC=50,
    )


def prepare_orbit_regor(
    source: torch.Tensor,
    target: torch.Tensor,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    placeholder: torch.Tensor,
    config: OrbitRegorConfig | None = None,
) -> tuple[OrbitRegorRuntime, object]:
    matcher = create_matcher()
    matcher.gt_trans = placeholder
    _, _, _, raw_source, raw_target, _, _ = matcher.match_pair(
        source, target, source_features, target_features
    )
    native_poses, _ = matcher.SC2_PCR(raw_source, raw_target)
    runtime = OrbitRegorRuntime(config)
    prepared = runtime.prepare(
        source[0],
        target[0],
        source_features[0],
        target_features[0],
        raw_source[0],
        raw_target[0],
        native_poses[0],
    )
    return runtime, prepared


def finalize_orbit_regor(
    runtime: OrbitRegorRuntime,
    prepared: object,
    variant: str,
    source: torch.Tensor,
    target: torch.Tensor,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    placeholder: torch.Tensor,
) -> OrbitRegorResult:
    result = runtime.solve(prepared, variant)
    if variant == "no_point_regor" or len(result.seed_source) < 3:
        result.pose = result.pose[None]
        result.telemetry["regor_rounds"] = 0
        result.telemetry["final_correspondence_count"] = len(result.seed_source)
        return result
    source_seed, target_seed = runtime.ensure_seed_count(result.seed_source, result.seed_target)
    source_corr = source_seed[None]
    target_corr = target_seed[None]
    regenerator = Regenerator()
    counts = [int(source_corr.shape[1])]
    for knn, sampling in zip(runtime.config.regor_knn, runtime.config.regor_sampling):
        source_corr, target_corr, _ = regenerator.regenerate(
            source_corr,
            target_corr,
            source,
            target,
            source_features,
            target_features,
            placeholder,
            knn_num=knn,
            sampling_num=sampling,
        )
        counts.append(int(source_corr.shape[1]))
    final_pose, _, _, _ = Estimator(num_node=5000).estimator(
        source_corr, target_corr, source, target, placeholder
    )
    result.pose = final_pose
    result.telemetry["regor_rounds"] = 2
    result.telemetry["regor_correspondence_counts"] = counts
    result.telemetry["final_correspondence_count"] = int(source_corr.shape[1])
    return result
