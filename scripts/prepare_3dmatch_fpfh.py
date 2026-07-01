from pathlib import Path
import shutil
import numpy as np
import open3d as o3d


SCENES = [
    "7-scenes-redkitchen",
    "sun3d-home_at-home_at_scan1_2013_jan_1",
    "sun3d-home_md-home_md_scan9_2012_sep_30",
    "sun3d-hotel_uc-scan3",
    "sun3d-hotel_umd-maryland_hotel1",
    "sun3d-hotel_umd-maryland_hotel3",
    "sun3d-mit_76_studyroom-76-1studyroom2",
    "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",
]


def cloud_index(path):
    return int(path.stem.split("_")[-1])


def compute_fpfh(ply_path, voxel_size):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    normal_radius = voxel_size * 2
    feature_radius = voxel_size * 5
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
    )
    xyz = np.asarray(pcd.points, dtype=np.float32)
    fpfh = np.asarray(feature.data, dtype=np.float32).T
    return xyz, fpfh


def main():
    source_root = Path("3dmatch/test")
    target_root = Path("data/3DMatch")
    voxel_size = 0.03

    if not source_root.exists():
        raise FileNotFoundError(f"Missing raw 3DMatch test data: {source_root}")

    for scene in SCENES:
        source_scene = source_root / scene / "fragments"
        target_scene = target_root / "fragments" / scene
        target_scene.mkdir(parents=True, exist_ok=True)

        for ply_path in sorted(source_scene.glob("cloud_bin_*.ply"), key=cloud_index):
            idx = cloud_index(ply_path)
            out_path = target_scene / f"cloud_bin_{idx}_fpfh.npz"
            if out_path.exists():
                continue
            xyz, feature = compute_fpfh(ply_path, voxel_size)
            np.savez_compressed(out_path, xyz=xyz, feature=feature)
            print(f"{scene}/cloud_bin_{idx}: {xyz.shape[0]} points")

        gt_source = Path("benchmarks/3DMatch") / scene
        gt_target = target_root / "gt_result" / f"{scene}-evaluation"
        gt_target.mkdir(parents=True, exist_ok=True)
        for name in ["gt.log", "gt.info"]:
            shutil.copy2(gt_source / name, gt_target / name)


if __name__ == "__main__":
    main()
