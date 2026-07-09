# Redkitchen RGB-D FSV 验证任务说明

## 0. 总目标

当前任务不是修改完整 REGOR-S2 主流程，而是先在单个困难场景上验证：

**真实 image-level FSV 是否能对 REGOR baseline 的成功 / 失败产生有效区分，并通过 FSV gate 提升输出可靠性。**

优先选择场景：

```text
7-scenes-redkitchen
```

选择理由：

```text
1. 3DLoMatch 中该场景有多组低重叠 pair；
2. overlap 分布困难，例如 0.10～0.23；
3. raw RGB-D 数据约 6GB，可控；
4. 适合先做单场景闭环验证。
```

---

## 1. 当前已有数据

本地已有：

```text
3dmatch/test/7-scenes-redkitchen/fragments/cloud_bin_*.ply
3dmatch/test/7-scenes-redkitchen/poses/cloud_bin_*.txt
data/3DMatch/fragments/*_fpfh.npz
3DLoMatch.pkl 或对应 pair list
```

这些是 fragment-level 数据，只包含：

```text
fragment 点云
fragment pose / frame range 线索
fragment 特征
```

它们不能直接构建真实 image-level FSV。

---

## 2. 当前缺失数据

需要下载 raw RGB-D 3DMatch 数据，至少包括：

```text
depth frames
camera intrinsics
per-frame camera poses
```

目标目录应类似：

```text
3dmatch_raw/test/7-scenes-redkitchen/
    camera-intrinsics.txt
    seq-01/
        frame-000000.depth.png
        frame-000000.pose.txt
        frame-000001.depth.png
        frame-000001.pose.txt
        ...
```

如果没有这些文件，不能计算真实 image-level FSV。

---

## 3. 严格禁止事项

禁止：

```text
1. 用 bbox / NN distance / Chamfer proxy 冒充 FSV；
2. 用 trajectory ray-carving proxy 冒充 image-level FSV；
3. 缺少 depth / intrinsics / per-frame pose 时自动降级为 proxy；
4. 使用 GT relative transform 计算 FSV；
5. 使用 GT relative transform 修复坐标系；
6. 只输出报错然后停止；
7. 静默跳过失败 fragment；
8. 未完成完整结果却声称任务完成；
9. 在 redkitchen FSV 验证完成前修改完整 REGOR-S2 主流程。
```

GT 只能用于最终评估：

```text
success / failure
RE
TE
Reg Recall
AUC label
```

不能用于构建 FSV。

---

## 4. 下载 raw RGB-D 数据

目标只下载：

```text
7-scenes-redkitchen.zip
```

Windows PowerShell 示例：

```powershell
$RAW_ROOT = "C:\Users\zhang\Desktop\regor\3dmatch_raw\test"
New-Item -ItemType Directory -Force -Path $RAW_ROOT | Out-Null
Set-Location $RAW_ROOT

$url = "https://3dvision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/7-scenes-redkitchen.zip"
$out = "$RAW_ROOT\7-scenes-redkitchen.zip"

curl.exe -L -C - $url -o $out

Expand-Archive -Path $out -DestinationPath $RAW_ROOT -Force
```

下载后必须自动检查：

```text
camera-intrinsics.txt
seq-XX/frame-XXXXXX.depth.png
seq-XX/frame-XXXXXX.pose.txt
```

---

## 5. 需要新增的脚本

### 5.1 `collect_redkitchen_subset.py`

功能：

```text
1. 读取 3DLoMatch.pkl；
2. 筛选 scene == 7-scenes-redkitchen 的 pair；
3. 收集所有涉及的 cloud_bin_<id>；
4. 从 3dmatch/test/7-scenes-redkitchen/poses/cloud_bin_<id>.txt 第一行解析：
   - sequence
   - frame_start
   - frame_end
5. 输出 manifest 和 pair list。
```

输出：

```text
data/3DMatch/free_space/redkitchen_manifest.json
data/3DMatch/free_space/redkitchen_pairs.json
```

manifest 每项至少包含：

```json
{
  "scene": "7-scenes-redkitchen",
  "fragment_id": "cloud_bin_17",
  "fragment_ply": ".../fragments/cloud_bin_17.ply",
  "fragment_pose_file": ".../poses/cloud_bin_17.txt",
  "sequence": "seq-01",
  "frame_start": 850,
  "frame_end": 899,
  "raw_scene_root": ".../3dmatch_raw/test/7-scenes-redkitchen"
}
```

---

### 5.2 `check_redkitchen_rgbd_raw.py`

功能：

检查 manifest 中每个 fragment 对应的 raw RGB-D 文件是否存在：

```text
camera-intrinsics.txt
seq-XX/frame-XXXXXX.depth.png
seq-XX/frame-XXXXXX.pose.txt
```

但注意：**检查失败不是最终停止条件。**

遇到路径问题、目录层级问题、命名问题时，必须尝试自动修复。

---

### 5.3 `precompute_redkitchen_fsv.py`

功能：

对 redkitchen 中用到的 target fragment 预计算 image-level free-space volume。

输入：

```text
redkitchen_manifest.json
raw RGB-D root
fragment ply root
fragment pose root
```

输出：

```text
data/3DMatch/free_space/7-scenes-redkitchen/cloud_bin_<id>_fsv.npz
```

每个 npz 至少包含：

```text
free_voxel_indices
occupied_voxel_indices
origin
voxel_size
grid_shape
scene
fragment_id
sequence
frame_start
frame_end
depth_scale
camera_intrinsics
num_depth_frames
surface_truncation
free_space_truncation
coordinate_system_note
```

第一版参数：

```text
voxel_size = 0.05 m
depth_stride = 2
max_depth = 4.0 m
surface_truncation = 0.05 m
free_space_truncation = 0.02 m
min_valid_depth = 0.2 m
query_source_points = 5000
```

所有参数必须写入日志，不能静默写死。

---

### 5.4 `eval_redkitchen_fsv.py`

功能：

对 baseline REGOR 输出的预测位姿计算 FSV。

输入：

```text
outputs/redkitchen_baseline/predictions.csv
data/3DMatch/free_space/7-scenes-redkitchen/*_fsv.npz
```

流程：

```text
1. 加载 target fragment 的 free-space volume；
2. 加载 source fragment 点云；
3. 使用 baseline 输出的 T_est 将 source 点变换到 target 坐标系；
4. 下采样 source 点到最多 5000 个；
5. 查询 transformed source 是否落入 target observed free-space；
6. 计算 FSV(T_est)；
7. 保存 FSV score。
```

FSV 定义：

```text
FSV =
transformed source points falling into target observed free-space
/
valid transformed source points inside target FSV grid
```

注意：

```text
grid 外点不要直接算 violation，否则低重叠场景会被误罚。
```

输出：

```text
outputs/redkitchen_fsv/fsv_scores.csv
outputs/redkitchen_fsv/fsv_auc.json
```

---

## 6. 坐标系 sanity check

预处理 free-space 前必须做坐标系检查。

方法：

```text
1. 从 raw depth + intrinsics + frame pose 重建少量 depth points；
2. 转到 fragment 坐标系；
3. 与 fragments/cloud_bin_*.ply 做 NN / Chamfer 距离；
4. 平均距离应在合理范围内，例如 < 5cm～10cm；
5. 如果距离很大，说明坐标系解释错误。
```

如果失败，不能直接停止，必须尝试：

```text
1. cam-to-world / world-to-cam 取反；
2. fragment-to-world / world-to-fragment 取反；
3. depth scale 1000 / 5000 切换；
4. pose 矩阵行列读取方式检查；
5. scene root / seq root 路径修正。
```

每次修正后重新跑 sanity check。

最终必须记录采用的坐标系解释。

---

## 7. Baseline 实验

先跑 redkitchen subset 的 baseline REGOR。

输出：

```text
outputs/redkitchen_baseline/predictions.csv
outputs/redkitchen_baseline/metrics.json
```

`predictions.csv` 至少包含：

```text
pair_id
scene
src_fragment
tgt_fragment
overlap
T_est
success
RE
TE
model_time
```

`metrics.json` 至少包含：

```text
num_pairs
Mean Reg Recall
Mean Re
Mean Te
Mean FMR
Mean model time
```

---

## 8. FSV 判别实验

对 baseline 的 `T_est` 计算 FSV 后，评估：

```text
AUC(success/failure)
pairwise discrimination accuracy
95% specificity sensitivity
90% specificity sensitivity
FSV mean for success
FSV mean for failure
```

目标是先证明：

```text
FSV 能区分 baseline 成功和失败样本。
```

---

## 9. Baseline vs FSV verifier 对比

先不接 active Round2。

只做：

```text
A. Baseline REGOR：所有 redkitchen pair 全部输出
B. REGOR + FSV gate @ 90% specificity
C. REGOR + FSV gate @ 95% specificity
```

每组报告：

```text
accepted_pairs
coverage
accepted_RR / accepted_precision
rejected_pairs
failed_interception_rate
Mean Re on accepted
Mean Te on accepted
mean FSV accepted
mean FSV rejected
```

注意：

```text
FSV gate 的目标不是提高全覆盖 RR；
它的目标是提高放行结果 precision，并拦截失败样本。
```

不能把 coverage 降低后的 accepted precision 直接说成同等条件超过 baseline。

---

## 10. 错误处理规则

本任务要求大模型完成端到端结果。  
遇到报错时，不允许只输出错误然后停止。

### 10.1 可修复问题必须自动修复并继续

包括：

```text
1. Windows / Linux 路径分隔符问题；
2. 相对路径 / 绝对路径混乱；
3. raw root、fragment root、output root 配置错误；
4. zip 解压后多一层目录；
5. scene 目录名大小写或前缀不一致；
6. camera-intrinsics.txt 位于 scene root 或 seq root；
7. frame 命名格式不一致；
8. manifest 缺失；
9. free-space npz 缺失；
10. output 目录缺失；
11. 下载中断；
12. zip 不完整；
13. 坐标系 sanity check 初次失败。
```

处理方式：

```text
1. 自动规范化路径；
2. 递归搜索 scene / seq / intrinsics / depth / pose；
3. 自动重建 manifest；
4. 自动重新预处理缺失 fragment；
5. 自动创建输出目录；
6. 下载失败时断点续传并重试至少 3 次；
7. 尝试常见坐标系解释修正。
```

---

### 10.2 只有不可修复问题允许停止

包括：

```text
1. 官方下载链接不可访问，且重试 3 次仍失败；
2. raw RGB-D zip 无法下载，也没有本地可用镜像；
3. 下载后的数据中确实不存在 depth frames；
4. 下载后的数据中确实不存在 per-frame camera poses；
5. 下载后的数据中确实不存在 camera intrinsics，且无法从官方格式推断；
6. 坐标系 sanity check 在所有常见解释下都失败；
7. baseline REGOR 代码本身无法运行，且错误不在本次 FSV 数据流程范围内。
```

如果停止，必须输出：

```text
1. 已完成到哪一步；
2. 具体阻塞文件或阻塞条件；
3. 已尝试过哪些修复；
4. 为什么仍无法继续；
5. 下一步人工需要提供什么。
```

---

## 11. 禁止的错误处理方式

禁止：

```text
1. 只打印 Missing camera intrinsics 然后停止；
2. 只打印 FileNotFoundError 然后停止；
3. 缺少 npz 就跳过该 fragment；
4. 缺少 depth 就退化成 proxy；
5. 缺少 intrinsics 就用默认内参硬凑；
6. 未验证坐标系就继续算 FSV；
7. 用 GT relative transform 做 sanity check 修正；
8. 下载失败一次就结束任务。
```

---

## 12. 最终 summary 要求

最终必须生成：

```text
outputs/redkitchen_fsv/summary.md
```

内容必须包含：

```text
1. 下载的数据路径和大小；
2. raw RGB-D 完整性检查结果；
3. redkitchen subset pair 数量；
4. 预处理 fragment 数量；
5. free-space volume 参数；
6. 坐标系 sanity check 结果；
7. baseline redkitchen subset 结果；
8. FSV 判别 AUC；
9. 90% specificity gate 结果；
10. 95% specificity gate 结果；
11. 是否优于 bbox/NN proxy；
12. 当前是否足以接入 Round2 reset prior；
13. 所有新增参数列表；
14. 所有失败、修复、重试记录。
```

---

## 13. 任务完成条件

只有以下文件全部存在，且 `summary.md` 中包含完整数值结果，任务才算完成：

```text
data/3DMatch/free_space/redkitchen_manifest.json
data/3DMatch/free_space/redkitchen_pairs.json
data/3DMatch/free_space/7-scenes-redkitchen/cloud_bin_*_fsv.npz
outputs/redkitchen_baseline/predictions.csv
outputs/redkitchen_baseline/metrics.json
outputs/redkitchen_fsv/fsv_scores.csv
outputs/redkitchen_fsv/fsv_auc.json
outputs/redkitchen_fsv/summary.md
```

否则不能停止。

---

## 14. 最重要的执行原则

这次任务不是写一个检查脚本，也不是写一个半成品 pipeline。

它必须完成：

```text
下载 raw RGB-D
→ 修复路径和数据问题
→ 生成 manifest
→ 检查 raw 数据
→ 坐标系统一
→ 预处理 true image-level FSV
→ 跑 redkitchen baseline
→ 计算 FSV score
→ 评估 FSV AUC
→ 做 FSV gate 对比
→ 输出 summary
```

只有完成以上闭环，才能停止。

一句话版本：

**遇到可修复错误必须修复并继续；只有确认数据源本身缺失或坐标系无法确认时，才允许停止。任务目标是拿到 redkitchen 上 baseline vs FSV verifier 的完整数值对比，而不是停在报错信息。**
