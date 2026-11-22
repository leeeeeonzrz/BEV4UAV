# Preparing private data for BEV4UAV

[English README](../README.md) · [中文说明](../README_zh-CN.md)

Prepare your own synchronized LiDAR frames, images, labels, calibration, and splits locally. Nothing in this guide supplies or downloads a dataset. The commands below write generated metadata and ground-truth database files inside your local data directory.

## Coordinate contract

The directory and text layout resemble KITTI, but this fork's `KittiDataset.parse_ann_info` constructs `LiDARInstance3DBoxes` directly from stored boxes with `origin=(0.5, 0.5, 0)`. Thus `bbox_3d` must be `[x, y, z, l, w, h, yaw]`, with a bottom-center location and yaw in the MMDetection3D LiDAR convention. Standard KITTI camera-coordinate 3D labels need an explicit coordinate/dimension/yaw conversion before use.

For the `--kitti-dim-order lwh` option, label lines use these space-separated fields:

```text
type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom l w h x y z yaw
```

Use the actual 2D annotation values for the image fields; do not confuse them with the 3D LiDAR coordinates. The configured class order is `Car,TrafficPole`. Unknown classes are mapped to ignored labels by the information converter. Changing classes requires consistent changes to the metadata, model/evaluator configuration, and checkpoint initialization.

Point clouds are binary `float32` arrays with four values per point: `x, y, z, intensity`. Coordinates, labels, and calibration must refer to the same frame and physical units. Images are read from `image_2` as the `CAM2` view.

## Directory layout

Use six-digit numeric frame names and one frame ID per split-file line:

```text
data/bev4uav/
├── ImageSets/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── training/
│   ├── velodyne/       # <frame_id>.bin
│   ├── image_2/        # <frame_id>.png
│   ├── label_2/        # <frame_id>.txt
│   └── calib/          # <frame_id>.txt
└── testing/            # needed for frame IDs listed in test.txt
    ├── velodyne/
    ├── image_2/
    └── calib/
```

Both training and validation IDs are resolved under `training/`. The converter reads all three split files; if you do not use a separate test set, keep `test.txt` present and empty. The released experiment configurations evaluate the validation split.

The calibration parser expects the seven KITTI-style lines in this order: `P0`, `P1`, `P2`, `P3` (each 3×4), `R0_rect` (3×3), `Tr_velo_to_cam` (3×4), and `Tr_imu_to_velo` (3×4), with row-major numeric values. Supply valid calibration for your sensor setup. Image projection uses `P2 @ R0_rect @ Tr_velo_to_cam`; dummy or inconsistent matrices invalidate multimodal alignment.

## Generate metadata

With the environment and custom CUDA operations installed, run from the repository root:

```bash
python tools/create_data.py kitti \
  --root-path data/bev4uav \
  --out-dir data/bev4uav \
  --extra-tag kitti \
  --kitti-classes Car,TrafficPole \
  --kitti-dim-order lwh \
  --kitti-box-coordinate lidar \
  --skip-kitti-reduced \
  --kitti-gt-points-source raw
```

Keep `--root-path` and `--out-dir` identical for this entry point. The configurations expect `kitti_infos_train.pkl` and `kitti_infos_val.pkl`; conversion also creates train/validation-combined and test metadata, plus a ground-truth database for training. Existing metadata with the same names may be overwritten. The `raw` option means using points from `training/velodyne`; if that directory already contains CSF-processed points, those are the points used. It does not undo preprocessing.

`--skip-kitti-reduced` avoids the legacy camera-FOV point-cloud reduction, and `--kitti-gt-points-source raw` keeps the ground-truth database source consistent with the public configurations. Neither option runs CSF.

## Known converter limitations

Use `--kitti-box-coordinate lidar` to count raw LiDAR points directly inside the supplied LiDAR boxes; the default `camera` retains the legacy KITTI camera-coordinate/FOV counting behavior. This switch affects point counting only and does not convert stored box coordinates. Auxiliary camera instance/center/depth generation still assumes camera-coordinate labels and is not a verified LiDAR-label conversion pipeline. Inspect and adapt those fields before relying on camera-instance annotations or downstream filters. Changing dimension order alone does not repair coordinate assumptions.

The release preserves these behaviors rather than silently changing the original experiment's geometry. Verify converted `instances[*].bbox_3d`, class indices, box origins, and projections on representative frames before training. The custom KITTI evaluator also needs to be checked against your annotation and coordinate conventions; a successful conversion command does not establish evaluation correctness.

## CSF and configuration adaptation

Keep unfiltered and CSF-preprocessed inputs in separate dataset roots with consistent frame IDs, splits, labels, and calibration. Run metadata generation separately for each root, replacing both directory arguments above. CSF itself is an external preprocessing step; its algorithm implementation, parameters, and processed results are not part of this repository.

Pass the selected root through the training wrapper:

```bash
python tools/train_bev4uav.py projects/BEVFusion/configs/lidar_csf.py \
  --data-root data/bev4uav_csf \
  --check-only
```

For another sensor setup, review point-cloud and voxel ranges, sparse shapes, image resolution/cropping, view-transform bounds, and evaluator ranges together. Changing only `data_root` does not adapt the geometric assumptions.

Generated metadata, point databases, calibration, split lists, sensor recordings, and model weights should remain local. The repository's ignore rules are a safeguard, not a replacement for inspecting files before sharing.

## 中文要点

- 这是自定义 KITTI 风格数据接口。三维框必须已在 LiDAR 坐标系，顺序为 `[x, y, z, l, w, h, yaw]`，位置为底面中心；不能直接使用标准 KITTI 相机坐标标签。
- 默认类别和顺序为 `Car,TrafficPole`，点云每点为四个 `float32`：`x,y,z,intensity`。标定、图像、点云与标签必须对齐。
- 转换需要 `train.txt`、`val.txt`、`test.txt`；没有独立测试集时保留空 `test.txt`。训练和验证帧均位于 `training/`。
- 按上面的命令使用 `--kitti-dim-order lwh`、`--kitti-box-coordinate lidar`、`--skip-kitti-reduced`、`--kitti-gt-points-source raw`，并保持输入根目录与输出目录相同。命令会生成或覆盖本地元信息和 GT 数据库。
- LiDAR 模式的点数按完整 LiDAR 点云和原始 LiDAR 框统计；相机实例/中心/深度辅助逻辑仍按相机坐标处理，不能视为经过验证的 LiDAR 标签转换。依赖这些字段前需要核对或适配。训练框、投影和评估器也应通过代表性样本检查。
- CSF 由使用者离线执行；配置文件不会自动过滤点云。分别准备未过滤与已处理数据目录，并为各自数据生成元信息。
