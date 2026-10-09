# BEV4UAV

[简体中文](README_zh-CN.md) · [Data preparation](docs/data_preparation.md)

BEV4UAV is a research codebase for UAV LiDAR–camera 3D object detection. It builds on a modified MMDetection3D 1.4.0 codebase and the BEVFusion project, with aerial view transformation, residual fusion, a merged image feature pyramid, and an IoU quality prediction branch.

This repository contains source code and experiment configurations. Datasets, annotations, calibration files, pretrained weights, trained checkpoints, experiment outputs, and paper files are not distributed.

## Implementation

| Component | Implementation |
| --- | --- |
| LiDAR detector and multimodal model | `projects/BEVFusion/bevfusion/bevfusion.py` |
| Aerial view transformation | `AerialDepthLSSTransformGN` in `depth_lss.py` |
| AFRF configuration: view transformation and residual fusion | `depth_lss.py` and `LidarResidualGNEMAFuser` in `emafuser.py` |
| Merged feature pyramid (MFPN) | `MergedLSSFPN` in `mergedlssfpn.py` |
| IoU quality supervision and score calibration | `TransFusionHead` in `transfusion_head.py` |
| Training-stage module freezing | `ModuleFreezeHook` in `hooks.py` |

The IoU branch predicts a scalar quality score, uses `mmdet.VarifocalLoss` against matched 3D IoU targets, and calibrates detection scores when enabled. It is not a distance-IoU (DIoU) box regression loss.

CSF denotes an offline point-cloud preprocessing condition. A CSF implementation and processed point clouds are not included. Selecting a CSF configuration does not perform filtering; provide your own consistently preprocessed data.

## Installation

Use a separate environment with a CUDA-capable PyTorch installation, a matching CUDA toolkit, and a C++ compiler. MMCV must include its compiled operators. This repository installs a modified package named `mmdet3d`; avoid mixing it with a separate upstream installation in the same environment.

The source and dependency files require the following compatible ranges:

| Dependency | Range |
| --- | --- |
| MMCV | `>=2.0.0rc4,<2.2.0` |
| MMDetection | `>=3.0.0,<3.3.0` |
| MMEngine | `>=0.8.0,<1.0.0` |

After installing PyTorch for your hardware, run from the repository root:

```bash
python -m pip install openmim
mim install 'mmcv>=2.0.0rc4,<2.2.0'
python -m pip install -r requirements/mminstall.txt
python -m pip install -r requirements/runtime.txt
python -m pip install --no-build-isolation -e .
python projects/BEVFusion/setup.py build_ext --inplace
```

The BEVFusion extension build lets PyTorch select the CUDA architecture for the available GPU. Set `TORCH_CUDA_ARCH_LIST` explicitly when targeting another architecture. Sparse convolution must be available through the supported MMCV backend or a compatible `spconv` installation. These dependency ranges are not a complete environment lock, and installation, CUDA execution, and end-to-end training have not been validated for every supported combination.

## Configurations

All public experiment configurations are in `projects/BEVFusion/configs/`.

| Configuration | Input and components |
| --- | --- |
| `lidar.py` | LiDAR baseline on unfiltered input |
| `lidar_csf.py` | LiDAR baseline on externally CSF-preprocessed input |
| `fusion_baseline.py` | LiDAR–camera baseline on unfiltered input |
| `fusion_csf.py` | Fusion baseline on externally CSF-preprocessed input |
| `fusion_csf_afrf.py` | CSF + AFRF |
| `fusion_csf_afrf_mfpn.py` | CSF + AFRF + MFPN |
| `bev4uav.py` | CSF + AFRF + MFPN + IoU quality supervision |

The inherited experiment classes are **`Car` and `TrafficPole`**, in that order. Adapting to other classes requires corresponding annotation metadata, model/evaluator configuration, and compatible checkpoints. Dataset-specific geometry, ranges, augmentation, and schedules remain explicit in the configurations; review them for your own sensor setup.

## Data preparation

Follow [the dataset guide](docs/data_preparation.md) before training. The loader uses a custom KITTI-style layout with **LiDAR-coordinate 3D labels**. Standard KITTI camera-coordinate labels cannot be used unchanged.

The default data root is `data/bev4uav/`. It can be overridden with `--data-root` in the training wrapper or the `BEV4UAV_DATA_ROOT` environment variable. Use separate roots for unfiltered and CSF-preprocessed data.

## Training

Training follows two stages. First train a LiDAR detector on the matching input condition:

```bash
python tools/train_bev4uav.py projects/BEVFusion/configs/lidar_csf.py \
  --data-root data/bev4uav_csf \
  --work-dir work_dirs/lidar_csf
```

Then select a trained LiDAR checkpoint and supply a compatible pretrained Swin-Tiny checkpoint for fusion training. The paths below are local placeholders to populate yourself:

```bash
python tools/train_bev4uav.py projects/BEVFusion/configs/bev4uav.py \
  --data-root data/bev4uav_csf \
  --lidar-checkpoint checkpoints/lidar_csf.pth \
  --image-checkpoint checkpoints/swin_tiny.pth \
  --work-dir work_dirs/bev4uav
```

The fusion configurations freeze the image backbone, LiDAR encoders/backbone/neck, and detection head while training the remaining image and fusion components. They therefore require suitable initialization; they are not a recipe for training the full fusion network from scratch. Match the LiDAR checkpoint's architecture, classes, and preprocessing to the selected configuration. For `fusion_baseline.py`, train `lidar.py` on the unfiltered input condition first.

Use `--check-only` with the same arguments to parse the configuration and validate required checkpoint paths without starting training. It does not inspect dataset contents, validate checkpoint contents, or replace a GPU forward/backward test. `--cfg-options KEY=VALUE ...` supports native MMEngine overrides; `--resume PATH` resumes a local training checkpoint. When resuming fusion training, the resume checkpoint replaces the stage-one checkpoint requirement, but the image initialization checkpoint is still required.

Defaults can also be set with `BEV4UAV_LIDAR_CHECKPOINT` and `BEV4UAV_IMAGE_CHECKPOINT`. No checkpoint is downloaded automatically by the release configurations.

## Evaluation

Evaluate your own complete fusion checkpoint on the configured validation split:

```bash
BEV4UAV_DATA_ROOT=data/bev4uav_csf/ \
python tools/test.py projects/BEVFusion/configs/bev4uav.py \
  checkpoints/bev4uav.pth \
  --work-dir work_dirs/bev4uav_eval \
  --cfg-options model.img_backbone.init_cfg=None
```

The public configurations use the validation dataloader for testing. Review the custom KITTI evaluator and its class/IoU settings before comparing results. No benchmark scores or claims of paper reproduction are supplied by this source release.

## Release scope and provenance

The modified `mmdet3d/` package is included because the project relies on changes to data loading and evaluation as well as the BEVFusion modules. Its namespace and upstream version identifier are retained for compatibility. Public configuration names and local paths are simplified; private experiment locations and artifacts are excluded.

The release has not been independently validated through a complete training/evaluation run with the original private data and checkpoints. A compatible environment, your own calibrated data, external CSF preprocessing when selected, and matching pretrained checkpoints are required for reproduction.

Before sharing a source release, run `python -B tools/check_release.py`. After staging files with Git, run `python -B tools/check_release.py --staged` to catch prohibited data and weights, including files staged with `git add --force`.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for licensing and attribution. Upstream foundations are [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) and [BEVFusion](https://github.com/mit-han-lab/bevfusion). File-level third-party notices remain applicable. BEV4UAV publication metadata and a paper citation are not included in this release.
