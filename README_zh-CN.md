# BEV4UAV

[English](README.md) · [数据准备说明](docs/data_preparation.md)

BEV4UAV 是面向无人机 LiDAR–相机三维目标检测的研究代码，基于修改后的 MMDetection3D 1.4.0 和 BEVFusion，包含航拍视角变换、残差融合、合并图像特征金字塔与 IoU 质量预测分支。

本仓库仅发布源码和实验配置，不提供数据集、标注、标定文件、预训练权重、训练检查点、实验结果或论文文件。

## 核心实现

| 组件 | 实现位置 |
| --- | --- |
| LiDAR 检测器与多模态模型 | `projects/BEVFusion/bevfusion/bevfusion.py` |
| 航拍视角变换 | `depth_lss.py` 中的 `AerialDepthLSSTransformGN` |
| AFRF 配置中的视角变换与残差融合 | `depth_lss.py` 与 `emafuser.py` 中的 `LidarResidualGNEMAFuser` |
| 合并特征金字塔 MFPN | `mergedlssfpn.py` 中的 `MergedLSSFPN` |
| IoU 质量监督与分数校准 | `transfusion_head.py` 中的 `TransFusionHead` |
| 分阶段训练的模块冻结 | `hooks.py` 中的 `ModuleFreezeHook` |

IoU 分支预测标量质量分数，使用 `mmdet.VarifocalLoss` 监督匹配的三维 IoU 目标，启用时参与检测分数校准。它并非 distance-IoU（DIoU）边框回归损失。

CSF 表示离线点云预处理条件。仓库不包含 CSF 实现或处理后的点云；选择名称带 CSF 的配置不会自动执行过滤，需要自行准备预处理一致的数据。

## 安装

请使用独立环境，先安装适配硬件的 CUDA 版 PyTorch、对应 CUDA 工具链和 C++ 编译器。MMCV 需要包含编译算子。本仓库以 `mmdet3d` 包名安装修改后的框架，请避免在同一环境混用其他上游安装。

源码与依赖文件对应的兼容范围为：MMCV `>=2.0.0rc4,<2.2.0`，MMDetection `>=3.0.0,<3.3.0`，MMEngine `>=0.8.0,<1.0.0`。安装 PyTorch 后，在仓库根目录执行：

```bash
python -m pip install openmim
mim install 'mmcv>=2.0.0rc4,<2.2.0'
python -m pip install -r requirements/mminstall.txt
python -m pip install -r requirements/runtime.txt
python -m pip install --no-build-isolation -e .
python projects/BEVFusion/setup.py build_ext --inplace
```

BEVFusion 扩展由 PyTorch 根据可用 GPU 选择 CUDA 架构；如需指定其他目标架构，可设置 `TORCH_CUDA_ARCH_LIST`。稀疏卷积需要受支持的 MMCV 后端或兼容的 `spconv`。上述范围不是完整锁定环境，未对所有组合完成安装、CUDA 执行和端到端训练验证。

## 实验配置

公开配置统一位于 `projects/BEVFusion/configs/`。

| 配置 | 输入与组件 |
| --- | --- |
| `lidar.py` | 未过滤点云上的 LiDAR 基线 |
| `lidar_csf.py` | 外部 CSF 预处理点云上的 LiDAR 基线 |
| `fusion_baseline.py` | 未过滤点云上的 LiDAR–相机融合基线 |
| `fusion_csf.py` | 外部 CSF 预处理点云上的融合基线 |
| `fusion_csf_afrf.py` | CSF + AFRF |
| `fusion_csf_afrf_mfpn.py` | CSF + AFRF + MFPN |
| `bev4uav.py` | CSF + AFRF + MFPN + IoU 质量监督 |

配置保留原实验的两类及顺序：**`Car`、`TrafficPole`**。换用其他类别时，需要同步调整标注元信息、模型和评估配置，并提供相匹配的检查点。几何范围、数据增强、传感器参数和训练日程仍在配置中显式定义，使用自己的数据前需要核对。

## 数据准备

详见[数据准备说明](docs/data_preparation.md)。数据采用自定义 KITTI 风格目录，但三维标签已经位于 **LiDAR 坐标系**；标准 KITTI 的相机坐标标签不能直接使用。

默认数据目录为 `data/bev4uav/`。训练入口支持 `--data-root`，配置也支持 `BEV4UAV_DATA_ROOT` 环境变量。原始点云与 CSF 处理点云应分别存放。

## 两阶段训练

先训练与输入预处理条件一致的 LiDAR 检测器：

```bash
python tools/train_bev4uav.py projects/BEVFusion/configs/lidar_csf.py \
  --data-root data/bev4uav_csf \
  --work-dir work_dirs/lidar_csf
```

随后选择训练得到的 LiDAR 检查点，并自行准备兼容的 Swin-Tiny 预训练权重，再训练融合模型。以下权重路径仅为本地占位示例，需要自行准备：

```bash
python tools/train_bev4uav.py projects/BEVFusion/configs/bev4uav.py \
  --data-root data/bev4uav_csf \
  --lidar-checkpoint checkpoints/lidar_csf.pth \
  --image-checkpoint checkpoints/swin_tiny.pth \
  --work-dir work_dirs/bev4uav
```

融合配置冻结图像主干、LiDAR 编码器/主干/颈部以及检测头，训练其余图像分支与融合组件，因此必须提供合适的初始化权重，不能直接作为整个融合网络的从零训练方案。LiDAR 检查点的网络结构、类别和预处理条件应与配置一致。对于 `fusion_baseline.py`，先使用未过滤点云训练 `lidar.py`。

在相同命令后添加 `--check-only`，可在不启动训练的情况下解析配置并检查所需权重路径；此检查不检查数据集内容、不验证权重内容，也不能代替 GPU 前向/反向测试。`--cfg-options KEY=VALUE ...` 支持 MMEngine 原生覆盖参数，`--resume PATH` 用于恢复本地检查点。恢复融合训练时，恢复检查点可代替第一阶段检查点，但仍需提供图像初始化权重。

权重默认路径还可通过 `BEV4UAV_LIDAR_CHECKPOINT` 与 `BEV4UAV_IMAGE_CHECKPOINT` 设置。公开配置不会自动下载权重。

## 评估

使用自行训练的完整融合模型检查点评估配置中的验证集：

```bash
BEV4UAV_DATA_ROOT=data/bev4uav_csf/ \
python tools/test.py projects/BEVFusion/configs/bev4uav.py \
  checkpoints/bev4uav.pth \
  --work-dir work_dirs/bev4uav_eval \
  --cfg-options model.img_backbone.init_cfg=None
```

公开配置使用验证集 dataloader 作为测试入口。比较结果前请核对自定义 KITTI 评估器、类别和 IoU 阈值。此源码发布不提供性能数字，也不宣称已完成论文结果复现。

## 发布范围与来源

仓库保留修改后的完整 `mmdet3d/`，因为项目同时依赖数据加载、评估与 BEVFusion 模块的修改。为保持兼容，保留原 Python 包命名空间与上游版本标识。公开配置名称和本地路径已整理，私人实验路径与产物不随仓库分发。

此发布版本尚未使用原始私有数据和权重独立完成一次完整训练/评估。复现需要兼容环境、自有标定数据、按所选配置完成的外部 CSF 预处理，以及匹配的预训练权重。

分享源码前运行 `python -B tools/check_release.py`。通过 Git 暂存文件后，再运行 `python -B tools/check_release.py --staged`，检查禁止发布的数据与权重，包括通过 `git add --force` 强制暂存的文件。

许可证与归属见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。基础项目为 [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) 和 [BEVFusion](https://github.com/mit-han-lab/bevfusion)，各文件保留的第三方声明仍然适用。本次发布不附带 BEV4UAV 的正式论文元数据或引用条目。
