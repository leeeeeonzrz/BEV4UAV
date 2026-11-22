"""Synthetic regression tests for BEV4UAV's KITTI-style LiDAR labels.

Run after enabling the BEV4UAV/MMDetection3D environment:

    python -m pytest -q tests/test_kitti_lidar_count.py

No private dataset is required; tests use temporary synthetic points.
"""
from copy import deepcopy

import numpy as np
import pytest

mmengine = pytest.importorskip('mmengine')
pytest.importorskip('mmcv')

from tools.dataset_converters import kitti_converter as kc
from tools.dataset_converters.create_gt_database import create_groundtruth_database
from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric


LIDAR2CAM = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float32)
GT_BOX_LIDAR = np.array([10., 2., -1., 4., 2., 2., np.pi / 6],
                        dtype=np.float32)


def synthetic_points():
    # Four inside the rotated bottom-centered box and three outside.
    local = np.array([
        [0., 0., .3], [1.5, .5, 1.2], [-1.5, -.5, 1.6],
        [.4, -.6, .7], [4.5, 0., .7], [0., 0., -.5], [0., 0., 2.5]
    ], dtype=np.float32)
    yaw = float(GT_BOX_LIDAR[6])
    c, s = np.cos(yaw), np.sin(yaw)
    xy = local[:, :2] @ np.array([[c, s], [-s, c]], dtype=np.float32)
    xyz = np.column_stack((xy, local[:, 2])) + GT_BOX_LIDAR[:3]
    return np.column_stack((xyz, np.arange(len(xyz), dtype=np.float32)))


def _info(label_box):
    return dict(
        image=dict(image_shape=np.array([128, 128])),
        point_cloud=dict(velodyne_path='training/velodyne/000001.bin'),
        calib=dict(R0_rect=np.eye(4, dtype=np.float32),
                   Tr_velo_to_cam=LIDAR2CAM,
                   P2=np.eye(4, dtype=np.float32)),
        annos=dict(
            name=np.array(['Car', 'DontCare']),
            dimensions=np.array([label_box[3:6], [0, 0, 0]], dtype=np.float32),
            location=np.array([label_box[:3], [0, 0, 0]], dtype=np.float32),
            rotation_y=np.array([label_box[6], 0.], dtype=np.float32),
        ),
    )


@pytest.fixture
def synth_root(tmp_path):
    folder = tmp_path / 'training' / 'velodyne'
    folder.mkdir(parents=True)
    synthetic_points().astype(np.float32).tofile(folder / '000001.bin')
    return tmp_path


def test_lidar_count_uses_supplied_box_without_camera_conversion(synth_root):
    info = _info(GT_BOX_LIDAR)
    kc._calculate_num_points_in_gt(
        synth_root, [info], relative_path=True,
        remove_outside=False, box_coordinate='lidar')
    assert info['annos']['num_points_in_gt'].tolist() == [4, -1]


def test_legacy_camera_mode_still_converts(synth_root):
    # Legacy camera box dimensions are [l, h, w].
    box_cam = np.concatenate((
        (LIDAR2CAM @ np.r_[GT_BOX_LIDAR[:3], 1.])[:3],
        GT_BOX_LIDAR[[3, 5, 4]],
        [-float(GT_BOX_LIDAR[6]) - np.pi / 2],
    )).astype(np.float32)
    info = _info(box_cam)
    kc._calculate_num_points_in_gt(
        synth_root, [info], relative_path=True,
        remove_outside=False, box_coordinate='camera')
    assert info['annos']['num_points_in_gt'].tolist() == [4, -1]


def test_create_kitti_info_file_passes_lidar_mode_and_full_cloud(
        synth_root, monkeypatch):
    sets = synth_root / 'ImageSets'
    sets.mkdir()
    (sets / 'train.txt').write_text('000001\n')
    (sets / 'val.txt').write_text('')
    (sets / 'test.txt').write_text('')

    def fake_infos(*args, **kwargs):
        return [deepcopy(_info(GT_BOX_LIDAR)) for _ in kwargs['image_ids']]

    monkeypatch.setattr(kc, 'get_kitti_image_info', fake_infos)
    kc.create_kitti_info_file(
        synth_root, save_path=synth_root,
        dim_order='lwh', box_coordinate='lidar')
    out = mmengine.load(synth_root / 'kitti_infos_train.pkl')
    assert out[0]['annos']['num_points_in_gt'].tolist() == [4, -1]


def test_gt_database_and_evaluator_ignore_poisoned_metadata_count(synth_root):
    # Deliberately poison the metadata count: GT database must re-crop.
    instance = dict(
        bbox_3d=GT_BOX_LIDAR.tolist(),
        bbox_label=0, bbox_label_3d=0,
        bbox=[10., 10., 45., 45.],
        truncated=0., occluded=0, alpha=0., score=1.,
        difficulty=0, num_lidar_pts=0,
    )
    v2 = dict(
        metainfo=dict(categories=dict(Car=0), dataset='kitti',
                      info_version='1.1'),
        data_list=[dict(
            sample_idx=0,
            lidar_points=dict(lidar_path='000001.bin', num_pts_feats=4),
            images=dict(CAM2=dict(
                img_path='000001.png', height=128, width=128,
                lidar2cam=LIDAR2CAM.tolist(),
                cam2img=np.eye(4, dtype=np.float32).tolist())),
            instances=[instance],
        )],
    )
    mmengine.dump(v2, synth_root / 'kitti_infos_train.pkl')

    from mmengine.registry import init_default_scope
    import mmdet3d.datasets  # noqa: F401; register KittiDataset/transforms
    init_default_scope('mmdet3d')
    create_groundtruth_database(
        'KittiDataset', str(synth_root), 'kitti',
        info_path='kitti_infos_train.pkl',
        kitti_pts_prefix='training/velodyne',
        kitti_classes=['Car'])

    db_infos = mmengine.load(synth_root / 'kitti_dbinfos_train.pkl')
    db_entry = db_infos['Car'][0]
    assert db_entry['num_points_in_gt'] == 4  # independently recomputed
    np.testing.assert_allclose(db_entry['box3d_lidar'], GT_BOX_LIDAR, atol=1e-5)
    db_points = np.fromfile(
        synth_root / db_entry['path'], dtype=np.float32).reshape(-1, 4)
    assert db_points.shape[0] == 4
    np.testing.assert_allclose(
        db_points[:, :3] + GT_BOX_LIDAR[:3],
        synthetic_points()[:4, :3], atol=1e-5)

    # GT format conversion for KITTI evaluation ignores num_lidar_pts.
    metric = KittiMetric(ann_file=str(synth_root / 'kitti_infos_train.pkl'),
                         eval_coordinate='lidar')
    converted = metric.convert_annos_to_kitti_annos(deepcopy(v2))
    gt_eval_box = converted[0]['kitti_annos']
    np.testing.assert_allclose(gt_eval_box['location_lidar'][0],
                               GT_BOX_LIDAR[:3], atol=1e-5)
    np.testing.assert_allclose(gt_eval_box['dimensions_lidar'][0],
                               GT_BOX_LIDAR[3:6], atol=1e-5)
    np.testing.assert_allclose(gt_eval_box['rotation_y_lidar'][0],
                               GT_BOX_LIDAR[6], atol=1e-5)
