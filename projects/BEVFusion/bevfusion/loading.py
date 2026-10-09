# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional

import mmcv
import numpy as np
from mmengine.fileio import get

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BEVLoadKittiImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # TODO: consider split the multi-sweep part out of this pipeline
        # Derive the mask and transform for loading of multi-sweep data
        filename,cam2img,lidar2cam,cam2lidar,lidar2img = [], [], [], [], []
        filename.append(results['images']['CAM2']['img_path'])
        lidar2cam.append(results['images']['CAM2']['lidar2cam'])
        lidar2cam_array = np.array(results['images']['CAM2']['lidar2cam']).astype(np.float32)
        lidar2cam_rot = lidar2cam_array[:3, :3]
        lidar2cam_trans = lidar2cam_array[:3, 3:4]
        camera2lidar = np.eye(4)
        camera2lidar[:3, :3] = lidar2cam_rot.T
        camera2lidar[:3,3:4] = -1 * np.matmul(lidar2cam_rot.T, lidar2cam_trans.reshape(3, 1))
        cam2lidar.append(camera2lidar)
        cam2img_array = np.eye(4).astype(np.float32)
        cam2img_array[:3, :3] = np.array(results['images']['CAM2']['cam2img'])[:3, :3].astype(np.float32)
        cam2img.append(cam2img_array)
        lidar2img.append(cam2img_array @ lidar2cam_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['cam2lidar'] = np.stack(cam2lidar, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [get(name, backend_args=self.backend_args) for name in filename]
        imgs = [mmcv.imfrombytes(img_byte,flag=self.color_type,backend='pillow',channel_order='rgb') for img_byte in img_bytes]

        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs]
        img = np.stack(imgs,axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
    # unravel to list, see `DefaultFormatBundle` in formating.py
    # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
    # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels, dtype=np.float32),std=np.ones(num_channels, dtype=np.float32),to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results
        