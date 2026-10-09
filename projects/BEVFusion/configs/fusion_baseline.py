# Camera-LiDAR baseline with frozen pretrained branches.
# Both user-supplied checkpoints are required; no weights are distributed.
_base_ = ['./lidar.py']

load_from = '{{$BEV4UAV_LIDAR_CHECKPOINT:checkpoints/lidar.pth}}'

input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False),
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        frozen_stages=1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='{{$BEV4UAV_IMAGE_CHECKPOINT:checkpoints/swin_tiny.pth}}')),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    view_transform=dict(
        type='AerialDepthLSSTransformGN',
        in_channels=256,
        out_channels=80,
        image_size=[768, 1024],
        feature_size=[96, 128],
        xbound=[-64.0, 64.0, 1.0],
        ybound=[0.0, 64.0, 1.0],
        zbound=[-30.0, 0.0, 30.0],
        dbound=[1.0, 50.0, 0.5],
        downsample=2,
        coord_channels=32,
        gate_init=0.0,
        image_gate_limit=0.10,
        depth_prior_limit=0.10,
        context_prior_limit=0.10),
    fusion_layer=dict(
        type='ConvFuser',
        in_channels=[80, 128],
        out_channels=128),
    bbox_head=dict(loss_iou=dict(loss_weight=0.0)))

train_pipeline = [
    dict(
        type='BEVLoadKittiImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='AerialObliqueImageAug3D',
        final_dim=[768, 1024],
        resize_lim=[0.50, 0.56],
        bot_pct_lim=[0.0, 0.0],
        top_crop_pct_lim=[0.02, 0.05],
        horizontal_crop_bias=0.52,
        horizontal_crop_noise=0.04,
        artifact_masks=[],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=True),
    dict(type='PointsRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='ObjectNameFilter', classes=_base_.class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'lidar_aug_matrix',
            'num_pts_feats'
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadKittiImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='AerialObliqueImageAug3D',
        final_dim=[768, 1024],
        resize_lim=[0.54, 0.54],
        bot_pct_lim=[0.0, 0.0],
        top_crop_pct_lim=[0.03, 0.03],
        horizontal_crop_bias=0.52,
        horizontal_crop_noise=0.0,
        artifact_masks=[],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=_base_.data_root,
        ann_file='kitti_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=_base_.metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=_base_.data_prefix,
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=_base_.data_root,
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=_base_.metainfo,
        modality=input_modality,
        data_prefix=_base_.data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = val_dataloader

lr = 3.0e-6
train_cfg = dict(by_epoch=True, max_epochs=40, val_begin=5, val_interval=5)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),
            'img_neck': dict(lr_mult=0.5),
            'view_transform': dict(lr_mult=1.0),
            'fusion_layer': dict(lr_mult=0.5),
            'pts_middle_encoder': dict(lr_mult=0.0),
            'pts_backbone': dict(lr_mult=0.0),
            'pts_neck': dict(lr_mult=0.0),
            'bbox_head': dict(lr_mult=0.0),
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=5, norm_type=2))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.5,
        begin=0,
        end=3,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=37,
        eta_min=lr * 0.2,
        begin=3,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True),
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(type='CheckpointHook', interval=5))

custom_hooks = [
    dict(
        type='ModuleFreezeHook',
        module_names=[
            'img_backbone', 'pts_middle_encoder', 'pts_backbone', 'pts_neck',
            'bbox_head'
        ])
]

work_dir = 'work_dirs/fusion_baseline'
