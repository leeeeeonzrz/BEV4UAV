# Full BEV4UAV configuration.
# Enable the auxiliary IoU quality predictor supervised by VarifocalLoss.
_base_ = ['./fusion_csf_afrf_mfpn.py']

model = dict(bbox_head=dict(loss_iou=dict(loss_weight=1.0)))
work_dir = 'work_dirs/bev4uav'
