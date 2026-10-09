# Same fusion architecture on data preprocessed externally with CSF.
# Use the LiDAR checkpoint trained on the matching preprocessed dataset.
_base_ = ['./fusion_baseline.py']

load_from = '{{$BEV4UAV_LIDAR_CHECKPOINT:checkpoints/lidar_csf.pth}}'
work_dir = 'work_dirs/fusion_csf'
