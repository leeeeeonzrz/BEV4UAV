# LiDAR stage on data preprocessed externally with CSF.
# Set BEV4UAV_DATA_ROOT to that dataset before training.
_base_ = ['./lidar.py']

work_dir = 'work_dirs/lidar_csf'
