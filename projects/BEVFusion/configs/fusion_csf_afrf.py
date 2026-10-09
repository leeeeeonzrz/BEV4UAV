# Add the aerial residual fusion module.
_base_ = ['./fusion_csf.py']

model = dict(
    fusion_layer=dict(
        type='LidarResidualGNEMAFuser',
        ema_factor=32,
        gate_init=0.0,
        max_residual_scale=0.12))

work_dir = 'work_dirs/fusion_csf_afrf'
