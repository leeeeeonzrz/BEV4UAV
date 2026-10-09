# Merge multi-scale image features before the view transform.
_base_ = ['./fusion_csf_afrf.py']

model = dict(img_neck=dict(type='MergedLSSFPN'))
work_dir = 'work_dirs/fusion_csf_afrf_mfpn'
