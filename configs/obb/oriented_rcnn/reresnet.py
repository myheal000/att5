_base_ = './faster_rcnn_orpn_r50_fpn_3x_hrsc.py'

model = dict(
    type='OrientedRCNN',
    pretrained='/kaggle/input/recheckpoint/re_resnet50_c8_batch256-25b16846.pth',
    backbone=dict(
        type='ReResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'))
