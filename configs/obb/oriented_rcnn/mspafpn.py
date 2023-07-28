_base_ = './faster_rcnn_orpn_r50_fpn_1x_dota10.py'
model = dict(
    neck=
        dict(type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5))