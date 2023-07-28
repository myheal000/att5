from mmdet.models.builder import DETECTORS
from .obb_two_stage import OBBTwoStageDetector
#from .obb_two_stagehp import OBBTwoStageDetectorHp


@DETECTORS.register_module()
class OrientedRCNN(OBBTwoStageDetector):

    def __init__(self,
                 backbone,
                neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(OrientedRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
           test_cfg=test_cfg,
            pretrained=pretrained)

# class OrientedRCNN(OBBTwoStageDetectorHp):

#     def __init__(self,
#                  backbone,
#                  neck=None,
#                  rpn_head=None,
#                  roi_head=None,
#                  train_cfg=None,
#                  test_cfg=None,
#                  pretrained=None):
#         super(OrientedRCNN, self).__init__(
#             backbone=backbone,
#             neck=neck,
#             rpn_head=rpn_head,
#             roi_head=roi_head,
#             train_cfg=train_cfg,
#             test_cfg=test_cfg,
#             pretrained=pretrained)
