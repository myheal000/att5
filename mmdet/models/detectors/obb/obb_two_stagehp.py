import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from .obb_base import OBBBaseDetector
from .obb_test_mixins import RotateAugRPNTestMixin
import os



def draw_feature_map1(features, img_path, save_dir = './work_dirs/feature_map/',name = None):
    '''
    :param features: 特征层。可以是单层，也可以是一个多层的列表
    :param img_path: 测试图像的文件路径
    :param save_dir: 保存生成图像的文件夹
    :return:
    '''
    img = cv2.imread(img_path)      #读取文件路径
    i=0
    if isinstance(features,torch.Tensor):   # 如果是单层
        features = [features]       # 转为列表
    for featuremap in features:     # 循环遍历
        heatmap = featuremap_2_heatmap1(featuremap)	#主要是这个，就是取特征层整个的求和然后平均，归一化
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap0 = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
        heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
        plt.imshow(heatmap0)  # ,cmap='gray' ，这里展示下可视化的像素值
        # plt.imshow(superimposed_img)  # ,cmap='gray'
        plt.close()	#关掉展示的图片
        # 下面是用opencv查看图片的
        # cv2.imshow("1",superimposed_img)
        # cv2.waitKey(0)     #这里通过安键盘取消显示继续运行。
        # cv2.destroyAllWindows()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, name + str(i) + '.png'), superimposed_img) #superimposed_img：保存的是叠加在原图上的图，也可以保存过程中其他的自己看看
        print(os.path.join(save_dir, name + str(i) + '.png'))
        i = i + 1

def featuremap_2_heatmap1(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    # heatmap = feature_map[:,0,:,:]*0    #
    heatmap = feature_map[:1, 0, :, :] * 0 #取一张图片,初始化为0
    for c in range(feature_map.shape[1]):   # 按通道
        heatmap+=feature_map[:1,c,:,:]      # 像素值相加[1,H,W]
    heatmap = heatmap.cpu().numpy()    #因为数据原来是在GPU上的
    heatmap = np.mean(heatmap, axis=0) #计算像素点的平均值,会下降一维度[H,W]

    heatmap = np.maximum(heatmap, 0)  #返回大于0的数[H,W]
    heatmap /= np.max(heatmap)      #/最大值来设置透明度0-1,[H,W]
    #heatmaps.append(heatmap)

    return heatmap
def feature_map_channel(features,img_path,save_dir = 'work_dirs/feature_map',name = 'noresbnsie2ltft_'):
	# 随便定义a,b,c,d去取对应的特征层，把通道数变换到最后一个维度，将计算的环境剥离由GPU变成CPU，tensor变为numpy
    a = torch.squeeze(features[0][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    b = torch.squeeze(features[1][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    c = torch.squeeze(features[2][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    d = torch.squeeze(features[3][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    img = cv2.imread(img_path)
    for j,x in enumerate([d]):
    				# x.shape[-1]：表示所有通道数，不想可视化这么多，可以自己写对应的数量
        for i in range(x.shape[-1]): 
            heatmap = x[:, :, i]
            # heatmap = np.maximum(heatmap, 0) #一个通道应该不用归一化了
            # heatmap /= np.max(heatmap)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            heatmap0 = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
            heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  
            superimposed_img = heatmap * 0.4 + img  # 将热力图应用于原始图像
            # plt.figure()  # 展示
            # plt.title(str(j))
            # plt.imshow(heatmap0) #, cmap='gray'
            # # plt.savefig(os.path.join(save_dir,  name+str(j)+str(i) + '.png'))
            # plt.close()
            cv2.imwrite(os.path.join(save_dir, name + str(j)+str(i) + '.png'), superimposed_img)



@DETECTORS.register_module()
class OBBTwoStageDetectorHp(OBBBaseDetector, RotateAugRPNTestMixin):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(OBBTwoStageDetectorHp, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(OBBTwoStageDetectorHp, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    #def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
      #  x = self.backbone(img)
       # if self.with_neck:
        #    x = self.neck(x)
        #return x
    def extract_feat(self, img,img_metas):
        """Directly extract features from the backbone+neck."""
        imgpath = img_metas[0]['filename']  # 主要是要图片的原始路径 
        filename = imgpath.split('/')[-1]
        imgname = filename.split('.')[0]
        x = self.backbone(img)
       
        #feature_map_channel(x,imgpath,name='chanel_')
        if self.with_neck:
            x = self.neck(x)
            draw_feature_map1(x,imgpath,name=imgname) #特征层，图片路径，保存的文件名
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        proposal_type = 'hbb'
        if self.with_rpn:
            proposal_type = getattr(self.rpn_head, 'bbox_type', 'hbb')
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )

        if proposal_type == 'hbb':
            proposals = torch.randn(1000, 4).to(img.device)
        elif proposal_type == 'obb':
            proposals = torch.randn(1000, 5).to(img.device)
        else:
            # poly proposals need to be generated in roi_head
            proposals = None
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_obboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_obboxes_ignore=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        super( OBBTwoStageDetectorHp, self).forward_train(img, img_metas)
        x = self.extract_feat(img, img_metas)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_type = getattr(self.rpn_head, 'bbox_type', 'hbb')
            target_bboxes = gt_bboxes if proposal_type == 'hbb' else gt_obboxes
            target_bboxes_ignore = gt_bboxes_ignore if proposal_type == 'hbb' \
                    else gt_obboxes_ignore

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                target_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=target_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_obboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_obboxes_ignore,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.rotate_aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
