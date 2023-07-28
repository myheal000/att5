import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import (build_conv_layer,  constant_init,build_norm_layer,
                      kaiming_init)
  
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import ResLayer




class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        layers = [
                  nn.Conv2d(in_channels, out_channels, kernel_size=1, bias = False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True),
        ]

        if stride ==2:

            layers2 = [
            nn.Conv2d(out_channels, out_channels,stride= 1, kernel_size=3, padding=1, bias= False),
            space_to_depth(),   # the output of this will result in 4*out_channels
            nn.BatchNorm2d(4*out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(4*out_channels, out_channels* BottleNeck.expansion, kernel_size=1, bias = False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),                       
            ]

        else:

            layers2 = [
            nn.Conv2d(out_channels, out_channels,stride= stride, kernel_size=3, padding=1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels* BottleNeck.expansion, kernel_size=1, bias = False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),                       
            ]

        layers.extend(layers2)

        self.residual_function = torch.nn.Sequential(*layers)

		
     
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

@BACKBONES.register_module()
class ResNet_spd(nn.Module):
    arch_settings = {
        
        50: (BottleNeck, (3, 4, 6, 3)),
        101: (BottleNeck, (3, 4, 23, 3)),
        152: (BottleNeck, (3, 8, 36, 3))
    }

    def __init__(self,  
                 depth,
                 in_channels=64,
                 num_stages=4,
                 base_channels=64,
                 strides=(1, 2, 2, 2),
                 out_indices=(0, 1, 2, 3), 
                #norm_cfg=dict(type='BN', requires_grad=True),
                #norm_eval=True,
                #style='pytorch',
                #deep_stem=False,
                #avg_down=False,
                #frozen_stages=-1,
                #conv_cfg=None
                 ):
        super(ResNet_spd, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
       
        self.depth = depth
        self.in_channels = in_channels
        self.num_stages = num_stages
        self.base_channels = base_channels
        self.strides = strides
        self.out_indices = out_indices
        #self.norm_cfg = norm_cfg      
        #self.norm_eval = norm_eval
        #self.style = style
        #self.deep_stem = deep_stem
        #self.avg_down = avg_down
        # self.frozen_stages = frozen_stages
        #self.conv_cfg = conv_cfg
       
        #self.with_cp = with_cp
        self.block, stage_blocks = self.arch_settings[depth]
        #这里的block应该是bottleneck,stage_blocks应该是列表{3，4，6，3}
        self.stage_blocks = stage_blocks[:num_stages]
        #这里的num_stage=4,是不是相当于又加了一层保障
        #self.inplanes = stem_channels
        # self.norm1_name, norm1 = build_norm_layer(norm_cfg, self.in_channels, postfix=1)
        
        
        self.conv1 = Focus(3, 64, k=1,s=1)
        """ self.conv1 = build_conv_layer(
            inplanes=3, 
            planes = 64,
            kernel_size=1,
            stride=1,
            bias=False) """
        #self.add_module(self.norm1_name, norm1) 
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            planes = base_channels * 2**i
            res_layer = self._make_layer(
                block=self.block,
                out_channels=planes,
                num_blocks=num_blocks,
                stride=stride,
                )
            self.in_channels = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        
            
       

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')
    def forward(self, x):
        x = self.conv1(x)
       

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
    # def train(self, mode=True):
    #     """Convert the model into training mode while keep normalization layer
    #     freezed"""
    #     super(ResNet_spd, self).train(mode)
    #     self._freeze_stages()
    #     if mode and self.norm_eval:
    #         for m in self.modules():
    #             # trick: eval have effect on BatchNorm only
    #             if isinstance(m, _BatchNorm):
    #                 m.eval()
    # @property
    # def norm1(self):
    #     return getattr(self, self.norm1_name)  
    # def _freeze_stages(self):
    #     if self.frozen_stages >= 0:
    #         if self.deep_stem:
    #             self.stem.eval()
    #             for param in self.stem.parameters():
    #                 param.requires_grad = False
    #         else:
    #             self.norm1.eval()
    #             for m in [self.conv1, self.norm1]:
    #                 for param in m.parameters():
    #                     param.requires_grad = False

    #     for i in range(1, self.frozen_stages + 1):
    #         m = getattr(self, f'layer{i}')
    #         m.eval()
    #         for param in m.parameters():
    #             param.requires_grad = False


