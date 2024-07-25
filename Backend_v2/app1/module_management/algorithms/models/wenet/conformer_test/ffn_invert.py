# --------------------------------------------------------
# PSLT
# --------------------------------------------------------

# from module import SepConv2d
import torch
import torch.nn as nn
# import math
# from torch import einsum, sqrt
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
# import torch.nn.functional as F
# from einops import rearrange
# from module import SepConv2d
# import math
# from mmcv_custom import load_checkpoint
# import os
# from visualizer import get_local


class Mlp_Light(nn.Module):
    def __init__(self, in_features,  hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., reduce_scale=4, **kargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # hidden_features = int(hidden_features*reduce_scale)
        hidden_features = hidden_features #参照原始conformer设定2048,这里256
        #可以尝试缩小4倍并使用残差连接中间
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, hidden_features, 1, 1),
            act_layer(),
            nn.BatchNorm1d(hidden_features),
            nn.Conv1d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features),
            act_layer(),
            nn.BatchNorm1d(hidden_features),
            nn.Conv1d(hidden_features, out_features, 1, 1),
            nn.BatchNorm1d(out_features),
        )
        
        
    def forward(self, x):
        # H, W = self.input_resolution
        # B, L, C = x.shape
        # assert H*W == L , "input feature has wrong size"

        # x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)


        return x


