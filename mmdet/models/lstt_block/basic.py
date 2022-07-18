import torch
import torch.nn.functional as F
import math
from torch import nn
from .gct import GCT

class GroupNorm1D(nn.Module):
    def __init__(self, indim, groups=8):
        super().__init__()
        self.gn = nn.GroupNorm(groups, indim)

    def forward(self, x):
        return self.gn(x.permute(1, 2, 0)).permute(2, 0, 1)


class GNActGCTDWConv2d(nn.Module):
    def __init__(self, indim):
        super().__init__()
        self.gn = nn.GroupNorm(32, indim)
        self.GCT = GCT(indim)
        self.conv = nn.Conv2d(indim, indim, 5, dilation=1,
                              padding=2, groups=indim, bias=False)

    def forward(self, x, size_2d):
        h, w = size_2d
        _, bs, c = x.size()
        x = x.view(h, w, bs, c).permute(2, 3, 0, 1)
        x = self.gn(x)
        x = F.gelu(x)
        x = self.GCT(x)
        x = self.conv(x)
        x = x.view(bs, c, h * w).permute(2, 0, 1)
        return x

class ConvGN(nn.Module):
    def __init__(self, indim, outdim, kernel_size):
        super().__init__()

        # gn_groups = outdim // 4
        # if gn_groups > 32:
        #     gn_groups = 32
        gn_groups = 8

        self.gct = GCT(indim)
        self.conv = nn.Conv2d(indim, outdim, kernel_size, padding=kernel_size // 2)
        self.gn = nn.GroupNorm(gn_groups, outdim)

    def forward(self, x):
        return self.gn(self.conv(self.gct(x)))

def seq_to_2d(tensor, size_2d):
    h, w = size_2d
    _, n, c = tensor.size()
    tensor = tensor.view(h, w, n, c).permute(2, 3, 0, 1).contiguous()
    return tensor

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
