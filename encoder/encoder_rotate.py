import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
import util
from encoder.encoder_rotate_prep import Encoder_rotate_prep

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

class Encoder_rotate(nn.Module):
    def __init__(self,config=GlobalConfig()):
        super(Encoder_rotate, self).__init__()
        self.config = config
        # Prep
        self.Prepare = Encoder_rotate_prep(config)

        # Hiding
        self.initialH3 = nn.Sequential(
            DoubleConv(192 + 3, 64, mode=0),
            DoubleConv(64, 64, mode=0),
        )
        self.initialH4 = nn.Sequential(
            DoubleConv(192 + 3, 64, mode=1),
            DoubleConv(64, 64, mode=1),
        )
        self.initialH5 = nn.Sequential(
            DoubleConv(192 + 3, 64, mode=2),
            DoubleConv(64, 64, mode=2),
        )
        self.finalH3 = nn.Sequential(
            DoubleConv(192, 64, mode=0),
            DoubleConv(64, 64, mode=0),
        )
        self.finalH4 = nn.Sequential(
            DoubleConv(192, 64, mode=1),
            DoubleConv(64, 64, mode=1),
        )
        self.finalH5 = nn.Sequential(
            DoubleConv(192, 64, mode=2),
            DoubleConv(64, 64, mode=2),
        )
        self.finalH = nn.Sequential(
            nn.Conv2d(192, 3, kernel_size=1, padding=0))

    def forward(self, p, flip):
        # Conduct Image Rotation

        # Cover_flip_xy = flip(Cover_flip_y, 3).detach()
        p_rotate = self.Prepare(flip[0], flip[1])
        # p_rotate2 = self.Prepare(flip[1])
        # p_rotate3 = self.Prepare(flip[2])

        mid = torch.cat((p_rotate, p), 1)
        # mid = torch.cat((p_rotate1, p), 1)
        h1 = self.initialH3(mid)
        h2 = self.initialH4(mid)
        h3 = self.initialH5(mid)
        h1_cat = torch.cat((h1, h2, h3), 1)
        h4 = self.finalH3(h1_cat)
        h5 = self.finalH4(h1_cat)
        h6 = self.finalH5(h1_cat)
        h2_cat = torch.cat((h4, h5, h6), 1)
        out = self.finalH(h2_cat)
        return out
