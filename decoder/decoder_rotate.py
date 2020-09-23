import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
import util


class Decoder_rotate(nn.Module):
    def __init__(self, config=GlobalConfig()):
        super(Decoder_rotate, self).__init__()
        self.config = config
        self.initialR3 = nn.Sequential(
            DoubleConv(3, 64, mode=0),
            DoubleConv(64, 64, mode=0),
        )
        self.initialR4 = nn.Sequential(
            DoubleConv(3, 64, mode=1),
            DoubleConv(64, 64, mode=1),
        )
        self.initialR5 = nn.Sequential(
            DoubleConv(3, 64, mode=2),
            DoubleConv(64, 64, mode=2),
        )
        self.finalR3 = nn.Sequential(
            DoubleConv(192, 64, mode=0),
            DoubleConv(64, 64, mode=0),
        )
        self.finalR4 = nn.Sequential(
            DoubleConv(192, 64, mode=1),
            DoubleConv(64, 64, mode=1),
        )
        self.finalR5 = nn.Sequential(
            DoubleConv(192, 64, mode=2),
            DoubleConv(64, 64, mode=2),
        )
        self.finalR = nn.Sequential(
            nn.Conv2d(192, 3, kernel_size=1, padding=0))

    def forward(self, r):
        r1 = self.initialR3(r)
        r2 = self.initialR4(r)
        r3 = self.initialR5(r)
        mid = torch.cat((r1, r2, r3), 1)
        r4 = self.finalR3(mid)
        r5 = self.finalR4(mid)
        r6 = self.finalR5(mid)
        mid2 = torch.cat((r4, r5, r6), 1)
        out = self.finalR(mid2)

        return out

        # # shuffled data
        # for i in range(16):
        #     for j in range(16):
        #         portion = p[:, :, 16*i:16*(i+1), 16*j:16*(j+1)]
        #         portion = portion.repeat(1, 1, 16, 16)
        #         l3 = torch.cat([l3, portion], dim=1)
        #         # Test
        #         # imgs = [portion.data, p.data]
        #         # util.imshow(imgs, '(After Net 1) Fig.1 After EncodeAndAttacked Fig.2 Original', std=self.config.std,
        #         #             mean=self.config.mean)

        # hiding = self.hiding_1_1(l3)
        # hiding_1_2 = self.hiding_1_2(l3)
        # hiding_1_3 = self.hiding_1_3(l3)
        # hiding_1 = torch.cat([hiding_1_1, hiding_1_2, hiding_1_3], dim=1)
        # hiding_2_1 = self.hiding_2_1(hiding_1)
        # hiding_2_2 = self.hiding_2_2(hiding_1)
        # hiding_2_3 = self.hiding_2_3(hiding_1)
        # hiding_2 = torch.cat([hiding_2_1, hiding_2_2, hiding_2_3], dim=1)
        # # Level 4
        # il4_1 = self.invertLevel4_1(hiding_2)
        # il4_2 = self.invertLevel4_2(hiding_2)
        # il4_3 = self.invertLevel4_3(hiding_2)
        # il4 = torch.cat([il4_1, il4_2, il4_3], dim=1)
        # Level 3
        # il3 = self.invertLevel3_1(hiding)
        # il3_2 = self.invertLevel3_2(hiding_2)
        # il3_3 = self.invertLevel3_3(hiding_2)
        # il3 = torch.cat([il3_1, il3_2, il3_3], dim=1)
        # Level 2
        # il3_cat = torch.cat([il3, l2], dim=1)
        # il2 = self.invertLevel2_1(il3_cat)
        # il2_2 = self.invertLevel2_2(il3_cat)
        # il2_3 = self.invertLevel2_3(il3_cat)
        # il2 = torch.cat([il2_1, il2_2, il2_3], dim=1)
        # Level 1
        # il2_cat = torch.cat([il2, l1], dim=1)
        # il1 = self.invertLevel1_1(il2_cat)
        # il1_2 = self.invertLevel1_2(il2_cat)
        # il1_3 = self.invertLevel1_3(il2_cat)
        # il1 = torch.cat([il1_1, il1_2, il1_3], dim=1)
        # il1_cat = torch.cat([il1, p], dim=1)
        # out = self.final(il1_cat)
        #
        # return out