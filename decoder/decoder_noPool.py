import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv


class Decoder_noPool(nn.Module):
    def __init__(self, config=GlobalConfig()):
        super(Decoder_noPool, self).__init__()
        self.config = config


        # Level 5
        self.hiding_1_1 = nn.Sequential(
            DoubleConv(3, 256),
            # DoubleConv(256, 256),
        )
        self.hiding_1_2 = nn.Sequential(
            DoubleConv(3, 128),
            # DoubleConv(128, 128),
        )
        # self.hiding_1_3 = nn.Sequential(
        #     DoubleConv(3, 40),
        #     DoubleConv(40, 40),
        # )
        # self.hiding_2_1 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # self.hiding_2_2 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # self.hiding_2_3 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # # Level 4
        # self.invertLevel4_1 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # self.invertLevel4_2 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # self.invertLevel4_3 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # Level 3
        self.invertLevel3_1 = nn.Sequential(
            DoubleConv(256, 128),
            # DoubleConv(128, 128),
        )
        # self.invertLevel3_2 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # self.invertLevel3_3 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # Level 2
        self.invertLevel2_1 = nn.Sequential(
            DoubleConv(256, 128),
            # DoubleConv(128, 128),
        )
        # self.invertLevel2_2 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # self.invertLevel2_3 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # Level 1
        self.invertLevel1_1 = nn.Sequential(
            DoubleConv(256, 128),
            # DoubleConv(128, 128),
        )
        # self.invertLevel1_2 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # self.invertLevel1_3 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        self.final = nn.Conv2d(128 + 3, 3, kernel_size=1, padding=0)
        #self.final = nn.Conv2d(120, 3, kernel_size=1, padding=0)
        #self.final = DoubleConv(120, 3, disable_last_activate=True)

    def forward(self, p):

        # Level 5
        hiding = self.hiding_1_1(p)
        hiding_2 = self.hiding_1_2(p)
        # hiding_1_3 = self.hiding_1_3(p)
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
        il3 = self.invertLevel3_1(hiding)
        # il3_2 = self.invertLevel3_2(hiding_2)
        # il3_3 = self.invertLevel3_3(hiding_2)
        # il3 = torch.cat([il3_1, il3_2, il3_3], dim=1)
        # Level 2
        il3_cat = torch.cat([il3, hiding_2], dim=1)
        il2 = self.invertLevel2_1(il3_cat)
        # il2_2 = self.invertLevel2_2(il3)
        # il2_3 = self.invertLevel2_3(il3)
        # il2 = torch.cat([il2_1, il2_2, il2_3], dim=1)
        # Level 1
        il2_cat = torch.cat([il2, il3], dim=1)
        il1 = self.invertLevel1_1(il2_cat)
        # il1_2 = self.invertLevel1_2(il2)
        # il1_3 = self.invertLevel1_3(il2)
        # il1 = torch.cat([il1_1, il1_2, il1_3], dim=1)
        il1_cat = torch.cat([il1, p], dim=1)
        out = self.final(il1_cat)

        return out
