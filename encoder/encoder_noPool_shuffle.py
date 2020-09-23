import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
import util

class EncoderNetwork_noPool_shuffle(nn.Module):
    def __init__(self,config=GlobalConfig()):
        super(EncoderNetwork_noPool_shuffle, self).__init__()
        self.config = config
        # self.init = DoubleConv(3, 40)
        # Level 1
        self.Level1_1 = nn.Sequential(
            DoubleConv(3, 64, mode=0), # 3*3
            DoubleConv(64, 64, mode=0),
        )
        self.Level1_2 = nn.Sequential(
            DoubleConv(3, 64, mode=1), # 5*5 3*3
            DoubleConv(64, 64, mode=1),
        )
        self.Level1_large_1 = nn.Sequential(
            DoubleConv(3, 64, mode=2), # 9*9
            DoubleConv(64, 64, mode=2),
        )
        self.Level1_large_2 = nn.Sequential(
            DoubleConv(3, 64, mode=3), # 11*11
            DoubleConv(64, 64, mode=3),
        )
        # Level 2
        self.Level2_1 = nn.Sequential(
            DoubleConv(128, 64, mode=0),
            DoubleConv(64, 64, mode=0),
        )
        self.Level2_2 = nn.Sequential(
            DoubleConv(128, 64, mode=1),
            DoubleConv(64, 64, mode=1),
        )
        self.Level2_large_1 = nn.Sequential(
            DoubleConv(128, 64, mode=2),
            DoubleConv(64, 64, mode=2),
        )
        self.Level2_large_2 = nn.Sequential(
            DoubleConv(128, 64, mode=3),
            DoubleConv(64, 64, mode=3),
        )
        # Level 3
        self.Level3_1 = nn.Sequential(
            DoubleConv(128, 64, mode=0),
            DoubleConv(64, 64, mode=0),
        )
        self.Level3_2 = nn.Sequential(
            DoubleConv(128, 64, mode=1),
            DoubleConv(64, 64, mode=1),
        )
        self.Level3_large_1 = nn.Sequential(
            DoubleConv(128, 64, mode=2),
            DoubleConv(64, 64, mode=2),
        )
        self.Level3_large_2 = nn.Sequential(
            DoubleConv(128, 64, mode=3),
            DoubleConv(64, 64, mode=3),
        )

        # # Level 4
        # self.Level4_1 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # self.Level4_2 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # self.Level4_3 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # Level 5
        self.hiding_1_1 = nn.Sequential(
            DoubleConv(256, 64, mode=2),
            DoubleConv(64, 64, mode=2),
        )
        self.hiding_1_2 = nn.Sequential(
            DoubleConv(256, 64, mode=3),
            DoubleConv(64, 64, mode=3),
        )
        self.hiding_2_1 = nn.Sequential(
            DoubleConv(128, 64, mode=2),
            DoubleConv(64, 64, mode=2),
        )
        self.hiding_2_2 = nn.Sequential(
            DoubleConv(128, 64, mode=3),
            DoubleConv(64, 64, mode=3),
        )
        self.hiding_3_1 = nn.Sequential(
            DoubleConv(128, 64, mode=2),
            DoubleConv(64, 64, mode=2),
        )
        self.hiding_3_2 = nn.Sequential(
            DoubleConv(128, 64, mode=3),
            DoubleConv(64, 64, mode=3),
        )
        self.finalH = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=1, padding=0))
        # self.hiding_1_3 = nn.Sequential(
        #     DoubleConv(120+768, 40),
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
        # self.invertLevel3_1 = nn.Sequential(
        #     DoubleConv(256, 128),
        #     DoubleConv(128, 128)
        # )
        # self.invertLevel3_2 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # self.invertLevel3_3 = nn.Sequential(
        #     DoubleConv(120, 40),
        #     DoubleConv(40, 40),
        # )
        # Level 2
        # self.invertLevel2_1 = nn.Sequential(
        #     DoubleConv(128+128, 128),
        #     # DoubleConv(128, 128)
        # )
        # self.invertLevel2_2 = nn.Sequential(
        #     DoubleConv(240, 40),
        #     DoubleConv(40, 40),
        # )
        # self.invertLevel2_3 = nn.Sequential(
        #     DoubleConv(240, 40),
        #     DoubleConv(40, 40),
        # )
        # Level 1
        # self.invertLevel1_1 = nn.Sequential(
        #     DoubleConv(128+64, 128),
        #     # DoubleConv(128, 128)
        # )
        # self.invertLevel1_2 = nn.Sequential(
        #     DoubleConv(240, 40),
        #     DoubleConv(40, 40),
        # )
        # self.invertLevel1_3 = nn.Sequential(
        #     DoubleConv(240, 40),
        #     DoubleConv(40, 40),
        # )
        # self.final = nn.Conv2d(128+3, 3, kernel_size=1, padding=0)
        # self.final = DoubleConv(120, 3,disable_last_activate=True)


    def forward(self, p):
        # Level 1
        l1_1 = self.Level1_1(p)
        l1_2 = self.Level1_2(p)
        l1 = torch.cat([l1_1, l1_2], dim=1)
        l1_large_1 = self.Level1_large_1(p)
        l1_large_2 = self.Level1_large_2(p)
        l1_large = torch.cat([l1_large_1, l1_large_2], dim=1)
        # Level 2
        l2_1 = self.Level2_1(l1)
        l2_2 = self.Level2_2(l1)
        l2 = torch.cat([l2_1, l2_2], dim=1)
        l2_large_1 = self.Level2_large_1(l1_large)
        l2_large_2 = self.Level2_large_2(l1_large)
        l2_large = torch.cat([l2_large_1, l2_large_2], dim=1)
        l2_cat = torch.cat([l2, l2_large], dim=1)
        # Level 3
        l3_1 = self.Level3_1(l2)
        l3_2 = self.Level3_2(l2)
        l3 = torch.cat([l3_1, l3_2], dim=1)
        l3_large_1 = self.Level3_large_1(l2_large)
        l3_large_2 = self.Level3_large_2(l2_large)
        l3_large = torch.cat([l3_large_1, l3_large_2], dim=1)
        l3_cat = torch.cat([l3, l3_large], dim=1)
        hiding_1_1 = self.hiding_1_1(l3_cat)
        hiding_1_2 = self.hiding_1_2(l3_cat)
        hiding_1 = torch.cat([hiding_1_1, hiding_1_2], dim=1)
        hiding_2_1 = self.hiding_2_1(hiding_1)
        hiding_2_2 = self.hiding_2_2(hiding_1)
        hiding_2 = torch.cat([hiding_2_1, hiding_2_2], dim=1)
        hiding_3_1 = self.hiding_3_1(hiding_2)
        hiding_3_2 = self.hiding_3_2(hiding_2)
        hiding_3 = torch.cat([hiding_3_1, hiding_3_2], dim=1)
        out = self.finalH(hiding_3)

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