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
        # self.Level1_large_1 = nn.Sequential(
        #     DoubleConv(3, 64, mode=2), # 9*9
        #     DoubleConv(64, 64, mode=2),
        # )
        # self.Level1_large_2 = nn.Sequential(
        #     DoubleConv(3, 64, mode=3), # 11*11
        #     DoubleConv(64, 64, mode=3),
        # )
        # Level 2
        self.Level2_1 = nn.Sequential(
            DoubleConv(128, 64, mode=0),
            DoubleConv(64, 64, mode=0),
        )
        self.Level2_2 = nn.Sequential(
            DoubleConv(128, 64, mode=1),
            DoubleConv(64, 64, mode=1),
        )
        # self.Level2_large_1 = nn.Sequential(
        #     DoubleConv(128, 64, mode=2),
        #     DoubleConv(64, 64, mode=2),
        # )
        # self.Level2_large_2 = nn.Sequential(
        #     DoubleConv(128, 64, mode=3),
        #     DoubleConv(64, 64, mode=3),
        # )
        # Level 3
        self.Level3_1 = nn.Sequential(
            DoubleConv(128, 64, mode=0),
            DoubleConv(64, 64, mode=0),
        )
        self.Level3_2 = nn.Sequential(
            DoubleConv(128, 64, mode=1),
            DoubleConv(64, 64, mode=1),
        )
        # self.Level3_large_1 = nn.Sequential(
        #     DoubleConv(128, 64, mode=2),
        #     DoubleConv(64, 64, mode=2),
        # )
        # self.Level3_large_2 = nn.Sequential(
        #     DoubleConv(128, 64, mode=3),
        #     DoubleConv(64, 64, mode=3),
        # )
        self.Down1_conv = DoubleConv(3, 64)
        self.Down1_pool = nn.MaxPool2d(2)
        self.Down2_conv = DoubleConv(64, 128)
        self.Down2_pool = nn.MaxPool2d(2)
        self.Down3_conv = DoubleConv(128, 256)
        self.Down3_pool = nn.MaxPool2d(2)
        self.Down4_conv = DoubleConv(256, 512)
        self.Down4_pool = nn.MaxPool2d(2)
        self.Conv5 = nn.Sequential(
            DoubleConv(512, 1024)
        )
        self.Up4_convT = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.Up4_conv = DoubleConv(1024, 512)
        self.Up3_convT = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.Up3_conv = DoubleConv(512, 256)
        self.Up2_convT = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.Up2_conv = DoubleConv(256, 128)
        self.Up1_convT = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.Up1_conv = DoubleConv(128, 64)
        # Hiding
        # self.hiding_1_1 = nn.Sequential(
        #     DoubleConv(192, 64, mode=3),
        #     DoubleConv(64, 64, mode=3),
        # )
        # self.hiding_1_2 = nn.Sequential(
        #     DoubleConv(192, 64, mode=3),
        #     DoubleConv(64, 64, mode=3),
        # )
        # self.hiding_2_1 = nn.Sequential(
        #     DoubleConv(128, 64, mode=2),
        #     DoubleConv(64, 64, mode=2),
        # )
        # self.hiding_2_2 = nn.Sequential(
        #     DoubleConv(128, 64, mode=2),
        #     DoubleConv(64, 64, mode=2),
        # )
        # self.hiding_3_1 = nn.Sequential(
        #     DoubleConv(128, 64, mode=1),
        #     DoubleConv(64, 64, mode=1),
        # )
        # self.hiding_3_2 = nn.Sequential(
        #     DoubleConv(128, 64, mode=1),
        #     DoubleConv(64, 64, mode=1),
        # )
        # self.hiding_4_1 = nn.Sequential(
        #     DoubleConv(128, 64, mode=0),
        #     DoubleConv(64, 64, mode=0),
        # )
        # self.hiding_4_2 = nn.Sequential(
        #     DoubleConv(128, 64, mode=0),
        #     DoubleConv(64, 64, mode=0),
        # )
        self.hide_Down1_conv = DoubleConv(192, 64)
        self.hide_Down1_pool = nn.MaxPool2d(2)
        self.hide_Down2_conv = DoubleConv(64, 128)
        self.hide_Down2_pool = nn.MaxPool2d(2)
        self.hide_Down3_conv = DoubleConv(128, 256)
        self.hide_Down3_pool = nn.MaxPool2d(2)
        self.hide_Down4_conv = DoubleConv(256, 512)
        self.hide_Down4_pool = nn.MaxPool2d(2)
        self.hide_Conv5 = nn.Sequential(
            DoubleConv(512, 1024)
        )
        self.hide_Up4_convT = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.hide_Up4_conv = DoubleConv(1024, 512)
        self.hide_Up3_convT = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.hide_Up3_conv = DoubleConv(512, 256)
        self.hide_Up2_convT = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.hide_Up2_conv = DoubleConv(256, 128)
        self.hide_Up1_convT = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.hide_Up1_conv = DoubleConv(128, 64)
        self.finalH = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, padding=0))
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
        # 低维度
        l1_1 = self.Level1_1(p)
        l1_2 = self.Level1_2(p)
        l1 = torch.cat([l1_1, l1_2], dim=1)
        l2_1 = self.Level2_1(l1)
        l2_2 = self.Level2_2(l1)
        l2 = torch.cat([l2_1, l2_2], dim=1)
        l3_1 = self.Level3_1(l2)
        l3_2 = self.Level3_2(l2)
        l3 = torch.cat([l3_1, l3_2], dim=1)
        # 高维度
        down1_c = self.Down1_conv(p)
        down1_p = self.Down1_pool(down1_c)
        down2_c = self.Down2_conv(down1_p)
        down2_p = self.Down2_pool(down2_c)
        down3_c = self.Down3_conv(down2_p)
        down3_p = self.Down3_pool(down3_c)
        down4_c = self.Down4_conv(down3_p)
        down4_p = self.Down4_pool(down4_c)
        mid = self.Conv5(down4_p)
        up4_convT = self.Up4_convT(mid)
        merge4 = torch.cat([up4_convT, down4_c], dim=1)
        up4_conv = self.Up4_conv(merge4)
        up3_convT = self.Up3_convT(up4_conv)
        merge3 = torch.cat([up3_convT, down3_c], dim=1)
        up3_conv = self.Up3_conv(merge3)
        up2_convT = self.Up2_convT(up3_conv)
        merge2 = torch.cat([up2_convT, down2_c], dim=1)
        up2_conv = self.Up2_conv(merge2)
        up1_convT = self.Up1_convT(up2_conv)
        merge1 = torch.cat([up1_convT, down1_c], dim=1)
        up1_conv = self.Up1_conv(merge1)
        l3_cat = torch.cat([l3, up1_conv], dim=1)
        # Hiding
        hide_down1_c = self.hide_Down1_conv(l3_cat)
        hide_down1_p = self.hide_Down1_pool(hide_down1_c)
        hide_down2_c = self.hide_Down2_conv(hide_down1_p)
        hide_down2_p = self.hide_Down2_pool(hide_down2_c)
        hide_down3_c = self.hide_Down3_conv(hide_down2_p)
        hide_down3_p = self.hide_Down3_pool(hide_down3_c)
        hide_down4_c = self.hide_Down4_conv(hide_down3_p)
        hide_down4_p = self.hide_Down4_pool(hide_down4_c)
        hide_mid = self.hide_Conv5(hide_down4_p)
        hide_up4_convT = self.hide_Up4_convT(hide_mid)
        hide_merge4 = torch.cat([hide_up4_convT, hide_down4_c], dim=1)
        hide_up4_conv = self.hide_Up4_conv(hide_merge4)
        hide_up3_convT = self.hide_Up3_convT(hide_up4_conv)
        hide_merge3 = torch.cat([hide_up3_convT, hide_down3_c], dim=1)
        hide_up3_conv = self.hide_Up3_conv(hide_merge3)
        hide_up2_convT = self.hide_Up2_convT(hide_up3_conv)
        hide_merge2 = torch.cat([hide_up2_convT, hide_down2_c], dim=1)
        hide_up2_conv = self.hide_Up2_conv(hide_merge2)
        hide_up1_convT = self.hide_Up1_convT(hide_up2_conv)
        hide_merge1 = torch.cat([hide_up1_convT, hide_down1_c], dim=1)
        hide_up1_conv = self.hide_Up1_conv(hide_merge1)
        # hiding_1_1 = self.hiding_1_1(l3_cat)
        # hiding_1_2 = self.hiding_1_2(l3_cat)
        # hiding_1 = torch.cat([hiding_1_1, hiding_1_2], dim=1)
        # hiding_2_1 = self.hiding_2_1(hiding_1)
        # hiding_2_2 = self.hiding_2_2(hiding_1)
        # hiding_2 = torch.cat([hiding_2_1, hiding_2_2], dim=1)
        # hiding_3_1 = self.hiding_3_1(hiding_2)
        # hiding_3_2 = self.hiding_3_2(hiding_2)
        # hiding_3 = torch.cat([hiding_3_1, hiding_3_2], dim=1)
        # hiding_4_1 = self.hiding_4_1(hiding_3)
        # hiding_4_2 = self.hiding_4_2(hiding_3)
        # hiding_4 = torch.cat([hiding_4_1, hiding_4_2], dim=1)
        out = self.finalH(hide_up1_conv)

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