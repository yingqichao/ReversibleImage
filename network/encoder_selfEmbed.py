import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv


class EncoderNetwork(nn.Module):
    def __init__(self, is_embed_message=True, config=GlobalConfig()):
        super(EncoderNetwork, self).__init__()
        self.config = config
        self.is_embed_message = is_embed_message
        # self.init = DoubleConv(3, 32)
        # Size: 256->128
        self.Down1_conv = DoubleConv(3,64)
        self.Down1_pool = nn.MaxPool2d(2)
        self.Down1_conv_low = DoubleConv(3, 16)
        self.Down1_pool_low = nn.MaxPool2d(2)
        # Size: 128->64
        self.Down2_conv = DoubleConv(64,128)
        self.Down2_pool = nn.MaxPool2d(2)
        self.Down2_conv_low = DoubleConv(16,32)
        self.Down2_pool_low = nn.MaxPool2d(2)
        # Size: 64->32
        self.Down3_conv = DoubleConv(128,256)
        self.Down3_pool = nn.MaxPool2d(2)
        self.Down3_conv_low = DoubleConv(32,64)
        self.Down3_pool_low = nn.MaxPool2d(2)
        # Size: 32->16
        self.Down4_conv = DoubleConv(256,512)
        self.Down4_pool = nn.MaxPool2d(2)
        self.Down4_conv_low = DoubleConv(64,128)
        self.Down4_pool_low = nn.MaxPool2d(2)
        # 随机嵌入信息卷积到图中
        self.after_concat_layer = ConvBNRelu(1024+self.config.water_features, 1024)
        self.Conv5 = nn.Sequential(
            DoubleConv(512+128, 1024),
            DoubleConv(1024, 1024),
            DoubleConv(1024, 1024),
            DoubleConv(1024, 1024),
        )
        # Size:16->32
        self.Up4_convT = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.Up4_conv = DoubleConv(1024, 512)

        # Size:32->64
        self.Up3_convT = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.Up3_conv = DoubleConv(512, 256)
        # Size:64->128
        self.Up2_convT = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.Up2_conv = DoubleConv(256, 128)
        # Size:128->256
        self.Up1_convT = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.Up1_conv = nn.Sequential(
            DoubleConv(128, 64),
            DoubleConv(64, 3),
        )
        # 最后一个卷积层得到输出
        #self.final_conv_with = nn.Conv2d(64+3, 3, 1)
        self.final_conv = nn.Conv2d(6, 3, 1)

    def forward(self, p):
        # p1 = self.init(p)
        # Size: 256->128
        down1_c = self.Down1_conv(p)
        down1_p = self.Down1_pool(down1_c)
        down1_c_self = self.Down1_conv_low(p)
        down1_p_self = self.Down1_pool_low(down1_c_self)
        # Size: 128->64
        down2_c = self.Down2_conv(down1_p)
        down2_p = self.Down2_pool(down2_c)
        down2_c_self = self.Down2_conv_low(down1_p_self)
        down2_p_self = self.Down2_pool_low(down2_c_self)
        # Size: 64->32
        down3_c = self.Down3_conv(down2_p)
        down3_p = self.Down3_pool(down3_c)
        down3_c_self = self.Down3_conv_low(down2_p_self)
        down3_p_self = self.Down3_pool_low(down3_c_self)
        # Size: 32->16
        down4_c = self.Down4_conv(down3_p)
        down4_p = self.Down4_pool(down4_c)
        down4_c_self = self.Down4_conv_low(down3_p_self)
        down4_p_self = self.Down4_pool_low(down4_c_self)
        # 自嵌入
        down4_cat = torch.cat((down4_p, down4_p_self), 1)
        mid = self.Conv5(down4_cat)

        # if self.is_embed_message:
        #     message = torch.ones(conv5.shape[0], self.config.water_features, conv5.shape[2], conv5.shape[3]).to(self.config.device)
        #     # 嵌入一定信息，与down4合并
        #     mid = torch.cat((conv5, message), 1)
        #     embedded = self.after_concat_layer(mid)
        # else:
        #     embedded = conv5

        # 开始反卷积，并叠加原始层
        # Size: 16->32
        up4_convT = self.Up4_convT(mid)
        merge4 = torch.cat([up4_convT, down4_c], dim=1)
        up4_conv = self.Up4_conv(merge4)
        # Size: 32->64
        up3_convT = self.Up3_convT(up4_conv)
        merge3 = torch.cat([up3_convT, down3_c], dim=1)
        up3_conv = self.Up3_conv(merge3)
        # Size: 64->128
        up2_convT = self.Up2_convT(up3_conv)
        merge2 = torch.cat([up2_convT, down2_c], dim=1)
        up2_conv = self.Up2_conv(merge2)
        # Size: 128->256
        up1_convT = self.Up1_convT(up2_conv)
        merge1 = torch.cat([up1_convT, down1_c], dim=1)
        up1_conv = self.Up1_conv(merge1)
        merge0 = torch.cat([up1_conv, p], dim=1)
        out = self.final_conv(merge0)
        return out