import torch
import torch.nn as nn
from  config import GlobalConfig
from noise_layers.cropout import Cropout
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
from noise_layers.identity import Identity
import numpy as np
from network.conv_bn_relu import ConvBNRelu
from network.down_sample import Down
from network.up_sample import Up
from network.double_conv import DoubleConv
from network.encoder import EncoderNetwork

class LocalizeNetwork(nn.Module):
    def __init__(self, config=GlobalConfig()):
        super(LocalizeNetwork, self).__init__()
        self.config = config
        # channels = int(self.config.Width*self.config.Height/self.config.block_size/self.config.block_size)
        # self.conv_localize = nn.Sequential(
        #     ConvBNRelu(3, self.config.decoder_channels),
        #     #nn.MaxPool2d(2),
        #     ConvBNRelu(self.config.decoder_channels, self.config.decoder_channels),
        #     #nn.MaxPool2d(2),
        #     ConvBNRelu(self.config.decoder_channels, self.config.decoder_channels),
        #     #nn.MaxPool2d(2),
        #     ConvBNRelu(self.config.decoder_channels, self.config.decoder_channels),
        #     #nn.MaxPool2d(2),
        # )

        # # Size: 256->128
        # self.Down1_conv = DoubleConv(3, 64)
        # self.Down1_pool = nn.MaxPool2d(2)
        # # Size: 128->64
        # self.Down2_conv = DoubleConv(64, 128)
        # self.Down2_pool = nn.MaxPool2d(2)
        # # Size: 64->32
        # self.Down3_conv = DoubleConv(128, 256)
        # self.Down3_pool = nn.MaxPool2d(2)
        # # Size: 32->16
        # self.Down4_conv = DoubleConv(256, 512)
        # self.Down4_pool = nn.MaxPool2d(2)

        # Level 1
        self.Level1_1 = nn.Sequential(
            DoubleConv(3, 40),
            DoubleConv(40, 40),
        )
        self.Level1_2 = nn.Sequential(
            DoubleConv(3, 40),
            DoubleConv(40, 40),
        )
        self.Level1_3 = nn.Sequential(
            DoubleConv(3, 40),
            DoubleConv(40, 40),
        )
        self.Down1_pool = nn.MaxPool2d(2)
        # Level 2
        self.Level2_1 = nn.Sequential(
            DoubleConv(120, 40),
            DoubleConv(40, 40),
        )
        self.Level2_2 = nn.Sequential(
            DoubleConv(120, 40),
            DoubleConv(40, 40),
        )
        self.Level2_3 = nn.Sequential(
            DoubleConv(120, 40),
            DoubleConv(40, 40),
        )
        self.Down2_pool = nn.MaxPool2d(2)
        # Level 3
        self.Level3_1 = nn.Sequential(
            DoubleConv(120, 40),
            DoubleConv(40, 40),
        )
        self.Level3_2 = nn.Sequential(
            DoubleConv(120, 40),
            DoubleConv(40, 40),
        )
        self.Level3_3 = nn.Sequential(
            DoubleConv(120, 40),
            DoubleConv(40, 40),
        )
        self.Down3_pool = nn.MaxPool2d(2)
        # Level 4
        self.Level4_1 = nn.Sequential(
            DoubleConv(120, 40),
            DoubleConv(40, 40),
        )
        self.Level4_2 = nn.Sequential(
            DoubleConv(120, 40),
            DoubleConv(40, 40),
        )
        self.Level4_3 = nn.Sequential(
            DoubleConv(120, 40),
            DoubleConv(40, 40),
        )
        self.Down4_pool = nn.MaxPool2d(2)

        if self.config.num_classes == 2:
            self.last_conv = nn.Sequential(
                nn.Conv2d(120,2,kernel_size=1,stride=1),
                nn.BatchNorm2d(2),
                nn.Sigmoid()
            )
        else:
            self.last_conv = nn.Sequential(
                nn.Conv2d(120, 1, kernel_size=1, stride=1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )


    def forward(self, r):
        # r1 = self.initialR3(r)
        # # Size: 256->128
        # down1_c = self.Down1_conv(r)
        # down1_p = self.Down1_pool(down1_c)
        # # Size: 128->64
        # down2_c = self.Down2_conv(down1_p)
        # down2_p = self.Down2_pool(down2_c)
        # # Size: 64->32
        # down3_c = self.Down3_conv(down2_p)
        # down3_p = self.Down3_pool(down3_c)
        # # Size: 32->16
        # down4_c = self.Down4_conv(down3_p)
        # down4_p = self.Down4_pool(down4_c)
        # r1_conv = self.last_conv(down4_p)

        # Level 1
        l1_1 = self.Level1_1(r)
        l1_2 = self.Level1_2(r)
        l1_3 = self.Level1_3(r)
        l1 = torch.cat([l1_1, l1_2, l1_3], dim=1)
        down1_p = self.Down1_pool(l1)
        # Level 2
        l2_1 = self.Level2_1(down1_p)
        l2_2 = self.Level2_2(down1_p)
        l2_3 = self.Level2_3(down1_p)
        l2 = torch.cat([l2_1, l2_2, l2_3], dim=1)
        down2_p = self.Down2_pool(l2)
        # Level 3
        l3_1 = self.Level3_1(down2_p)
        l3_2 = self.Level3_2(down2_p)
        l3_3 = self.Level3_3(down2_p)
        l3 = torch.cat([l3_1, l3_2, l3_3], dim=1)
        down3_p = self.Down3_pool(l3)
        # Level 4
        l4_1 = self.Level4_1(down3_p)
        l4_2 = self.Level4_2(down3_p)
        l4_3 = self.Level4_3(down3_p)
        l4 = torch.cat([l4_1, l4_2, l4_3], dim=1)
        down4_p = self.Down4_pool(l4)
        r1_conv = self.last_conv(down4_p)
        return r1_conv
