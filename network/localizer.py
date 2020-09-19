import torch
import torch.nn as nn
from  config import Encoder_Localizer_config
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
    def __init__(self, config=Encoder_Localizer_config()):
        super(LocalizeNetwork, self).__init__()
        self.config = config
        channels = int(self.config.Width*self.config.Height/self.config.block_size/self.config.block_size)
        self.conv_localize = nn.Sequential(
            ConvBNRelu(3, self.config.decoder_channels),
            #nn.MaxPool2d(2),
            ConvBNRelu(self.config.decoder_channels, self.config.decoder_channels),
            #nn.MaxPool2d(2),
            ConvBNRelu(self.config.decoder_channels, self.config.decoder_channels),
            #nn.MaxPool2d(2),
            ConvBNRelu(self.config.decoder_channels, self.config.decoder_channels),
            #nn.MaxPool2d(2),
        )
        # Size: 256->128
        self.Down1_conv = DoubleConv(3, 64)
        self.Down1_pool = nn.MaxPool2d(2)
        # Size: 128->64
        self.Down2_conv = DoubleConv(64, 128)
        self.Down2_pool = nn.MaxPool2d(2)
        # Size: 64->32
        self.Down3_conv = DoubleConv(128, 256)
        self.Down3_pool = nn.MaxPool2d(2)
        # Size: 32->16
        self.Down4_conv = DoubleConv(256, 512)
        self.Down4_pool = nn.MaxPool2d(2)

        if self.config.num_classes == 2:
            self.last_conv = nn.Sequential(
                nn.Conv2d(512,2,kernel_size=1,stride=1),
                #nn.BatchNorm2d(2),
                nn.Sigmoid()
            )
        else:
            self.last_conv = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=1, stride=1),
                #nn.BatchNorm2d(1),
                nn.Sigmoid()
            )


    def forward(self, r):
        # r1 = self.initialR3(r)
        # Size: 256->128
        down1_c = self.Down1_conv(r)
        down1_p = self.Down1_pool(down1_c)
        # Size: 128->64
        down2_c = self.Down2_conv(down1_p)
        down2_p = self.Down2_pool(down2_c)
        # Size: 64->32
        down3_c = self.Down3_conv(down2_p)
        down3_p = self.Down3_pool(down3_c)
        # Size: 32->16
        down4_c = self.Down4_conv(down3_p)
        down4_p = self.Down4_pool(down4_c)
        r1_conv = self.last_conv(down4_p)


        return r1_conv