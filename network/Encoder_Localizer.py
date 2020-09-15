# %matplotlib inline
import torch
import torch.nn as nn
from  config import Encoder_Localizer_config
from noise_layers.cropout import Cropout
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
from noise_layers.identity import Identity
import numpy as np
from network.conv_bn_relu import ConvBNRelu

device = torch.device("cuda")

def gaussian(tensor, mean=0, stddev=0.1):
    '''Adds random noise to a tensor.'''

    noise = torch.nn.init.normal_(torch.Tensor(tensor.size()).to(device), mean, stddev)

    return tensor + noise


# Preparation Network (2 conv layers)
class PrepNetwork(nn.Module):
    def __init__(self):
        super(PrepNetwork, self).__init__()
        self.initialP3 = nn.Sequential(
            ConvBNRelu(3, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64)
            )
        self.initialP4 = nn.Sequential(
            ConvBNRelu(3, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64)
        )
        # self.initialP5 = nn.Sequential(
        #     ConvBNRelu(3, 64),
        #     ConvBNRelu(64, 64),
        #     ConvBNRelu(64, 64),
        #     ConvBNRelu(64, 64),
        #     ConvBNRelu(64, 64)
        # )
        self.finalP3 = nn.Sequential(
            ConvBNRelu(128, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64),
        )
        self.finalP4 = nn.Sequential(
            ConvBNRelu(128, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64),
        )
        # self.finalP5 = nn.Sequential(
        #     nn.Conv2d(164, 64, kernel_size=5, padding=2),
        #     nn.ReLU())

    def forward(self, p):
        p1 = self.initialP3(p)
        p2 = self.initialP4(p)
        # p3 = self.initialP5(p)
        mid = torch.cat((p1, p2), 1)
        p4 = self.finalP3(mid)
        p5 = self.finalP4(mid)
        # p6 = self.finalP5(mid)
        out = torch.cat((p4, p5), 1)
        return out


# Hiding Network (5 conv layers)
class HidingNetwork(nn.Module):
    def __init__(self):
        super(HidingNetwork, self).__init__()
        self.initialH3 = nn.Sequential(
            ConvBNRelu(128, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64)
        )
        # self.initialH4 = nn.Sequential(
        #     ConvBNRelu(128, 64),
        #     ConvBNRelu(64, 64),
        #     ConvBNRelu(64, 64),
        #     ConvBNRelu(64, 64),
        #     ConvBNRelu(64, 64)
        #     )

        # self.initialH5 = nn.Sequential(
        #     ConvBNRelu(128, 64),
        #     ConvBNRelu(64, 64),
        #     ConvBNRelu(64, 64),
        #     ConvBNRelu(64, 64),
        #     ConvBNRelu(64, 64)
        #     )
        # self.finalH3 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1),
        #     nn.ReLU())
        # self.finalH4 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=4, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=4, padding=2),
        #     nn.ReLU())
        # self.finalH5 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=5, padding=2),
        #     nn.ReLU())
        self.finalH = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, padding=0))

    def forward(self, h):
        h1 = self.initialH3(h)
        # h2 = self.initialH4(h)
        # h3 = self.initialH5(h)
        # mid = torch.cat((h1, h2), 1)
        # h4 = self.finalH3(mid)
        # h5 = self.finalH4(mid)
        # h6 = self.finalH5(mid)
        # mid2 = torch.cat((h4, h5), 1)
        out = self.finalH(h1)
        out_noise = gaussian(out.data, 0, 0.1)
        return out, out_noise


# Reveal Network (2 conv layers)
class LocalizeNetwork(nn.Module):
    def __init__(self,config:Encoder_Localizer_config=Encoder_Localizer_config()):
        super(LocalizeNetwork, self).__init__()
        channels = int(config.Width*config.Height/config.block_size/config.block_size)
        self.initialR3 = nn.Sequential(
            ConvBNRelu(3, config.decoder_channels),
            ConvBNRelu(config.decoder_channels, config.decoder_channels),
            ConvBNRelu(config.decoder_channels, config.decoder_channels),
            ConvBNRelu(config.decoder_channels, config.decoder_channels),
            ConvBNRelu(config.decoder_channels, config.decoder_channels),
            ConvBNRelu(config.decoder_channels, config.decoder_channels),
            ConvBNRelu(config.decoder_channels, channels),
            # nn.Conv2d(3, config.decoder_channels, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(config.decoder_channels, config.decoder_channels, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(config.decoder_channels, config.decoder_channels, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(config.decoder_channels, config.decoder_channels, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(config.decoder_channels, channels, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.finalLayer = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, r):
        r1 = self.initialR3(r)
        r2 = self.average_pool(r1)
        r2 = r2.squeeze()
        out = self.finalLayer(r2)
        return out


# Join three networks in one module
class Encoder_Localizer(nn.Module):
    def __init__(self,crop_size=(0.5,0.5)):
        super(Encoder_Localizer, self).__init__()
        self.encoder = PrepNetwork().to(device)

        self.hiding = HidingNetwork().to(device)
        self.cropout_noise_layer = Cropout(crop_size,crop_size).to(device)

        self.other_noise_layers = [Identity()]
        self.other_noise_layers.append(JpegCompression(device))
        self.other_noise_layers.append(Quantization(device))

        self.localize = LocalizeNetwork().to(device)

    def forward(self, secret, cover):
        # 得到Encode后的特征平面
        x_1 = self.encoder(secret)
        # Decode得到近似原图，这里的x_2_noise为添加高斯噪声后的结果
        x_2, x_2_noise = self.hiding(x_1)
        # 添加Cropout噪声，cover是跟secret无关的图
        x_1_crop, cropout_label = self.cropout_noise_layer(x_2, cover)
        # 添加一般噪声：Gaussian JPEG 等（optional）
        random_noise_layer = np.random.choice(self.other_noise_layers, 1)[0]
        x_1_crop_attacked = random_noise_layer(x_1_crop)

        #如果不添加其他攻击，就是x_1_crop，否则是x_1_crop_attacked
        pred_label = self.localize(x_1_crop)
        return x_2, pred_label, cropout_label