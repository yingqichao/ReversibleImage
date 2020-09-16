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
from network.down_sample import Down
from network.up_sample import Up

device = torch.device("cuda")

def gaussian(tensor, mean=0, stddev=0.1):
    '''Adds random noise to a tensor.'''

    noise = torch.nn.init.normal_(torch.Tensor(tensor.size()).to(device), mean, stddev)

    return tensor + noise

# Preparation Network (2 conv layers)
class EncoderNetwork(nn.Module):
    def __init__(self, config=Encoder_Localizer_config()):
        super(EncoderNetwork, self).__init__()
        self.config = config

        self.downsample = nn.Sequential(
            # Size: 256->128
            Down(3,self.config.encoder_features),
            # Size: 128->64
            Down(self.config.encoder_features, self.config.encoder_features),
            # Size: 64->32
            Down(self.config.encoder_features, self.config.encoder_features),
            # Size: 32->16
            Down(self.config.encoder_features, self.config.encoder_features),

        )

        self.furtherConv = nn.Sequential(
            ConvBNRelu(self.config.encoder_features+self.config.water_features, self.config.encoder_features),
            ConvBNRelu(self.config.encoder_features, self.config.encoder_features),
            ConvBNRelu(self.config.encoder_features, self.config.encoder_features),
            ConvBNRelu(self.config.encoder_features, self.config.encoder_features),
            ConvBNRelu(self.config.encoder_features, self.config.encoder_features),
        )


    def forward(self, p):
        p1 = self.downsample(p)
        message = torch.ones(p1.shape[0], 8, p1.shape[2],p1.shape[3]).to(device)

        mid = torch.cat((p1, message), 1)
        out = self.furtherConv(mid)

        return out


# Hiding Network (5 conv layers)
class DecoderNetwork(nn.Module):
    def __init__(self, config=Encoder_Localizer_config()):
        super(DecoderNetwork, self).__init__()
        self.config = config
        self.upsample = nn.Sequential(
            # Size:16->32
            Up(self.config.encoder_features,self.config.encoder_features),
            # Size:32->64
            Up(self.config.encoder_features, self.config.encoder_features),
            # Size:64->128
            Up(self.config.encoder_features, self.config.encoder_features),
            # Size:128->256
            Up(self.config.encoder_features, self.config.encoder_features)
        )

        self.conv_kernel_size_1 = nn.Sequential(
            nn.Conv2d(self.config.encoder_features, 3, kernel_size=1, padding=0))

    def forward(self, h):
        h1 = self.upsample(h)

        out = self.conv_kernel_size_1(h1)
        out_noise = gaussian(out.data, 0, 0.1)
        return out, out_noise


# Reveal Network (2 conv layers)
class LocalizeNetwork(nn.Module):
    def __init__(self, config=Encoder_Localizer_config()):
        super(LocalizeNetwork, self).__init__()
        self.config = config
        channels = int(self.config.Width*self.config.Height/self.config.block_size/self.config.block_size)
        self.initialR3 = nn.Sequential(
            ConvBNRelu(3, self.config.decoder_channels),
            nn.AdaptiveAvgPool2d(output_size=(int(self.config.Width / 2), int(self.config.Height / 2))),
            ConvBNRelu(self.config.decoder_channels, self.config.decoder_channels),
            nn.AdaptiveAvgPool2d(output_size=(int(self.config.Width / 4), int(self.config.Height / 4))),
            ConvBNRelu(self.config.decoder_channels, self.config.decoder_channels),
            nn.AdaptiveAvgPool2d(output_size=(int(self.config.Width / 8), int(self.config.Height / 8))),
            ConvBNRelu(self.config.decoder_channels, self.config.decoder_channels),
            nn.AdaptiveAvgPool2d(output_size=(int(self.config.Width / 16), int(self.config.Height / 16))),

        )
        self.last_conv = nn.Sequential(
            nn.Conv2d(self.config.decoder_channels,1,kernel_size=1,stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


    def forward(self, r):
        r1 = self.initialR3(r)
        r1_conv = self.last_conv(r1)


        return r1_conv


# Join three networks in one module
class Encoder_Localizer(nn.Module):
    def __init__(self,config=Encoder_Localizer_config(),crop_size=(0.5,0.5)):
        super(Encoder_Localizer, self).__init__()
        self.config = config
        self.encoder = EncoderNetwork(config).to(device)

        self.decoder = DecoderNetwork(config).to(device)
        self.cropout_noise_layer = Cropout(self.config.crop_size,config).to(device)

        self.other_noise_layers = [Identity()]
        self.other_noise_layers.append(JpegCompression(device))
        self.other_noise_layers.append(Quantization(device))

        self.localize = LocalizeNetwork(config).to(device)

    def forward(self, secret, cover):
        # 得到Encode后的特征平面
        x_1 = self.encoder(secret)
        # Decode得到近似原图，这里的x_2_noise为添加高斯噪声后的结果
        x_2, x_2_noise = self.decoder(x_1)
        # 添加Cropout噪声，cover是跟secret无关的图
        x_1_crop, cropout_label = self.cropout_noise_layer(x_2, cover)
        # 添加一般噪声：Gaussian JPEG 等（optional）
        random_noise_layer = np.random.choice(self.other_noise_layers, 1)[0]
        x_1_crop_attacked = random_noise_layer(x_1_crop)

        #如果不添加其他攻击，就是x_1_crop，否则是x_1_crop_attacked
        pred_label = self.localize(x_1_crop)
        return x_2, pred_label, cropout_label