# %matplotlib inline
import torch
import torch.nn as nn
from  config import Encoder_Localizer_config
from noise_layers.cropout import Cropout
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
from noise_layers.identity import Identity
import numpy as np
import util
from network.localizer import LocalizeNetwork
from network.encoder import EncoderNetwork
from torchvision import datasets, utils

device = torch.device("cuda")

def gaussian(tensor, mean=0, stddev=0.1):
    '''Adds random noise to a tensor.'''

    noise = torch.nn.init.normal_(torch.Tensor(tensor.size()).to(device), mean, stddev)

    return tensor + noise

# class DecoderNetwork(nn.Module):
#     def __init__(self, config=Encoder_Localizer_config()):
#         super(DecoderNetwork, self).__init__()
#         self.config = config
#         # self.upsample = nn.Sequential(
#         #     # Size:16->32
#         #     Up(self.config.encoder_features,self.config.encoder_features),
#         #     # Size:32->64
#         #     Up(self.config.encoder_features, self.config.encoder_features),
#         #     # Size:64->128
#         #     Up(self.config.encoder_features, self.config.encoder_features),
#         #     # Size:128->256
#         #     Up(self.config.encoder_features, self.config.encoder_features)
#         # )
#
#         self.conv_kernel_size_1 = nn.Sequential(
#             nn.Conv2d(self.config.encoder_features, 3, kernel_size=3, padding=1),
#             # nn.Conv2d(self.config.encoder_features, 3, kernel_size=1, padding=0))
#         )
#
#     def forward(self, h):
#         h1 = self.upsample(h)
#
#         out = self.conv_kernel_size_1(h1)
#         out_noise = gaussian(out.data, 0, 0.1)
#         return out, out_noise



# Join three networks in one module
class Encoder_Localizer(nn.Module):
    def __init__(self,config=Encoder_Localizer_config(),add_other_noise=False):
        super(Encoder_Localizer, self).__init__()
        self.config = config
        self.add_other_noise = add_other_noise
        self.encoder = EncoderNetwork(is_embed_message=True,config=config).to(device)

        # self.decoder = DecoderNetwork(config).to(device)
        self.cropout_noise_layer = Cropout(self.config.crop_size,config).to(device)

        self.jpeg_layer = JpegCompression(device)
        self.other_noise_layers = [Identity()]
        self.other_noise_layers.append(JpegCompression(device))
        self.other_noise_layers.append(Quantization(device))
        self.other_noise_layers_again = [Identity()]
        self.other_noise_layers_again.append(JpegCompression(device))
        self.other_noise_layers_again.append(Quantization(device))

        self.localize = LocalizeNetwork(config).to(device)

    def forward(self, secret, cover):
        # 得到Encode后的特征平面
        x_1 = self.encoder(secret)

        # Decode得到近似原图，这里的x_2_noise为添加高斯噪声后的结果
        # x_2, x_2_noise = self.decoder(x_1)

        # 添加Cropout噪声，cover是跟secret无关的图
        x_1_crop, cropout_label = self.cropout_noise_layer(x_1, cover)


        # 添加一般噪声：Gaussian JPEG 等（optional）
        # x_1_crop_attacked = self.jpeg_layer(x_1_crop)
        if self.add_other_noise:
            #layer_num = np.random.choice(2)
            random_noise_layer = np.random.choice(self.other_noise_layers,0)[0]
            x_1_crop_attacked = random_noise_layer(x_1_crop)
        else:
            # 固定加JPEG攻击（1），或者原图（0）
            layer_num = 1
            random_noise_layer = self.other_noise_layers[layer_num]
            x_1_crop_attacked = random_noise_layer(x_1_crop)

        # Test
        # imgs = [x_1_crop_attacked.data, secret.data]
        # imgs_tsor = torch.cat(imgs, 0)
        # util.imshow(utils.make_grid(imgs_tsor), 0, learning_rate=0.0001, beta=5,std=self.config.std,mean=self.config.mean)

        #如果不添加其他攻击，就是x_1_crop，否则是x_1_crop_attacked
        pred_label = self.localize(x_1_crop_attacked)

        # 开始第二个网络：根据部分信息恢复原始图像，这里不乘以之前的pred_label（防止网络太深）
        layer_num = 1
        random_noise_layer_again = self.other_noise_layers_again[layer_num]
        x_2_jpeg = random_noise_layer_again(x_1_crop)
        x_2_crop, _ = self.cropout_noise_layer(x_2_jpeg)
        # 经过类U-net得到恢复图像
        x_2_out = self.encoder(x_2_crop)

        return x_1, x_2_out, pred_label, cropout_label, self.jpeg_layer.__class__.__name__