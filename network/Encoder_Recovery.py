# %matplotlib inline
import torch
import torch.nn as nn
from  config import Encoder_Localizer_config
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
from noise_layers.identity import Identity
import numpy as np
from network.encoder import EncoderNetwork
import util as util


def gaussian(tensor, device, mean=0, stddev=0.1):
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


# class RecoveryNetwork(nn.Module):
#     def __init__(self, config=Encoder_Localizer_config()):
#         super(RecoveryNetwork, self).__init__()
#         self.config = config
#
#
#
#
#     def forward(self, r):
#         # 开始反卷积，并叠加原始层
#         # Size: 16->32
#         up4_convT = self.Up4_convT(embedded)
#         # merge4 = torch.cat([up4_convT, down4_c], dim=1)
#         up4_conv = self.Up4_conv(merge4)
#         # Size: 32->64
#         up3_convT = self.Up3_convT(up4_conv)
#         # merge3 = torch.cat([up3_convT, down3_c], dim=1)
#         up3_conv = self.Up3_conv(merge3)
#         # Size: 64->128
#         up2_convT = self.Up2_convT(up3_conv)
#         # merge2 = torch.cat([up2_convT, down2_c], dim=1)
#         up2_conv = self.Up2_conv(merge2)
#         # Size: 128->256
#         up1_convT = self.Up1_convT(up2_conv)
#         # merge1 = torch.cat([up1_convT, down1_c], dim=1)
#         up1_conv = self.Up1_conv(merge1)
#         out = self.final_conv(up1_conv)
#
#
#         return r1_conv


# Join three networks in one module
class Encoder_Recovery(nn.Module):
    def __init__(self,config=Encoder_Localizer_config(),crop_size=(0.5,0.5)):
        super(Encoder_Recovery, self).__init__()
        self.config = config
        device = config.device
        self.encoder = EncoderNetwork(is_embed_message=False, config=config).to(device)

        self.other_noise_layers = [Identity()]
        self.other_noise_layers.append(JpegCompression(device))
        self.other_noise_layers.append(Quantization(device))

        # self.recovery = RecoveryNetwork(config).to(device)

    def forward(self, secret, cover):
        # 得到Encode后的特征平面
        x_1 = self.encoder(secret)
        # Decode得到近似原图，这里的x_2_noise为添加高斯噪声后的结果
        # x_2, x_2_noise = self.decoder(x_1)
        # 添加Cropout噪声，cover是跟secret无关的图
        # x_1_crop, cropout_label = self.cropout_noise_layer(x_1, cover)
        # 添加一般噪声：Gaussian JPEG 等（optional）
        random_noise_layer = np.random.choice(self.other_noise_layers, 1)[0]
        x_1_attack = random_noise_layer(x_1)

        # 选择若干个块
        taken_blocks = set()
        x_input = torch.zeros((x_1_attack.shape[0], x_1_attack.shape[1],
                               self.config.block_size, self.config.block_size, self.config.min_required_block))
        while len(taken_blocks) < self.config.min_required_block:
            selected = int(util.random_float(0, self.num_blocks))
            if selected not in taken_blocks:
                taken_blocks.add(selected)
                selected_row_begin = selected / (self.Width / self.block_size)
                selected_col_begin = selected % (self.Width / self.block_size)
                x_input[:, :, :,:, selected] = x_1_attack[:, :,selected_row_begin:selected_row_begin + self.block_size,
                                             selected_col_begin:selected_col_begin + self.block_size]
        recovered_image = self.recovery(x_input)
        return x_1, recovered_image