import numpy as np
import torch
import torch.nn as nn

from config import GlobalConfig
from network.encode_noPool_recovery import EncoderNetwork_noPoolRecovery
from network.encoder_noPool import EncoderNetwork_noPool
from network.localizer import LocalizeNetwork
from noise_layers.cropout import Cropout
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization


class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config=GlobalConfig()):
        super(EncoderDecoder, self).__init__()
        self.config = config
        self.device = self.config.device
        # Generator Network
        self.encoder = EncoderNetwork_noPool(config=config).to(self.device)
        # Noise Network
        self.jpeg_layer = JpegCompression(self.device)
        self.other_noise_layers = [Identity()]
        self.other_noise_layers.append(JpegCompression(self.device))
        self.other_noise_layers.append(Quantization(self.device))
        self.cropout_layer = Cropout(config).to(self.device)
        # Recovery Network
        self.recovery = EncoderNetwork_noPoolRecovery(config=config).to(self.device)

    def forward(self, Cover, Another):
        # 训练Generator
        x_1_out = self.encoder(Cover)
        # 经过JPEG压缩等攻击
        # if self.add_other_noise:
        #     # layer_num = np.random.choice(2)
        #     random_noise_layer_again = np.random.choice(self.other_noise_layers_again, 0)[0]
        #     x_2_attack = random_noise_layer_again(x_1_out)
        # else:
        #     # 固定加JPEG攻击（1），或者原图（0）
        #     random_noise_layer_again = self.jpeg_layer
        #     x_2_attack = random_noise_layer_again(x_1_out)
        x_2_attack = self.jpeg_layer(x_1_out)
        # 经过Cropout攻击
        x_2_crop, cropout_label_2, mask = self.cropout_layer(x_2_attack)

        # 训练RecoverNetwork：根据部分信息恢复原始图像，这里不乘以之前的pred_label（防止网络太深）
        x_2_out = self.recovery(x_2_crop)
        # Test
        # if is_test:
        #     imgs = [x_2_attack.data, x_2_crop.data, x_2_out.data]
        #     util.imshow(utils.make_grid(imgs), 'Fig.1 EncoderAttackedByJpeg Fig.2 Then Cropped Fig.3 Recovered', std=self.config.std,
        #                 mean=self.config.mean)

        return x_1_out, x_2_out, mask, self.jpeg_layer.__class__.__name__
