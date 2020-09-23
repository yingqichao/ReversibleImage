import torch.nn as nn
import torch
from config import GlobalConfig
from decoder.decoder_pool import Decoder_pool
from noise_layers.cropout import Cropout
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.resize import Resize
from noise_layers.gaussian import Gaussian
from encoder.encoder_pool_shuffle import EncoderNetwork_pool_shuffle
from encoder.encoder_noPool_shuffle import EncoderNetwork_noPool_shuffle
from decoder.decoder_noPool import Decoder_noPool
from encoder.encoder_rotate import Encoder_rotate
from decoder.decoder_rotate import Decoder_rotate

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, username, config=GlobalConfig()):
        super(EncoderDecoder, self).__init__()
        self.config = config
        self.device = self.config.device
        self.username = username
        if self.username == "qichao":
            # Generator Network
            self.encoder = EncoderNetwork_pool_shuffle(config=config).to(self.device)
            # Recovery Network
            self.recovery = Decoder_pool(config=config).to(self.device)
        else:
            # Generator Network
            self.encoder = Encoder_rotate(config=config).to(self.device)
            # Recovery Network
            self.recovery1 = Decoder_rotate(config=config).to(self.device)
            self.recovery2 = Decoder_rotate(config=config).to(self.device)
            # self.recovery3 = Decoder_rotate(config=config).to(self.device)
        # Noise Network
        self.jpeg_layer = JpegCompression(self.device)

        # self.cropout_layer = Cropout(config).to(self.device)
        self.gaussian = Gaussian(config).to(self.device)
        self.resize_layer = Resize((0.5, 0.7)).to(self.device)

    def forward(self, Cover, flip):

        # 训练Generator

        x_hidden = self.encoder(Cover, flip)
        # 经过JPEG压缩等攻击
        # if self.add_other_noise:
        #     # layer_num = np.random.choice(2)
        #     random_noise_layer_again = np.random.choice(self.other_noise_layers_again, 0)[0]
        #     x_2_attack = random_noise_layer_again(x_1_out)
        # else:
        #     # 固定加JPEG攻击（1），或者原图（0）
        #     random_noise_layer_again = self.jpeg_layer
        #     x_2_attack = random_noise_layer_again(x_1_out)

        x_1_gaussian = self.gaussian(x_hidden)
        # x_1_resize = self.resize_layer(x_1_gaussian)
        x_2_attack = self.jpeg_layer(x_1_gaussian)
        # 经过Cropout攻击
        #x_2_crop, cropout_label_2, mask = self.cropout_layer(x_2_attack)

        # 训练RecoverNetwork：根据部分信息恢复原始图像，这里不乘以之前的pred_label（防止网络太深）
        out1 = self.recovery1(x_2_attack)
        out2 = self.recovery2(x_2_attack)
        # out3 = self.recovery3(x_2_attack)
        # out4 = self.recovery(x_2_attack)
        # Test
        # if is_test:
        #     imgs = [x_2_attack.data, x_2_crop.data, x_2_out.data]
        #     util.imshow(utils.make_grid(imgs), 'Fig.1 EncoderAttackedByJpeg Fig.2 Then Cropped Fig.3 Recovered', std=self.config.std,
        #                 mean=self.config.mean)

        return x_hidden, (out1, out2)
