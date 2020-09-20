# %matplotlib inline
import numpy as np
import torch
import torch.nn as nn
from network.encoder_decoder import EncoderDecoder
from config import GlobalConfig
from network.encode_noPool_recovery import EncoderNetwork_noPoolRecovery
from network.encoder_noPool import EncoderNetwork_noPool
from network.localizer import LocalizeNetwork
from noise_layers.cropout import Cropout
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
import torch.nn.functional as F

# def gaussian(tensor, mean=0, stddev=0.1):
#     '''Adds random noise to a tensor.'''
#
#     noise = torch.nn.init.normal_(torch.Tensor(tensor.size()).to(device), mean, stddev)
#
#     return tensor + noise

# Join three networks in one module

class ReversibleImageNetwork:
    def __init__(self, config=GlobalConfig(), add_other_noise=False):
        super(ReversibleImageNetwork, self).__init__()
        self.config = config
        self.hyper = self.config.beta
        self.device = self.config.device
        self.add_other_noise = add_other_noise
        # Generator and Recovery Network
        self.encoder_decoder = EncoderDecoder(config=config).to(self.device)

        # Localize Network
        self.localizer = LocalizeNetwork(config).to(self.device)

        # Optimizer
        self.optimizer_encoder_decoder = torch.optim.Adam(self.encoder_decoder.parameters())
        self.optimizer_localizer = torch.optim.Adam(self.localizer.parameters())

        # Attack Layers
        self.cropout_layer = Cropout(config).to(self.device)
        self.jpeg_layer = JpegCompression(self.device)
        self.other_noise_layers = [Identity()]
        self.other_noise_layers.append(JpegCompression(self.device))
        self.other_noise_layers.append(Quantization(self.device))

        # Loss
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(self.device)


    def train_on_batch(self, Cover, Another):
        batch_size = Cover.shape[0]
        self.encoder_decoder.train()
        self.localizer.train()

        with torch.enable_grad():
            # ---------------- Train the localizer -----------------------------
            self.optimizer_localizer.zero_grad()
            x_hidden, x_recover, mask, self.jpeg_layer.__class__.__name__ = self.encoder_decoder(Cover, Another)

            x_1_crop, cropout_label, _ = self.cropout_layer(x_hidden, Cover)
            x_1_attack = self.jpeg_layer(x_1_crop)
            pred_label = self.localizer(x_1_attack.detach())
            loss_localization = F.binary_cross_entropy(pred_label, cropout_label)
            loss_localization.backward()
            self.optimizer_localizer.step()
            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_encoder_decoder.zero_grad()
            pred_again_label = self.localizer(x_1_attack)
            loss_localization_again = F.binary_cross_entropy(pred_again_label, cropout_label)
            loss_cover = F.mse_loss(x_hidden, Cover)
            loss_recover = F.mse_loss(x_recover.mul(mask), Cover.mul(mask))
            loss_enc_dec = loss_localization_again*self.hyper[0]+loss_cover*self.hyper[1]+loss_recover*self.hyper[2]
            loss_enc_dec.backward()
            self.optimizer_encoder_decoder.step()

        losses = {
            'loss_sum': loss_enc_dec.item(),
            'loss_localization': loss_localization.item(),
            'loss_cover': loss_cover.item(),
            'loss_recover': loss_recover.item()
        }
        return losses, (x_hidden, x_recover.mul(mask)+Cover.mul(1-mask), pred_label, cropout_label)


    # def forward(self, Cover, Another, skipLocalizationNetwork, skipRecoveryNetwork, is_test=False):
    #     # 得到Encode后的特征平面
    #     x_1_out = self.encoder(Cover)
    #
    #     # 训练第一个网络：Localize
    #     pred_label, cropout_label = None, None
    #     # 添加Cropout噪声，cover是跟secret无关的图
    #     if not self.skipLocalizationNetwork:
    #         x_1_crop, cropout_label, _ = self.cropout_layer_1(x_1_out,Another)
    #
    #
    #         # 添加一般噪声：Gaussian JPEG 等（optional）
    #         if self.add_other_noise:
    #             #layer_num = np.random.choice(2)
    #             random_noise_layer = np.random.choice(self.other_noise_layers,0)[0]
    #             x_1_attack = random_noise_layer(x_1_crop)
    #         else:
    #             # 固定加JPEG攻击（1），或者原图（0）
    #             random_noise_layer = self.other_noise_layers[1]
    #             x_1_attack = random_noise_layer(x_1_crop)
    #
    #         # Test
    #         # if is_test:
    #         #     imgs = [x_1_attack.data, Cover.data]
    #         #     util.imshow(imgs, '(After Net 1) Fig.1 After EncodeAndAttacked Fig.2 Original', std=self.config.std, mean=self.config.mean)
    #
    #         #如果不添加其他攻击，就是x_1_crop，否则是x_1_crop_attacked
    #         pred_label = self.localize(x_1_attack)
    #
    #     x_2_out, cropout_label_2, mask = None, None, None
    #     # 训练第二个网络：根据部分信息恢复原始图像，这里不乘以之前的pred_label（防止网络太深）
    #     if not self.skipRecoveryNetwork:
    #         if self.add_other_noise:
    #             #layer_num = np.random.choice(2)
    #             random_noise_layer_again = np.random.choice(self.other_noise_layers_again,0)[0]
    #             x_2_attack = random_noise_layer_again(x_1_out)
    #         else:
    #             # 固定加JPEG攻击（1），或者原图（0）
    #             random_noise_layer_again = self.other_noise_layers_again[1]
    #             x_2_attack = random_noise_layer_again(x_1_out)
    #
    #         x_2_crop, cropout_label_2, mask = self.cropout_layer_2(x_2_attack)
    #         # 经过类U-net得到恢复图像
    #         x_2_out = self.recovery(x_2_crop)
    #         # Test
    #         # if is_test:
    #         #     imgs = [x_2_attack.data, x_2_crop.data, x_2_out.data]
    #         #     util.imshow(utils.make_grid(imgs), 'Fig.1 EncoderAttackedByJpeg Fig.2 Then Cropped Fig.3 Recovered', std=self.config.std,
    #         #                 mean=self.config.mean)
    #
    #     return x_1_out, x_2_out, pred_label, cropout_label, mask, self.jpeg_layer.__class__.__name__
