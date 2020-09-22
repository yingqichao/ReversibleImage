# %matplotlib inline
import torch
import torch.nn as nn
from encoder.encoder_decoder import EncoderDecoder
from config import GlobalConfig
from localizer.localizer import LocalizeNetwork
from localizer.localizer_noPool import LocalizeNetwork_noPool
from noise_layers.cropout import Cropout
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.resize import Resize
from noise_layers.gaussian import Gaussian
from network.discriminator import Discriminator
from loss.vgg_loss import VGGLoss

class ReversibleImageNetwork:
    def __init__(self, username, config=GlobalConfig()):
        super(ReversibleImageNetwork, self).__init__()
        self.config = config
        self.device = self.config.device
        self.username = username
        # Generator and Recovery Network
        self.encoder_decoder = EncoderDecoder(username, config=config).to(self.device)
        # Localize Network
        if self.username=="qichao":
            self.localizer = LocalizeNetwork(config).to(self.device)
        else:
            self.localizer = LocalizeNetwork_noPool(config).to(self.device)
        # Discriminator
        self.discriminator = Discriminator(config).to(self.device)
        self.cover_label = 1
        self.encoded_label = 0
        # Vgg

        self.vgg_loss = VGGLoss(3, 1, False)
        self.vgg_loss.to(self.device)

        # Loss
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)

        # Optimizer
        self.optimizer_encoder_decoder = torch.optim.Adam(self.encoder_decoder.parameters())
        self.optimizer_localizer = torch.optim.Adam(self.localizer.parameters())
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        # Attack Layers
        self.cropout_layer = Cropout(config).to(self.device)
        self.jpeg_layer = JpegCompression(self.device).to(self.device)
        self.resize_layer = Resize((0.5, 0.7)).to(self.device)
        self.gaussian = Gaussian(config).to(self.device)




    def train_on_batch(self, Cover, Another):
        batch_size = Cover.shape[0]
        self.encoder_decoder.train()
        self.localizer.train()

        with torch.enable_grad():
            # ---------------- Train the discriminator and localizer -----------------------------
            self.optimizer_localizer.zero_grad()
            x_hidden, x_recover, mask, self.jpeg_layer.__class__.__name__ = self.encoder_decoder(Cover, Another)
            # Discriminate
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_on_cover = self.discriminator(Cover)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            d_loss_on_cover.backward()
            d_on_encoded = self.discriminator(x_hidden.detach())
            d_on_recovered = self.discriminator(x_recover.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)
            d_loss_on_recovered = self.bce_with_logits_loss(d_on_recovered, d_target_label_encoded)
            d_loss_on_fake_total = d_loss_on_encoded + d_loss_on_recovered
            d_loss_on_fake_total.backward()
            self.optimizer_discrim.step()

            x_1_crop, cropout_label, _ = self.cropout_layer(x_hidden, Cover)
            x_1_gaussian = self.gaussian(x_1_crop)
            x_1_resize = self.resize_layer(x_1_gaussian)
            x_1_attack = self.jpeg_layer(x_1_resize)
            pred_label = self.localizer(x_1_attack.detach())
            loss_localization = self.bce_with_logits_loss(pred_label, cropout_label)
            loss_localization.backward()
            self.optimizer_localizer.step()
            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_encoder_decoder.zero_grad()
            pred_again_label = self.localizer(x_1_attack)
            loss_localization_again = self.bce_with_logits_loss(pred_again_label, cropout_label)
            if self.config.useVgg == False:
                # loss_cover = self.mse_loss(x_hidden, Cover)
                vgg_on_cov = self.vgg_loss(Cover)
                vgg_on_enc = self.vgg_loss(x_hidden)
                loss_cover = self.mse_loss(vgg_on_cov, vgg_on_enc)
                loss_recover = self.mse_loss(x_recover.mul(mask)+x_recover.mul(1-mask)*0.1,
                                             Cover.mul(mask)+Cover.mul(1-mask)*0.1) \
                               / self.config.min_required_block_portion
            else:
                vgg_on_cov = self.vgg_loss(Cover)
                vgg_on_enc = self.vgg_loss(x_hidden)
                loss_cover = self.mse_loss(vgg_on_cov, vgg_on_enc)
                # loss_recover = self.mse_loss(x_recover.mul(mask),
                #                              Cover.mul(mask)) / self.config.min_required_block_portion

                vgg_on_recovery = self.vgg_loss(x_recover)
                loss_recover = self.mse_loss(vgg_on_cov, vgg_on_recovery)

                # vgg_on_recovery = self.vgg_loss(x_recover.mul(mask) + Cover.mul(1-mask))
                # loss_recover = self.mse_loss(vgg_on_cov, vgg_on_recovery)
            d_on_encoded_for_enc = self.discriminator(x_hidden)
            g_loss_adv_enc = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
            d_on_encoded_for_recovery = self.discriminator(x_recover)
            g_loss_adv_recovery = self.bce_with_logits_loss(d_on_encoded_for_recovery, g_target_label_encoded)
            # Total loss for EncoderDecoder
            loss_enc_dec =  g_loss_adv_enc * self.config.hyper_discriminator \
                           + g_loss_adv_recovery * self.config.hyper_discriminator \
                           + loss_localization_again * self.config.hyper_localizer\
                           + loss_cover * self.config.hyper_cover\
                            + loss_recover * self.config.hyper_recovery
            loss_enc_dec.backward()
            self.optimizer_encoder_decoder.step()

        losses = {
            'loss_sum': loss_enc_dec.item(),
            'loss_localization': loss_localization.item(),
            'loss_cover': loss_cover.item(),
            'loss_recover': loss_recover.item(),
            'loss_discriminator_enc': g_loss_adv_enc.item(),
            'loss_discriminator_recovery': g_loss_adv_recovery.item()
        }
        return losses, (x_hidden, x_recover.mul(mask)+Cover.mul(1-mask), pred_label, cropout_label)

    def validate_on_batch(self, Cover, Another):
        batch_size = Cover.shape[0]
        self.encoder_decoder.eval()
        self.localizer.eval()
        with torch.enable_grad():
            x_hidden, x_recover, mask, self.jpeg_layer.__class__.__name__ = self.encoder_decoder(Cover, Another)

            x_1_crop, cropout_label, _ = self.cropout_layer(x_hidden, Cover)
            x_1_gaussian = self.gaussian(x_1_crop)
            x_1_attack = self.jpeg_layer(x_1_gaussian)
            pred_label = self.localizer(x_1_attack.detach())
            loss_localization = self.bce_with_logits_loss(pred_label, cropout_label)

            loss_cover = self.mse_loss(x_hidden, Cover)
            loss_recover = self.mse_loss(x_recover.mul(mask), Cover.mul(mask)) / self.config.min_required_block_portion
            loss_enc_dec = loss_localization * self.hyper[0] + loss_cover * self.hyper[1] + loss_recover * \
                           self.hyper[2]

        losses = {
            'loss_sum': loss_enc_dec.item(),
            'loss_localization': loss_localization.item(),
            'loss_cover': loss_cover.item(),
            'loss_recover': loss_recover.item()
        }
        return losses, (x_hidden, x_recover.mul(mask) + Cover.mul(1 - mask), pred_label, cropout_label)

    def save_state_dict(self, path):
        torch.save(self.encoder_decoder.state_dict(), path + '_encoder_decoder.pkl')
        torch.save(self.localizer.state_dict(), path + '_localizer.pkl')

    def load_state_dict(self,path):
        self.localizer.load_state_dict(torch.load(path + '_localizer.pkl'))
        self.encoder_decoder.load_state_dict(torch.load(path + '_encoder_decoder.pkl'))

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
