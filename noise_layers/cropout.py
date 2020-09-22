import torch
import torch.nn as nn
from noise_layers.crop import get_random_rectangle_inside
import matplotlib.pyplot as plt
import numpy as np
from config import GlobalConfig
import math

class Cropout(nn.Module):
    """
    Combines the noised and cover images into a single image, as follows: Takes a crop of the noised image, and takes the rest from
    the cover image. The resulting image has the same size as the original and the noised images.
    """
    def __init__(self, config=GlobalConfig()):
        super(Cropout, self).__init__()
        self.config = config
        self.device = config.device

    def forward(self, embedded_image,cover_image=None):

        if cover_image is not None:
            assert embedded_image.shape == cover_image.shape
        sum_attacked = 0
        cropout_mask = torch.zeros_like(embedded_image)
        block_height, block_width = int(embedded_image.shape[2] / 16), int(embedded_image.shape[3] / 16)
        # if self.config.num_classes==2:
        #     cropout_label = torch.zeros((embedded_image.shape[0], 2, block_height, block_width), requires_grad=False)
        #     cropout_label[:, 1, :, :] = 1
        # else:
        #     cropout_label = torch.zeros((embedded_image.shape[0], 1, block_height, block_width), requires_grad=False)

        # 不断修改小块，直到修改面积至少为全图的50%
        while sum_attacked<self.config.min_required_block_portion:
            h_start, h_end, w_start, w_end, ratio = get_random_rectangle_inside(
                image=embedded_image, height_ratio_range=(0.1, self.config.crop_size[0]), width_ratio_range=(0.1, self.config.crop_size[1]))
            sum_attacked += ratio
            # 被修改的区域内赋值1, dims: batch channel height width
            cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1
            # if self.config.num_classes == 2:
            #     cropout_label[:, 0, math.floor(h_start / 16):math.ceil(h_end / 16), math.floor(w_start / 16):math.ceil(w_end / 16)] = 1
            #     cropout_label[:, 1, math.floor(h_start / 16):math.ceil(h_end / 16), math.floor(w_start / 16):math.ceil(w_end / 16)] = 0
            # else:
            #     cropout_label[:, 0, math.floor(h_start / 16):math.ceil(h_end / 16), math.floor(w_start / 16):math.ceil(w_end / 16)] = 1



        # 生成label：被修改区域对应的8*8小块赋值为1, height/width

        if cover_image is not None:
            tampered_image = embedded_image * (1-cropout_mask) + cover_image * cropout_mask
        else:
            tampered_image = embedded_image * (1-cropout_mask)

        cropout_label = cropout_mask[:,0,:,:]
        # cropout_label = cropout_label.unsqueeze(1)
        # numpy_conducted = cropout_mask.clone().detach().cpu().numpy()
        # numpy_groundtruth = cropout_label.data.clone().detach().cpu().numpy()

        return tampered_image, cropout_label, cropout_mask # cropout_label.to(self.device)

