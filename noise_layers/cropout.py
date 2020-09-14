import torch
import torch.nn as nn
from noise_layers.crop import get_random_rectangle_inside


class Cropout(nn.Module):
    """
    Combines the noised and cover images into a single image, as follows: Takes a crop of the noised image, and takes the rest from
    the cover image. The resulting image has the same size as the original and the noised images.
    """
    def __init__(self, height_ratio_range, width_ratio_range,device=torch.device("cuda")):
        super(Cropout, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range
        self.device = device

    def forward(self, noised_image,cover_image):
        # noised_image = noised_and_cover[0]
        # cover_image = noised_and_cover[1]

        assert noised_image.shape == cover_image.shape

        cropout_mask = torch.zeros_like(noised_image)
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=noised_image,
                                                                     height_ratio_range=self.height_ratio_range,
                                                                     width_ratio_range=self.width_ratio_range)
        # 被修改的区域内赋值1, dims: batch channel height width
        cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1

        noised_image = noised_image * cropout_mask + cover_image * (1-cropout_mask)

        block_height, block_width = int(noised_image.shape[2]/16), int(noised_image.shape[3]/16)
        # 生成label：被修改区域对应的8*8小块赋值为1, height/width

        cropout_label = torch.zeros((noised_image.shape[0],block_height*block_width), requires_grad=False)
        for row in range(int(h_start/16),int(h_end/16)):
            cropout_label[:, row*block_width+int(w_start/16):row*block_width+int(w_end/16)] = 1

        return noised_image, cropout_label.to(self.device)

