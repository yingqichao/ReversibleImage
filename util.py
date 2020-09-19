import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, utils
import torch

def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min

def denormalize(image, std, mean):
    ''' Denormalizes a tensor of images.'''

    for t in range(image.shape[0]):
        image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
    return image


def imshow(input_img, text, std, mean):
    '''Prints out an image given in tensor format.'''
    imgs_tsor = torch.cat(input_img, 0)
    img = utils.make_grid(imgs_tsor)
    img = denormalize(img, std, mean)
    npimg = img.detach().cpu().numpy()
    if img.shape[0] == 3:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(text)
    plt.show()
    return