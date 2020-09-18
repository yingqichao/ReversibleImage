import matplotlib.pyplot as plt
import numpy as np


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


def imshow(img, idx, learning_rate, beta, std, mean):
    '''Prints out an image given in tensor format.'''

    img = denormalize(img, std, mean)
    npimg = img.detach().cpu().numpy()
    if img.shape[0] == 3:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title('Example ' + str(idx) + ', lr=' + str(learning_rate) + ', B=' + str(beta))
    plt.show()
    return