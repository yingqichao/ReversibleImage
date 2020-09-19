# toooest %matplotlib inline
import os
from loss.vgg_loss import VGGLoss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import utils
from torchvision import datasets, utils
from network.Encoder_Localizer import Encoder_Localizer
from config import Encoder_Localizer_config
import torch.nn as nn
import torch.nn.functional as F
from noise_layers.jpeg_compression import JpegCompression

# Directory path
# os.chdir("..")
if __name__ =='__main__':
    cwd = '.'
    device = torch.device("cuda")
    print(device)
    # Hyper Parameters
    num_epochs = 10
    batch_size = 4
    learning_rate = 0.0001
    use_Vgg = False
    use_dataset = 'COCO' # "ImageNet"
    beta = 5000
    if use_Vgg:
        beta = 5
    # Mean and std deviation of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
    if use_dataset == 'COCO':
        mean = [0.471, 0.448, 0.408]
        std = [0.234, 0.239, 0.242]
    else:
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]

    # TODO: Define train, validation and models
    MODELS_PATH = './output/models/'
    VALID_PATH = './sample/valid_coco/'
    TRAIN_PATH = './sample/train_coco/'
    TEST_PATH = './sample/test_coco/'

    if not os.path.exists(MODELS_PATH): os.mkdir(MODELS_PATH)

    criterion = nn.BCEWithLogitsLoss()

    def customized_loss(S_prime, C_prime, S, C, B):
        ''' Calculates loss specified on the paper.'''

        loss_cover = F.mse_loss(C_prime, C)
        loss_secret = F.mse_loss(S_prime, S)
        loss_all = loss_cover + B * loss_secret
        return loss_all, loss_cover, loss_secret

    def localization_loss(pred_label, cropout_label, train_hidden, train_covers, beta=1,use_vgg=False):
        ''' 自定义localization_loss '''
        numpy_watch_groundtruth = cropout_label.data.clone().detach().cpu().numpy()
        numpy_watch_predicted = pred_label.data.clone().detach().cpu().numpy()
        if config.num_classes==2:
            loss_localization = F.binary_cross_entropy(pred_label, cropout_label)
        else:
            loss_localization = criterion(pred_label, cropout_label)
        if use_vgg:
            vgg_loss = VGGLoss(3, 1, False).to(device)
            vgg_on_cov = vgg_loss(train_hidden)
            vgg_on_enc = vgg_loss(train_covers)
            loss_cover = F.mse_loss(vgg_on_cov, vgg_on_enc)
        else:
            # loss_fn = nn.MSELoss()
            loss_cover = F.mse_loss(train_hidden*255, train_covers*255)
        loss_all = beta * loss_localization + loss_cover
        return loss_all, loss_localization, loss_cover


    def denormalize(image, std, mean):
        ''' Denormalizes a tensor of images.'''

        for t in range(image.shape[0]):
            image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
        return image


    def imshow(img, idx, learning_rate, beta):
        '''Prints out an image given in tensor format.'''

        img = denormalize(img, std, mean)
        npimg = img.detach().cpu().numpy()
        if img.shape[0] == 3:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title('Example ' + str(idx) + ', lr=' + str(learning_rate) + ', B=' + str(beta))
        plt.show()
        return


    def train_model(net, train_loader, beta, learning_rate,isSelfRecovery=True):
        # batch:3 epoch:2 data:2*3*224*224

        # Save optimizer
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        loss_history = []
        # Iterate over batches performing forward and backward passes
        for epoch in range(num_epochs):

            # Train mode
            net.train()

            train_losses = []
            # Train one epoch
            for idx, train_batch in enumerate(train_loader):
                data, _ = train_batch

                # Saves secret images and secret covers
                if not isSelfRecovery:
                    train_covers = data[:len(data) // 2]
                    train_secrets = data[len(data) // 2:]
                else:
                    # self recovery
                    # train_covers = data[:]
                    # train_secrets = data[:]
                    train_covers = data[:len(data) // 2]
                    train_secrets = data[len(data) // 2:]

                # Creates variable from secret and cover images
                # train_cover作为tamper的图像
                train_secrets = torch.tensor(train_secrets, requires_grad=False).to(device)
                train_covers = torch.tensor(train_covers, requires_grad=False).to(device)

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                train_hidden, pred_label, cropout_label, _ = net(train_secrets, train_covers)

                # MSE标签距离 loss
                train_loss_all, train_loss_localization, train_loss_cover = \
                    localization_loss(pred_label, cropout_label, train_hidden, train_covers,beta=beta)

                # Calculate loss and perform backprop
                # train_loss, train_loss_cover, train_loss_secret = customized_loss(train_output, train_hidden, train_secrets,
                #                                                                   train_covers, beta)
                train_loss_all.backward()
                optimizer.step()

                # Saves training loss
                train_losses.append(train_loss_all.data.cpu().numpy())
                loss_history.append(train_loss_all.data.cpu().numpy())

                # Prints mini-batch losses
                print('Net 1 Training: Batch {0}/{1}. Total Loss {2:.4f}, Localization Loss {3:.4f}, Cover Loss {4:.4f} '.format(
                    idx + 1, len(train_loader), train_loss_all.data, train_loss_localization.data, train_loss_cover.data))

            torch.save(net.state_dict(), MODELS_PATH + 'Epoch N{}.pkl'.format(epoch + 1))

            mean_train_loss = np.mean(train_losses)

            # Prints epoch average loss
            print('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(
                epoch + 1, num_epochs, mean_train_loss))

            # Debug
            # imshow(utils.make_grid(train_covers), 0, learning_rate=learning_rate, beta=beta)
            # imshow(utils.make_grid(train_hidden), 0, learning_rate=learning_rate, beta=beta)
        return net, mean_train_loss, loss_history


    # Setting
    config = Encoder_Localizer_config()
    isSelfRecovery = True
    skipTraining = True
    # Creates net object
    net = Encoder_Localizer(config).to(device)

    # Creates training set
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            TRAIN_PATH,
            transforms.Compose([
                transforms.Scale(512),
                transforms.RandomCrop(512),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                     std=std)
            ])), batch_size=batch_size, num_workers=1,
        pin_memory=True, shuffle=True, drop_last=True)

    # Creates test set
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            TEST_PATH,
            transforms.Compose([
                transforms.Scale(512),
                transforms.RandomCrop(512),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                     std=std)
            ])), batch_size=1, num_workers=1,
        pin_memory=True, shuffle=True, drop_last=True)
    if not skipTraining:
        net, mean_train_loss, loss_history = train_model(net, train_loader, beta, learning_rate, isSelfRecovery)
        # Plot loss through epochs
        plt.plot(loss_history)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Batch')
        plt.show()
    # else:
    #     net.load_state_dict(torch.load(MODELS_PATH+'Epoch N10.pkl'))

    # Switch to evaluate mode
    net.eval()

    test_losses = []
    # Show images
    for idx, test_batch in enumerate(test_loader):
        # Saves images
        data, _ = test_batch

        # Saves secret images and secret covers
        if not isSelfRecovery:
            test_secret = data[:len(data) // 2]
            test_cover = data[len(data) // 2:]
        else:
            # Self Recovery
            test_secret = data[:]
            test_cover = data[:]


        # Creates variable from secret and cover images
        test_secret = torch.tensor(test_secret, requires_grad=False).to(device)
        test_cover = torch.tensor(test_cover, requires_grad=False).to(device)

        jpeg_layer = JpegCompression(device)
        test_hidden = jpeg_layer(test_secret)
        # test_hidden, pred_label, cropout_label, selected_attack = net(test_secret, test_cover)
        # MSE标签距离 loss
        # test_loss_all, test_loss_localization, test_loss_cover = \
        #     localization_loss(pred_label, cropout_label, test_hidden, test_cover, beta=1)

        #     diff_S, diff_C = np.abs(np.array(test_output.data[0]) - np.array(test_secret.data[0])), np.abs(np.array(test_hidden.data[0]) - np.array(test_cover.data[0]))

        #     print (diff_S, diff_C)

        if idx < 10:
            # print('Test: Batch {0}/{1}. Total Loss {2:.4f}, Localization Loss {3:.4f}, Cover Loss {4:.4f} '.format(
            #     idx + 1, len(train_loader), test_loss_all.data, test_loss_localization.data, test_loss_cover.data))
            # print('Selected: '+ selected_attack)
            # Creates img tensor
            # imgs = [test_secret.data,  test_cover.data, test_hidden.data, test_output.data] # 隐藏图像  宿主图像 输出图像 提取得到的图像
            imgs = [test_cover.data, test_hidden.data]
            mse_loss = F.mse_loss(test_cover*255,test_hidden*255)
            print("MSE loss: {0:.4f}".format(mse_loss.data))

            # prints the whole tensor
            torch.set_printoptions(profile="full")
            print('----Figure {0}----'.format(idx + 1))
            # print('[Expected]')
            # print(pred_label.data)
            #
            # print('[Real]')
            # print(cropout_label.data)
            print('------------------')
            # Prints Images
            imshow(imgs, idx + 1, learning_rate=learning_rate, beta=beta)
            # target_tensor = torch.tensor((pred_label.reshape(1,14,14).detach().cpu().numpy()*255).astype(np.uint8)).to(device)
            # imshow(target_tensor, idx+1, learning_rate=learning_rate, beta=beta)


        # test_losses.append(test_loss_all.data.cpu().numpy())

    mean_test_loss = np.mean(test_losses)

    print('Average loss on test set: {:.2f}'.format(mean_test_loss))
