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
import util

# Directory path
# os.chdir("..")
if __name__ =='__main__':
    # Setting
    config = Encoder_Localizer_config()
    isSelfRecovery = True
    skipTraining = config.skipTraining

    device = config.device
    print(device)
    # Hyper Parameters
    num_epochs = config.num_epochs
    train_batch_size = config.train_batch_size
    test_batch_size = config.test_batch_size
    learning_rate = config.learning_rate
    use_Vgg = config.use_Vgg
    use_dataset = config.use_dataset
    beta = config.beta
    if use_Vgg:
        beta = 10

    MODELS_PATH = config.MODELS_PATH
    VALID_PATH = config.VALID_PATH
    TRAIN_PATH = config.TRAIN_PATH
    TEST_PATH = config.TEST_PATH

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    criterion = nn.BCEWithLogitsLoss()

    def customized_loss(S_prime, C_prime, S, C, B):
        ''' Calculates loss specified on the paper.'''

        loss_cover = F.mse_loss(C_prime, C)
        loss_secret = F.mse_loss(S_prime, S)
        loss_all = loss_cover + B * loss_secret
        return loss_all, loss_cover, loss_secret

    def localization_loss(pred_label, cropout_label, cropout_label_2, train_hidden, train_covers, train_recovered, use_vgg=False):

        hyper1, hyper2, hyper3 = config.beta[0],config.beta[1],config.beta[2]
        # numpy_watch_groundtruth = cropout_label.data.clone().detach().cpu().numpy()
        # numpy_watch_predicted = pred_label.data.clone().detach().cpu().numpy()
        if config.num_classes == 2:
            loss_localization = F.binary_cross_entropy(pred_label, cropout_label)
        else:
            loss_localization = criterion(pred_label, cropout_label)
        if use_vgg:
            vgg_loss = VGGLoss(3, 1, False).to(device)
            vgg_on_cov = vgg_loss(train_hidden)
            vgg_on_enc = vgg_loss(train_covers)
            loss_cover = F.mse_loss(vgg_on_cov, vgg_on_enc)
            if cropout_label_2 is not None:
                vgg_loss_2 = VGGLoss(3, 1, False).to(device)
                vgg_on_cov_2 = vgg_loss(train_recovered)
                vgg_on_enc_2 = vgg_loss(train_covers)
                loss_recover = F.mse_loss(vgg_on_cov_2, vgg_on_enc_2)
        else:
            # loss_fn = nn.MSELoss()
            loss_cover = F.mse_loss(train_hidden*255, train_covers*255)
            if cropout_label_2 is not None:
                # imgs = [(train_recovered).mul(cropout_label_2[1]).data,(train_covers).mul(cropout_label_2[1])]
                # util.imshow(imgs,'(After Net 1) Fig.1 After EncodeAndAttacked Fig.2 Original',
                #             std=config.std, mean=config.mean)

                loss_recover = F.mse_loss((train_recovered*255).mul(cropout_label_2[1]), (train_covers*255).mul(cropout_label_2[1]))

        if loss_localization < 0.15:
            hyper1 = 0
        if cropout_label_2 is not None:
            loss_all = hyper1 * loss_localization + hyper2 * loss_cover + hyper3 * loss_recover
        else:
            loss_all = beta[0] * loss_localization + loss_cover
        return loss_all, loss_localization, loss_cover, loss_recover



    def train_model(net, train_loader, beta, learning_rate,isSelfRecovery=True):
        # batch:3 epoch:2 data:2*3*224*224

        # Save optimizer
        # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

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
                    train_covers = data[:]
                    train_secrets = data[:]
                    # train_covers = data[:]
                    # train_secrets = data[:]

                # Creates variable from secret and cover images
                # train_cover作为tamper的图像
                train_secrets = train_secrets.to(device)
                train_covers = train_covers.to(device)
                # train_secrets = torch.tensor(train_secrets, requires_grad=False).to(device)
                # train_covers = torch.tensor(train_covers, requires_grad=False).to(device)

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                train_hidden, train_recovered, pred_label, cropout_label, cropout_label_2, _ = net(train_secrets, train_covers)

                # MSE标签距离 loss
                train_loss_all, train_loss_localization, train_loss_cover, train_loss_recover = \
                    localization_loss(pred_label, cropout_label, cropout_label_2, train_hidden, train_covers, train_recovered)

                # Calculate loss and perform backprop
                # train_loss, train_loss_cover, train_loss_secret = customized_loss(train_output, train_hidden, train_secrets,
                #                                                                   train_covers, beta)
                train_loss_all.backward()
                optimizer.step()

                # Saves training loss
                train_losses.append(train_loss_all.data.cpu().numpy())
                loss_history.append(train_loss_all.data.cpu().numpy())

                if idx % 8==7:
                # Prints mini-batch losses
                    print('Net 1 Epoch {0}/{1} Training: Batch {2}/{3}. Total Loss {4:.4f}, Localization Loss {5:.4f}, Cover Loss {6:.4f}, Recover Loss {7:.4f} '
                        .format(epoch, num_epochs, idx + 1, len(train_loader), train_loss_all.data, train_loss_localization.data, train_loss_cover.data,train_loss_recover.data))

            torch.save(net.state_dict(), MODELS_PATH + 'Epoch N{}.pkl'.format(epoch + 1))

            mean_train_loss = np.mean(train_losses)

            # Prints epoch average loss
            print('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(
                epoch + 1, num_epochs, mean_train_loss))

            # Debug
            # imshow(utils.make_grid(train_covers), 0, learning_rate=learning_rate, beta=beta)
            # imshow(utils.make_grid(train_hidden), 0, learning_rate=learning_rate, beta=beta)
        return net, mean_train_loss, loss_history


    def test_model(net, test_loader, beta, learning_rate, isSelfRecovery=True):
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
                # test_secret = data[:]
                # test_cover = data[:]

            # Creates variable from secret and cover images
            test_cover = test_cover.to(device)
            test_secret = torch.tensor(test_secret, requires_grad=False).to(device)
            test_cover = torch.tensor(test_cover, requires_grad=False).to(device)

            test_hidden, test_recovered, pred_label, cropout_label, cropout_label_2, selected_attack = \
                net(test_secret, test_cover, is_test=False)
            # MSE标签距离 loss
            test_loss_all, test_loss_localization, test_loss_cover, test_loss_recover = \
                localization_loss(pred_label, cropout_label, cropout_label_2, test_hidden, test_cover, test_recovered)

            #     diff_S, diff_C = np.abs(np.array(test_output.data[0]) - np.array(test_secret.data[0])), np.abs(np.array(test_hidden.data[0]) - np.array(test_cover.data[0]))

            #     print (diff_S, diff_C)

            if idx < 10:
                print('Test: Batch {0}/{1}. Total Loss {2:.4f}, Localization Loss {3:.4f}, Cover Loss {4:.4f}, Recover Loss {5:.4f} '
                    .format(idx + 1, len(train_loader), test_loss_all.data, test_loss_localization.data, test_loss_cover.data,test_loss_recover.data))
                print('Selected: ' + selected_attack)
                # Creates img tensor
                imgs = [test_secret.data, test_hidden.data, test_recovered.data]

                # prints the whole tensor
                torch.set_printoptions(profile="full")
                print('----Figure {0}----'.format(idx + 1))
                print('[Expected]')
                print(pred_label.data)

                print('[Real]')
                print(cropout_label.data)
                print('------------------')
                # Prints Images
                util.imshow(imgs, 'Example ' + str(idx) + ', lr=' + str(learning_rate) + ', B=' + str(beta),
                            std=config.std, mean=config.mean)

            test_losses.append(test_loss_all.data.cpu().numpy())

        mean_test_loss = np.mean(test_losses)

        print('Average loss on test set: {:.2f}'.format(mean_test_loss))

    # ------------------------------------ Begin ---------------------------------------
    # Creates net object
    net = Encoder_Localizer(config, train_second_network=True).to(device)

    # Creates training set
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            TRAIN_PATH,
            transforms.Compose([
                transforms.Scale(config.Width),
                transforms.RandomCrop(config.Width),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.mean,
                                     std=config.std),

            ])), batch_size=train_batch_size, num_workers=1,
        pin_memory=True, shuffle=True, drop_last=True)

    # Creates test set
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            TEST_PATH,
            transforms.Compose([
                transforms.Scale(config.Width),
                transforms.RandomCrop(config.Width),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.mean,
                                     std=config.std)
            ])), batch_size=test_batch_size, num_workers=1,
        pin_memory=True, shuffle=True, drop_last=True)
    if not skipTraining:
        net, mean_train_loss, loss_history = train_model(net, train_loader, beta, learning_rate, isSelfRecovery)
        # Plot loss through epochs
        plt.plot(loss_history)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Batch')
        plt.show()
    else:
        net.load_state_dict(torch.load(MODELS_PATH+'Epoch N50.pkl'))

    test_model(net, test_loader, beta, learning_rate, isSelfRecovery=True)
