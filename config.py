import torch

class GlobalConfig():

    def __init__(self):
        self.Height = 224
        self.Width = 224
        self.block_size = 16
        self.decoder_channels = 128
        self.min_required_block = 64
        self.min_required_block_portion = 0.2
        self.crop_size = (0.2, 0.2)
        self.encoder_features = 64
        self.water_features = 256
        self.device = torch.device("cuda")
        self.num_classes = 1

        # self.beta = (0.1,1,1)
        self.hyper_localizer = 0.1
        self.useVgg = True
        if self.useVgg:
            self.hyper_cover = 1
        else:
            self.hyper_cover = 0.1
        self.hyper_discriminator = 0.001
        self.hyper_recovery = 1
        self.num_epochs = 50
        self.train_batch_size = 2
        self.test_batch_size = 1

        self.learning_rate = 0.0001

        self.use_dataset = 'COCO'  # "ImageNet"
        self.MODELS_PATH = './output/models/'
        self.VALID_PATH = './sample/valid_coco/'
        self.TRAIN_PATH = './sample/train_coco/'
        self.TEST_PATH = './sample/test_coco/'
        self.skipTraining = False
        # Discriminator
        self.discriminator_channels = 64
        self.discriminator_blocks = 3


        # Mean and std deviation of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
        if self.use_dataset == 'COCO':
            self.mean = [0.471, 0.448, 0.408]
            self.std = [0.234, 0.239, 0.242]
        else:
            self.std = [0.229, 0.224, 0.225]
            self.mean = [0.485, 0.456, 0.406]
