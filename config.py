import torch

class Encoder_Localizer_config():

    def __init__(self, Height: int=256, Width: int=256, block_size: int=16):
        self.Height = Height
        self.Width = Width
        self.block_size = block_size
        self.decoder_channels = 128
        self.min_required_block = 64
        self.min_required_block_portion = 0.4
        self.crop_size = (0.5, 0.5)
        self.encoder_features = 64
        self.water_features = 512
        self.required_attack_ratio = 0.5
        self.device = torch.device("cuda")
        self.num_classes = 2
        self.use_dataset = 'COCO'
        # localization cover recover
        self.beta = (5000,1,1)
        self.num_epochs = 10
        self.train_batch_size = 2
        self.test_batch_size = 2
        self.learning_rate = 0.0001
        self.use_Vgg = False
        self.use_dataset = 'COCO'  # "ImageNet"
        self.MODELS_PATH = './output/models/'
        self.VALID_PATH = './sample/valid_coco/'
        self.TRAIN_PATH = './sample/train_coco/'
        self.TEST_PATH = './sample/test_coco/'
        self.skipTraining = False

        # Mean and std deviation of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
        if self.use_dataset == 'COCO':
            self.mean = [0.471, 0.448, 0.408]
            self.std = [0.234, 0.239, 0.242]
        else:
            self.std = [0.229, 0.224, 0.225]
            self.mean = [0.485, 0.456, 0.406]
