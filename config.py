import torch

class Encoder_Localizer_config():
    """
    The HiDDeN network configuration.
    """

    def __init__(self, Height: int=256, Width: int=256, block_size: int=16):
        self.Height = Height
        self.Width = Width
        self.block_size = block_size
        self.decoder_channels = 128
        self.min_required_block = 64
        self.min_required_block_portion = 0.4
        self.crop_size = (0.7, 0.7)
        self.encoder_features = 64
        self.water_features = 512
        self.required_attack_ratio = 0.5
        self.device = torch.device("cuda")
        self.num_classes = 2
        self.use_dataset = 'COCO'

        if self.use_dataset == 'COCO':
            self.mean = [0.471, 0.448, 0.408]
            self.std = [0.234, 0.239, 0.242]
        else:
            self.std = [0.229, 0.224, 0.225]
            self.mean = [0.485, 0.456, 0.406]