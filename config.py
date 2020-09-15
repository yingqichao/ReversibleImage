class Encoder_Localizer_config():
    """
    The HiDDeN network configuration.
    """

    def __init__(self, Height: int=256, Width: int=256, block_size: int=16):
        self.Height = Height
        self.Width = Width
        self.block_size = block_size
        self.decoder_channels = 128
        self.min_required_block = 128
        self.min_required_block_portion = 0.4
        self.crop_size = (0.7, 0.7)
        self.encoder_features = 64
        self.water_features = 8
        self.required_attack_ratio = 0.5

