class Encoder_Localizer_config():
    """
    The HiDDeN network configuration.
    """

    def __init__(self, Height: int=224, Width: int=224, block_size: int=16):
        self.Height = Height
        self.Width = Width
        self.block_size = block_size
        self.decoder_channels = 64
        self.min_required_block = 64
        self.min_required_block_portion = 0.4
