# TODO: still thinking about the best way to handle this

basic_autoencoder = ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            ResNetLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="avg"),
            ResNetLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max")
        ],
        decoder_layers=[
            ResNetLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="nearest"),
            ResNetLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="bilinear"),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, activation="Sigmoid", sampling="upsample", upsample_method="nearest")
        ],
        activation="ReLU",
        norm="batch")