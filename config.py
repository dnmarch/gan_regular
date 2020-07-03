class Config:
    image_size = None
    lr = 0.00005
    nz = 100  # noise dimension
    image_size = 64
    image_size2 = 64
    nc = 3  # chanel of img
    ngf = 64  # generate channel
    ndf = 64  # discriminative channel
    beta1 = 0.5
    batch_size = 32
    max_epoch = 20  # =1 when debug
    workers = 0
    gpu = True  # use gpu or not
    clamp_num = 0.01  # WGAN clip gradient
    device = "cuda"

opt = Config()