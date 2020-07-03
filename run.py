import torch

from config import Config
from model.data_loader import Data
from model.loss import loss_wgan
from train import train

opt = Config()
opt.max_epoch = 20
data_loader = Data.build_data()
g, d = train(loss_wgan, data_loader, torch.optim.RMSprop, opt)
