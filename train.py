import torch
from torch.optim import RMSprop
from config import Config
from model.data_loader import Data
from model.net import Generator, Discriminator
from torch.autograd import Variable
from torchvision.utils import make_grid
from pylab import plt
import tqdm
from model.loss import *

opt = Config()

def weight_init(m):
    # weight_initialization: important for wgan
    class_name=m.__class__.__name__
    if class_name.find('Conv')!=-1:
        m.weight.data.normal_(0,0.02)
    elif class_name.find('Norm')!=-1:
        m.weight.data.normal_(1.0,0.02)


def plot_image(g, x_real):
    batch_size = x_real
    z_test = torch.randn(batch_size, g.z_dim, 1, 1, device=device)
    with torch.no_grad():
        g.eval()
        x_test = (g(z_test) + 1) / 2.
        imgs = make_grid(x_test.data * 0.5 + 0.5).cpu()  # CHW
        plt.imshow(imgs.permute(1, 2, 0).numpy())  # HWC
        plt.show()
        g.train()


def train(loss, data_loader, optim):
    netg = Generator(3, 100).to(opt.device)
    netd = Discriminator(3).to(opt.device)
    netg.apply(weight_init)
    netd.apply(weight_init)

    d_optimizer = optim(netd.parameters(), lr=opt.lr)
    g_optimizer = optim(netg.parameters(), lr=opt.lr)


    for i in range(opt.max_epoch):
        for x_real, y_real in tqdm.tqdm(data_loader):

            x_real, y_real = x_real.to(opt.device), y_real.to(opt.device)

            d_loss, g_loss = loss(netg, netd, x_real, opt.device)

            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            d_loss, g_loss = loss(netg, netd, x_real, opt.device)

            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

        print("epoch: %d, d_loss: %.3f, g_loss: %.3f"%(i, d_loss, g_loss))
        if (i + 1) % 5 or i == opt.max_epoch - 1:
            plot_image(g, x_real)




data_loader = Data.build_data()
train(loss_wgan, data_loader, torch.optim.RMSprop)