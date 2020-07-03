from model.net import Generator, Discriminator
from model.loss import *
from torchvision.utils import make_grid
from pylab import plt
import tqdm




def weight_init(m):
    # weight_initialization: important for wgan
    class_name=m.__class__.__name__
    if class_name.find('Conv')!=-1:
        m.weight.data.normal_(0,0.02)
    elif class_name.find('Norm')!=-1:
        m.weight.data.normal_(1.0,0.02)


def plot_image_test(g, batch_size, device):
    z_test = torch.randn(batch_size, g.z_dim, 1, 1, device=device)
    with torch.no_grad():
        g.eval()
        x_test = (g(z_test) + 1) / 2.
        imgs = make_grid(x_test.data * 0.5 + 0.5).cpu()  # CHW
        plt.imshow(imgs.permute(1, 2, 0).numpy())  # HWC
        plt.show()
        g.train()

def plot_image(x_real):
    x_real = (x_real + 1) / 2
    imgs = make_grid(x_real.data * 0.5 + 0.5).cpu()  # CHW
    plt.imshow(imgs.permute(1, 2, 0).numpy())  # HWC
    plt.show()


def train(loss, data_loader, optim, opt):
    image, _ = next(iter(data_loader))
    #channel = image[0].shape[1]
    plot_image(image)
    channel = opt.channel
    netg = Generator(channel, opt.z_dim).to(opt.device)
    netd = Discriminator(channel).to(opt.device)
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
        if (i + 1) % 2 or i == opt.max_epoch - 1:
            plot_image_test(netg, 36, opt.device)


    return netd, netg
