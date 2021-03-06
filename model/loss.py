import torch
from torch.nn import functional as F

torch.autograd.set_detect_anomaly(True)


def loss_nonsat(g, d, x_real, device):
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.z_dim, 1, 1, device=device)

    x_fake = g(z)
    d_real = d(x_real)
    d_fake = d(x_fake)

    g_loss = -F.logsigmoid(d_fake).mean()
    d_loss = -F.logsigmoid(d_real).mean() - F.logsigmoid(-d_fake).mean()

    return d_loss, g_loss




def loss_wgan(g, d, x_real, device):
    batch_size = x_real.shape[0]
    #z = torch.randn(batch_size, g.z_dim, 1, 1, device=device)
    #z = torch.randn(batch_size, g.z_dim, device=device)
    d_real = d(x_real)

    d_real_sigmoid = F.sigmoid(d_real)
    z = torch.rand_like(d_real_sigmoid, device=device)*0.2 + d_real_sigmoid

    z = z[torch.randperm(z.size(0))]

    x_fake = g(z)

    d_fake = d(x_fake)

    alpha = torch.rand(batch_size, 1, 1, 1, device=device)

    x_r = alpha * x_real + (1 - alpha) * x_fake
    gamma = 10
    d_r = d(x_r)

    grad = torch.autograd.grad(d_r.sum(), x_r, create_graph=True)
    grad_norm = grad[0].reshape(batch_size, -1).norm(dim=1)
    d_loss = (d_fake - d_real).mean() + gamma * ((grad_norm - 1) ** 2).mean()
    g_loss = -d_fake.mean()

    return d_loss, g_loss

