import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torchvision
import os

class Data:
    @staticmethod
    def build_data(opt):

        transform = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * opt.channel, [0.5] * opt.channel)
        ])


        # dataloader with multiprocessing
        #dataloader = torch.utils.data.DataLoader(data, opt.batch_size, shuffle=True, num_workers=opt.workers)

        #data_dir = path + '/celeba/'  # this path depends on your computer
        #dset = torch.datasets.ImageFolder(data_dir, transform)
        #dataloader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True, num_workers=opt.workers)

        #dataset = torchvision.datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
        dataset = torchvision.datasets.MNIST('../data', train=True, download=True,transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=opt.batch_size,
                                                  shuffle=True, drop_last=True)


        return dataloader
