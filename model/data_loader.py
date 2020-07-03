import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
import torchvision
from config import Config
import os

class Data:
    @staticmethod
    def build_data():
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        opt = Config()
        transform = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        dataset = CIFAR10(root='../data/cifar10/', transform=transform, download=True)
        # dataloader with multiprocessing
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 opt.batch_size,
                                                 shuffle=True,
                                                 num_workers=opt.workers)
        print(ROOT_DIR)
        return dataloader
