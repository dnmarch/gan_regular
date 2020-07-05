import torch
import torch.nn as nn


class Generator2(nn.Module):
    def __init__(self, channels, z_dim):
        super().__init__()
        self.z_dim = z_dim
        channel_1 = 12
        channel_2 = 20
        channel_3 = 25
        channel_4 = 32

        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=channel_1, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=channel_1),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=channel_1, out_channels=channel_2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=channel_2),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=channel_2, out_channels=channel_3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=channel_3),
            nn.ReLU(True),

            # State (256x16x16)
            # nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1)
            # out_channels = 3, get back to 3x32x32

            nn.ConvTranspose2d(in_channels=channel_3, out_channels=channel_4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=channel_4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32, out_channels=channels, kernel_size=4, stride=2, padding=1),

        )

        self.output = nn.Tanh()

    def forward(self, input):
        x = self.main_module(input)
        return self.output(x)


class Discriminator2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        channel_1 = 12
        channel_2 = 25
        channel_3 = 10
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=channel_1, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(channel_1, affine=True),
            nn.LeakyReLU(0.2),

            # State (256x16x16)
            nn.Conv2d(in_channels=channel_1, out_channels=channel_2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(channel_2, affine=True),
            nn.LeakyReLU(0.2),

            # State (512x8x8)
            nn.Conv2d(in_channels=channel_2, out_channels=channel_3, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(channel_3, affine=True),
            nn.LeakyReLU(0.2),
            # output of main module --> State (1024x4x4)
            nn.Conv2d(in_channels=channel_3, out_channels=1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, input):
        return self.main_module(input)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.size(0), *self.shape)


class Generator(torch.nn.Module):
    def __init__(self, num_channels=1, dim_z=64):
        super().__init__()
        self.z_dim = dim_z
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_z, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(512, 64 * 7 * 7),
            torch.nn.BatchNorm1d(64 * 7 * 7),
            torch.nn.ReLU(inplace=True),
            Reshape(64, 7, 7),

            torch.nn.PixelShuffle(2),
            torch.nn.Conv2d(64 // 4, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),

            torch.nn.PixelShuffle(2),
            torch.nn.Conv2d(32 // 4, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(torch.nn.Module):
    def __init__(self, num_channels=1, out_dim=512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),

            Reshape(64 * 7 * 7),
            torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Linear(512, out_dim)
            #Reshape()
        )

    def forward(self, x):
        return self.net(x)
