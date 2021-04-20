import torch
import torch.nn as nn


class EncoderMnist(nn.Module):
    def __init__(self, z_dim, h_dim=300, **kwargs):
        super(EncoderMnist, self).__init__()

        self.q_z_layers = nn.Sequential(
            nn.Linear(784, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(inplace=True))

        self.q_z_mean = nn.Linear(h_dim, z_dim)
        self.q_z_logvar = nn.Sequential(nn.Linear(h_dim, z_dim),
                                        nn.Hardtanh(min_val=-10., max_val=4.))

    def forward(self, x):
        x = self.q_z_layers(x)
        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mean, z_q_logvar


class DecoderMnist(nn.Module):
    def __init__(self, z_dim, h_dim, **kwargs):
        super(DecoderMnist, self).__init__()

        self.p_x_layers = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(inplace=True))

        self.p_x_mean = nn.Sequential(nn.Linear(h_dim, 784),
                                      nn.Sigmoid())

    def forward(self, x):
        z = self.p_x_layers(x)
        x_mean = self.p_x_mean(z)
        x_logvar = torch.zeros_like(x_mean)
        return x_mean, x_logvar


class EncoderMnistConv(nn.Module):
    def __init__(self, z_dim, **kwargs):
        super(EncoderMnistConv, self).__init__()

        self.q_z_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 12
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True),
            # 4
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True),
            # 2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.q_z_mean = nn.Sequential(
            nn.Conv2d(256, z_dim, kernel_size=3, stride=2, padding=1),
            nn.Flatten()
        )
        self.q_z_logvar = nn.Sequential(
            nn.Conv2d(256, z_dim, kernel_size=3, stride=2, padding=1),
            nn.Hardtanh(min_val=-10., max_val=4.),
            nn.Flatten()
        )

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.reshape(-1, 1, 28, 28)
        x = self.q_z_layers(x)
        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mean, z_q_logvar


class DecoderMnistConv(nn.Module):
    def __init__(self, z_dim, **kwargs):
        super(DecoderMnistConv, self).__init__()

        self.p_x_layers = nn.Sequential(
            # 3
            nn.ConvTranspose2d(z_dim, 256, 3, stride=1, padding=0),
            nn.ReLU(),

            # 7
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=0),
            nn.ReLU(),

            # 14
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),

            # 28
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x_mean = self.p_x_layers(z.reshape(-1,z.shape[1],1,1))
        x_logvar = torch.zeros_like(x_mean)
        return x_mean, x_logvar


def get_architecture(args):
    params_enc = {'h_dim': args.h_dim, 'z_dim': args.z_dim}
    params_dec = {'h_dim': args.h_dim, 'z_dim': args.z_dim}

    if args.dataset_name == 'mnist' or args.dataset_name == 'fashion_mnist':
        if args.arc_type == 'mlp':
            enc = EncoderMnist
            dec = DecoderMnist
        elif args.arc_type == 'conv':
            enc = EncoderMnistConv
            dec = DecoderMnistConv

    arc_getter = lambda : (enc(**params_enc), dec(**params_dec))
    return arc_getter, args