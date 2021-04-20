import torch
import torch.nn as nn

class AffineCoupling1d(nn.Module):
    def __init__(self, dim, hid_dim, mode=1.):
        """
        dim (int) - dimnetion of the input data
        mode - 0 or 1 (from which number the mask starts)
        """
        super(AffineCoupling1d, self).__init__()
        self.mask = torch.arange(mode, mode + dim).unsqueeze(0) % 2
        self.s = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(dim, hid_dim)),
            nn.LeakyReLU(),
            nn.utils.weight_norm(nn.Linear(hid_dim, hid_dim)),
            nn.LeakyReLU(),
            nn.utils.weight_norm(nn.Linear(hid_dim, dim)),
            nn.Tanh()
        )
        self.t = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(dim, hid_dim)),
            nn.LeakyReLU(),
            nn.utils.weight_norm(nn.Linear(hid_dim, hid_dim)),
            nn.LeakyReLU(),
            nn.utils.weight_norm(nn.Linear(hid_dim, dim))
        )

    def forward(self, x):
        """
        Make a forward pass and compute log determinant of the transformation
        """
        self.mask = self.mask.to(x.device)
        bx = self.mask * x
        s_bx = self.s(bx)
        t_bx = self.t(bx)
        z = bx + (1 - self.mask) * (x * torch.exp(s_bx) + t_bx)
        log_det = ((1 - self.mask) * s_bx).sum(1)
        return z, log_det

    def inverse(self, z):
        """
        Covert noize to data via inverse transformation
        """
        self.mask = self.mask.to(z.device)
        bz = self.mask * z
        s_bz = self.s(bz)
        t_bz = self.t(bz)
        x = bz + (1 - self.mask) * ((z - t_bz) * torch.exp(-s_bz))
        return x


class ResUnit(nn.Module):
    '''
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Identity Mappings in Deep Residual Networks", https://arxiv.org/abs/1603.05027
    The unit used here is called the full pre-activation.
    '''
    def __init__(self, number_channels, kernel_size=3, stride=1, padding=1, dilation=1, activation=nn.ReLU(True)):
        super(ResUnitBN, self).__init__()

        self.activation = activation
        # self.bn1 = nn.BatchNorm2d(number_channels)

        self.conv1 = nn.Conv2d(number_channels, number_channels, kernel_size, stride, padding, dilation)

        # self.bn2 = nn.BatchNorm2d(number_channels)
        self.conv2 = nn.Conv2d(number_channels, number_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        residual = x

        h_bn_1 = self.bn1(x)
        h_act_bn_1 = self.activation(h_bn_1)
        h_1 = self.conv1(h_act_bn_1)

        h_bn_2 = self.bn2(h_1)
        h_act_bn_2 = self.activation(h_bn_2)
        h_2 = self.conv2(h_act_bn_2)

        out = h_2 + residual

        return out
