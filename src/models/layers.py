import torch
import torch.nn as nn


class GateConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False):
        super(GateConv, self).__init__()
        self.out_channels = out_channels
        if transpose:
            self.gate_conv = nn.ConvTranspose2d(in_channels, out_channels * 2,
                                                kernel_size=kernel_size,
                                                stride=stride, padding=padding)
        else:
            self.gate_conv = nn.Conv2d(in_channels, out_channels * 2,
                                       kernel_size=kernel_size,
                                       stride=stride, padding=padding)

    def forward(self, x):
        x = self.gate_conv(x)
        (x, g) = torch.split(x, self.out_channels, dim=1)
        return x * torch.sigmoid(g)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False,
                 use_spectral_norm=False):
        super(Conv, self).__init__()
        self.out_channels = out_channels
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size, stride=stride,
                                           padding=padding, bias=not use_spectral_norm)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, bias=not use_spectral_norm)
        if use_spectral_norm:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x)


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class ResnetBlock(nn.Module):
    def __init__(self, input_dim, out_dim=None, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        if out_dim is not None:
            self.proj = nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=1, bias=False)
        else:
            self.proj = None
            out_dim = input_dim

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=0, dilation=dilation,
                                    bias=not use_spectral_norm), use_spectral_norm)
        )
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU(True)

        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
                                   spectral_norm(nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=0, dilation=1,
                                                           bias=not use_spectral_norm), use_spectral_norm))
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)
        y = self.conv1(x)
        y = self.bn1(y.to(torch.float32))
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y.to(torch.float32))
        out = x + y

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out
