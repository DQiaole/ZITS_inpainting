import torch.nn as nn
import torch.nn.functional as F


class StructureUpsampling(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=0),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))
        self.out = nn.Sequential(nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, line):
        x = line
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.convs(x)
        x2 = self.out(x)

        return x, x2
