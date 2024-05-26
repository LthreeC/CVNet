import torch
import torch.nn as nn
from torchvision.models.googlenet import BasicConv2d
import torch.nn.functional as F

class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, in_channels // 4, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, in_channels // 4, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(in_channels // 4, in_channels // 8, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(in_channels // 4, in_channels // 8, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, in_channels // 4, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(in_channels // 4, in_channels // 8, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(in_channels // 4, in_channels // 8, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, in_channels // 4, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)