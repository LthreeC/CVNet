import torch
import torch.nn as nn


class ECAModule(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(channels, channels, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # Global average pooling
        y = self.avg_pool(x).view(b, c, 1)
        # Apply 1D convolution
        y = self.conv(y).view(b, c)
        # Apply sigmoid activation
        y = self.sigmoid(y).view(b, c, 1, 1)
        # Scale the input features with the attention map
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class Double_Channel_Attn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2 * in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        attention = self.spatial_attention(x)
        return x * attention

class Half_Channel_Attn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # * 4
        self.conv1 = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels * 4)
        self.silu1 = nn.SiLU()

        # ECA
        self.eca = ECAModule(in_channels * 4)

        # / 8
        self.conv3 = nn.Conv2d(in_channels * 4, in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(in_channels // 2)
        self.silu3 = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu1(x)

        x = self.eca(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.silu3(x)
        return x