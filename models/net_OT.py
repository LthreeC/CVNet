import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        MODEL_NAME = 'efficientnet-b0'
        self.effnet = EfficientNet.from_name(MODEL_NAME)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.effnet._fc.in_features, num_classes)

    def forward(self, x):
        x = self.effnet.extract_features(x)
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
