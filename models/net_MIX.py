from models.base import *
from models.utils import *

class MIX(nn.Module):
    def __init__(self, block=BasicBlock, ic=3, oc=16, num_classes=10):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(ic, oc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(oc),
            nn.ReLU(True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.hc1 = Half_Channel_Attn(32)

        self.layer3 = Tree(block, 16, 32, level=1, stride=1)


        # 从这里开始只改深度

        # self.layer4 = Tree(block, 32, 64, level=1, stride=1)

        # self.branch1_1 = Half_Channel_Attn(64)
        # self.branch1_2 = Tree(block, 32, 64, level=2, stride=2)

        self.branch2_1 = SS_Conv_SSM(hidden_dim=256, drop_path=0.1, attn_drop_rate=0.1, d_state=2)
        self.branch2_2 = SpatialAttention()
        self.branch2_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch3_1 = Double_Channel_Attn(32)
        self.branch3_2 = InceptionE(64)
        self.branch3_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.branch3_4 = Half_Channel_Attn(64)

        # self.layer6 = Tree(block, 128, 128, level=2, stride=2)

        # 128输入，ECA 通道/2
        self.half_128 = Half_Channel_Attn(128)

        # 64输入，ECA 通道/2
        self.half_64 = Half_Channel_Attn(64)

        # self.channel_192_128(192, 128)

        # 直接通道权重
        self.SPA = SpatialAttention()


        # 后边才改主要改大小，改一点深度
        # 最后64个特征
        self.blockout = nn.Sequential(
            # Tree(block, 192, 128, level=1, stride=2),
            nn.MaxPool2d(2, 2),
            Half_Channel_Attn(128),
            Tree(block, 64, 32, level=1, stride=2),
            Tree(block, 32, 32, level=1, stride=2),
            nn.AvgPool2d(2, 2),
            Half_Channel_Attn(32),
            nn.Flatten(),
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(16 * 8 * 8, 32),
            # nn.Flatten(),  # Flatten the output to [batch_size, 128]
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)

        out = self.hc1(out)
        out = self.layer3(out)
        # out = self.layer4(out)

        # out1 = self.branch1_1(out)
        # out1 = self.branch1_2(out1)

        out2 = self.branch2_1(out)
        out2 = out2 * self.branch2_2(out2)
        out2 = self.branch2_3(out2)

        out3 = self.branch3_1(out)
        out3 = self.branch3_2(out3)
        out3 = self.branch3_3(out3)
        out3 = self.branch3_4(out3)

        # print(out2.shape)
        # print(out3.shape)

        # print("showing", out1.shape, out2.shape, out3.shape)
        # out = self.half_64(out1 + out2 + out3)
        # out = self.SPA(out1 + out2 + out3) * (out1 + out2 + out3)
        out = torch.cat((out2, out3, out3, out2), dim=1)
        out = self.SPA(out) * out

        # print("showing", out.shape)

        out = self.blockout(out)

        return out


if __name__ == '__main__':
    model = MIX().cuda()
    show_model(model, name="mix", path="../modelpng")