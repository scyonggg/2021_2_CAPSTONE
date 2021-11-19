import torch
import torch.nn as nn


class ResNet50_layer4(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50_layer4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 64, 256, False),
            ResidualBlock(256, 64, 256, False),
            ResidualBlock(256, 64, 256, True)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(256, 128, 512, False),
            ResidualBlock(512, 128, 512, False),
            ResidualBlock(512, 128, 512, False),
            ResidualBlock(512, 128, 512, True)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(512, 256, 1024, False),
            ResidualBlock(1024, 256, 1024, False),
            ResidualBlock(1024, 256, 1024, False),
            ResidualBlock(1024, 256, 1024, False),
            ResidualBlock(1024, 256, 1024, False),
            ResidualBlock(1024, 256, 1024, True)
        )
        self.layer5 = nn.Sequential(
            ResidualBlock(1024, 512, 2048, False),
            ResidualBlock(2048, 512, 2048, False),
            ResidualBlock(2048, 512, 2048, False)
        )
        self.fc = nn.Linear(2048, 10)
        self.avgpool = nn.AvgPool2d((2, 2), stride=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)

        return out