import torch
import torch.nn as nn
import torch.nn.functional as F


class WideModule(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideModule, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:  # Shortcut 부분에서 input과 output Feature Map이 달라진 경우 보정시켜줌.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = nn.ReLU(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = nn.ReLU(out)
        out = self.conv2(out)

        out += self.shortcut(x)
        return out


class WideResnet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes=10):
        super(WideResnet, self).__init__()
        self.in_planes = 16

        n = (depth - 4) / 6     # 최소 CONV : (conv1,2,3,4) + 6 * N
        k = widen_factor

        groups = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, groups[0])
        self.layer1 = self._wide_layer(WideModule, groups[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideModule, groups[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideModule, groups[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(groups[3])
        self.linear = nn.Linear(groups[3], num_classes)

    def _wide_layer(self, block: str, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
#
# if __name__ == "__main__":
#     net = WideResnet(28, 10, 0.3, 10)   # N=28, k=10, dropout rate = 0.3일 때 가장 결과가 좋았다고 함.
