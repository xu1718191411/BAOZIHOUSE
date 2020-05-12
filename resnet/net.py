import torch.nn as nn
import torch
import torch.nn.functional as F

from resnet.block import Block


class ResNet50(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block0 = self._building_block(output_channel=256, input_channel=64)

        self.block1 = nn.ModuleList([self._building_block(output_channel=256, input_channel=256) for _ in range(2)])

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=(2, 2))

        self.block2 = nn.ModuleList([self._building_block(input_channel=512, output_channel=512) for _ in range(4)])

        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(2, 2))

        self.block3 = nn.ModuleList([self._building_block(input_channel=1024, output_channel=1024) for _ in range(6)])

        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(2, 2))

        self.block4 = nn.ModuleList([self._building_block(input_channel=2048, output_channel=2048) for i in range(3)])

        self.avg_pool = GlobalAvgPool2D()

        self.fc = nn.Linear(2048, 1000)
        self.out = nn.Linear(1000, output_dim)


    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)

        h = self.pool1(h)
        h = self.block0(h)

        for block in self.block1:
            h = block(h)

        h = self.conv2(h)

        for block in self.block2:
            h = block(h)

        h = self.conv3(h)

        for block in self.block3:
            h = block(h)

        h = self.conv4(h)

        for block in self.block4:
            h = block(h)

        h = self.avg_pool(h)

        h = self.fc(h)
        h = torch.relu(h)
        h = self.out(h)

        y = torch.log_softmax(h,dim=-1)

        return y


    def _building_block(self, output_channel, input_channel=None):
        if input_channel == None:
            input_channel = output_channel
        return Block(input_channel, output_channel)


class GlobalAvgPool2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))
