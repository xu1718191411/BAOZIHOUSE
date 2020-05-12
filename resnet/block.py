import torch.nn as nn

class Block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        channel = output_channel
        self.conv1 = nn.Conv2d(input_channel, channel, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(channel, output_channel, kernel_size=(1, 1), padding=0)
        self.bn3 = nn.BatchNorm2d(output_channel)

        self.shortcut = self._shortcut(input_channel, output_channel)

        self.relu3 = nn.ReLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)

        h = self.conv3(h)
        h = self.bn3(h)

        shortcut = self.shortcut(x)

        y = h + shortcut
        y = self.relu3(y)
        return y

    def _shortcut(self, input_channel, output_channel):
        if input_channel != output_channel:
            return self._projection(input_channel, output_channel)
        else:
            return lambda x: x

    def _projection(self, input_channel, output_channel):
        return nn.Conv2d(input_channel, output_channel, kernel_size=(1, 1), padding=0)
