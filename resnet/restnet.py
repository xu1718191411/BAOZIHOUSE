import torch
import torchvision

# net = torchvision.models.resnet50()
#
# print(net)

class RestNet50(torch.nn.Module):
    def __init__(self):
        super(RestNet50, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                     bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = torch.nn.Sequential(
            Bottleneck(64, 64, 256, True),
            Bottleneck(256, 64, 256, False),
            Bottleneck(256, 64, 256, False),
        )

        self.layer2 = torch.nn.Sequential(
            Bottleneck(256, 128, 512, True),
            Bottleneck(512, 128, 512, False),
            Bottleneck(512, 128, 512, False),
            Bottleneck(512, 128, 512, False)
        )

        self.layer3 = torch.nn.Sequential(
            Bottleneck(512, 256, 1024, True),
            Bottleneck(1024, 256, 1024, False),
            Bottleneck(1024, 256, 1024, False),
            Bottleneck(1024, 256, 1024, False),
            Bottleneck(1024, 256, 1024, False),
            Bottleneck(1024, 256, 1024, False),
        )

        self.layer4 = torch.nn.Sequential(
            Bottleneck(1024, 512, 2048, True),
            Bottleneck(2048, 512, 2048, False),
            Bottleneck(2048, 512, 2048, False)
        )

        self.avgPool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgPool(out)
        out = out.flatten(1)
        out = self.fc(out)
        return out


class Bottleneck(torch.nn.Module):
    def __init__(self, input_channel, middle_channel, output_channel, down_sample):
        super(Bottleneck, self).__init__()
        self.inputChannel = input_channel
        self.middleChannel = middle_channel
        self.outputChannel = output_channel
        self.conv1 = torch.nn.Conv2d(input_channel, middle_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(middle_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = torch.nn.Conv2d(middle_channel, middle_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                     bias=False)
        self.bn2 = torch.nn.BatchNorm2d(middle_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = torch.nn.Conv2d(middle_channel, output_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = torch.nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu = torch.nn.ReLU(inplace=True)

        if down_sample:
            self.downSample = torch.nn.Sequential(
                torch.nn.Conv2d(input_channel, output_channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
                torch.nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        else:
            self.downSample = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downSample is not None:
            out2 = self.downSample(x)
            out += out2

        out = self.relu(out)
        return out


# net2 = RestNet50().cuda()
# input = torch.rand((1, 3, 640, 608)).cuda()
# input = input / 255.0
# input = input.float()
# output = net2(input)
# print("output is ....")
# print(output)