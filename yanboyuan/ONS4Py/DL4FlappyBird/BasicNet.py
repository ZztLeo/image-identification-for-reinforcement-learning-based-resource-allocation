import torch.nn as nn
from torch.nn.modules.loss import _Loss, _assert_no_grad
import torch


class SimpleNet(nn.Module):
    """

    """

    def __init__(self):
        """

        """
        super(SimpleNet, self).__init__()
        self.conv = nn.Sequential(
            # 80x80x4 --> 20x20x32
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # 20x20x32 --> 10x10x32
            nn.MaxPool2d(kernel_size=2),

            # 10x10x32 --> 5x5x64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 5x5x64 --> 3x3x64
            nn.MaxPool2d(kernel_size=2, padding=1),

            # 3x3x64 --> 3x3x64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 3x3x64 --> 2x2x64
            nn.MaxPool2d(kernel_size=2, padding=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=2)
        )

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight, mean=0, std=0.01)
                nn.init.constant(m.bias, 0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, mean=0, std=0.01)
                nn.init.constant(m.bias, 0.01)

    def forward(self, x):
        """

        :param x: shape is [N, 4, 80, 80]
        :return: shape is [N, 2]
        """
        x = self.conv(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class L2Loss(_Loss):  # TODO need test
    """
    L2 Loss calculation
    """
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, inp, target):
        _assert_no_grad(target)
        return torch.sum(torch.pow(target - inp, 2)) / inp.size()[0]