'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from qtorch import FloatingPoint
from qtorch.quant import Quantizer
from .td import Conv2d_TD, Linear_TD, Conv2d_col_TD

__all__ = ['MobileNetLP_TD']

def conv3x3_td(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False, groups=1, gamma=0.5, alpha=0.5, block_size=16):
    return Conv2d_TD(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=bias, gamma=gamma, alpha=alpha, block_size=block_size)
    # return Conv2d_col_TD(in_planes, out_planes, kernel_size=3, stride=stride,
    #                   padding=1, bias=False, gamma=gamma, alpha=alpha, block_size=block_size)

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, quant, stride=1, gamma=0.5, alpha=0.5, block_size=16):
        super(Block, self).__init__()
        self.conv1 = conv3x3_td(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_planes, gamma=gamma, alpha=alpha, block_size=block_size)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = conv3x3_td(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, gamma=gamma, alpha=alpha, block_size=block_size)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.quant = quant()

    def forward(self, x):
        out = self.conv1(x)
        out = self.quant(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.quant(out)

        out = self.conv2(out)
        out = self.quant(out)
        out = self.bn2(out)
        out = F.relu(out)
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, quant, num_classes, gamma=0.5, alpha=0.5, block_size=16):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32, stride=1, quant=quant, gamma=gamma, alpha=alpha, block_size=block_size)
        self.linear = nn.Linear(1024, num_classes)
        self.quant = quant()
        IBM_half = FloatingPoint(exp=6, man=9)
        self.quant_half = Quantizer(IBM_half, IBM_half, "nearest", "nearest")


    def _make_layers(self, in_planes, quant, stride=1, gamma=0.5, alpha=0.5, block_size=16):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, quant, stride, gamma=gamma, alpha=alpha, block_size=block_size))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant_half(x)
        x = self.conv1(x)
        x = self.quant(x)
        out = F.relu(self.bn1(x))
        out = self.layers(out)
        x = self.quant(x)

        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        x = self.quant_half(x)
        return out


class MobileNetLP_TD:
    base = MobileNet
    args = list()
    kwargs = {}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
