"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn
import torchvision.transforms as transforms
from .td import Conv2d_TD, Linear_TD

__all__ = ['VGG16_TD', 'VGG16BN_TD', 'VGG19_TD', 'VGG19BN_TD']


def make_layers(cfg, batch_norm=False, gamma=0.5, alpha=0.5, block_size=16):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if in_channels == 3:
                conv2d = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
            else:
                conv2d = Conv2d_TD(in_channels, filters, kernel_size=3, padding=1, gamma=gamma, 
                                    alpha=alpha, block_size=block_size)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
         512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False, gamma=0.5, alpha=0.5, block_size=16):
        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], batch_norm, gamma, alpha, block_size)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            Linear_TD(512, 512, gamma=gamma, alpha=alpha, block_size=block_size),
            nn.ReLU(True),
            nn.Dropout(),
            Linear_TD(512, 512, gamma=gamma, alpha=alpha, block_size=block_size),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Base:
    base = VGG
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


class VGG16_TD(Base):
    pass


class VGG16BN_TD(Base):
    kwargs = {'batch_norm': True}


class VGG19_TD(Base):
    kwargs = {'depth': 19}


class VGG19BN_TD(Base):
    kwargs = {'depth': 19, 'batch_norm': True}
