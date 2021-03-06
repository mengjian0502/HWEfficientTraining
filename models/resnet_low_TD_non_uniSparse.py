import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
from qtorch import FloatingPoint
from qtorch.quant import Quantizer
from torch.nn import init
from .td_non_uniSparse import Conv2d_TD, Linear_TD, Conv2d_col_TD


__all__ = ['ResNet20LP_TD_LayerSort','ResNet32LP_TD_LayerSort']

def conv3x3_td(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False, gamma=0.5, alpha=0.5, block_size=16, non_uni_sparse=True, threshold=0.0):
    return Conv2d_TD(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=bias, gamma=gamma, alpha=alpha, block_size=block_size, non_uni_sparse=non_uni_sparse, threshold=threshold)
    # return Conv2d_col_TD(in_planes, out_planes, kernel_size=3, stride=stride,
    #                   padding=1, bias=False, gamma=gamma, alpha=alpha, block_size=block_size)

class ResNetBasicblock(nn.Module):
  expansion = 1
  """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
  def __init__(self, inplanes, planes, quant, stride=1, downsample=None, gamma=0.5, alpha=0.5, block_size=16, non_uni_sparse=True, threshold=0.0):
    super(ResNetBasicblock, self).__init__() 
    # self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)   # full precision
    self.conv_a = conv3x3_td(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, gamma=gamma, alpha=alpha, block_size=block_size, non_uni_sparse=non_uni_sparse, threshold=threshold)
    self.bn_a = nn.BatchNorm2d(planes)
    # self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # full precision
    self.conv_b = conv3x3_td(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, gamma=gamma, alpha=alpha, block_size=block_size, non_uni_sparse=non_uni_sparse, threshold=threshold)
    self.bn_b = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.quant = quant()

  def forward(self, x):
    residual = x

    basicblock = self.conv_a(x)
    basicblock = self.quant(basicblock)
    basicblock = self.bn_a(basicblock)
    basicblock = F.relu(basicblock, inplace=True)
    basicblock = self.quant(basicblock)

    basicblock = self.conv_b(basicblock)
    basicblock = self.quant(basicblock)
    basicblock = self.bn_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    return F.relu(residual + basicblock)  


class CifarResNet(nn.Module):
  """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
  def __init__(self, quant, depth, num_classes, gamma=0.5, alpha=0.5, block_size=16, non_uni_sparse=True, threshold=0.0):
    """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
    super(CifarResNet, self).__init__()

    block = ResNetBasicblock

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
    layer_blocks = (depth - 2) // 6
    print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))
    self.non_uni_sparse = non_uni_sparse
    self.threshold = threshold

    self.num_classes = num_classes
    self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

    self.bn_1 = nn.BatchNorm2d(16)

    self.inplanes = 16
    self.stage_1 = self._make_layer(block, 16, layer_blocks, quant, 1, gamma=gamma, alpha=alpha, block_size=block_size)
    self.stage_2 = self._make_layer(block, 32, layer_blocks, quant, 2, gamma=gamma, alpha=alpha, block_size=block_size)
    self.stage_3 = self._make_layer(block, 64, layer_blocks, quant, 2, gamma=gamma, alpha=alpha, block_size=block_size)
    self.avgpool = nn.AvgPool2d(8)
    self.classifier = nn.Linear(64*block.expansion, num_classes)
    self.quant = quant()
    IBM_half = FloatingPoint(exp=6, man=9)
    self.quant_half = Quantizer(IBM_half, IBM_half, "nearest", "nearest")

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, quant, stride=1, gamma=0.5, alpha=0.5, block_size=16):
    downsample = None
    # if stride != 1 or self.inplanes != planes * block.expansion:
    #   downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
    if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = nn.Sequential(
          Conv2d_TD(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False, gamma=gamma, alpha=alpha, block_size=block_size, non_uni_sparse=self.non_uni_sparse, threshold=self.threshold),
          nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(self.inplanes, planes, quant, stride, downsample, gamma=gamma, alpha=alpha, block_size=block_size, non_uni_sparse=self.non_uni_sparse, threshold=self.threshold))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, quant, gamma=gamma, alpha=alpha, block_size=block_size, non_uni_sparse=self.non_uni_sparse, threshold=self.threshold))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.quant_half(x)
    x = self.conv_1_3x3(x)
    x = self.quant(x)
    x = F.relu(self.bn_1(x), inplace=True)
    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = self.quant(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    x = self.quant_half(x)
    return x


class ResNet32LP_TD_LayerSort:
  base = CifarResNet
  args = list()
  kwargs = {'depth': 32, 'non_uni_sparse': True}
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

class ResNet20LP_TD_LayerSort:
  base = CifarResNet
  args = list()
  kwargs = {'depth': 20, 'non_uni_sparse': True}
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



def tern_resnet20(num_classes=10):
  """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 20, num_classes)
  return model


def tern_resnet32(num_classes=10):
  """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 32, num_classes)
  return model


def tern_resnet44(num_classes=10):
  """Constructs a ResNet-44 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 44, num_classes)
  return model


def tern_resnet56(num_classes=10):
  """Constructs a ResNet-56 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 56, num_classes)
  return model

def tern_resnet110(num_classes=10):
  """Constructs a ResNet-110 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 110, num_classes)
  return model

