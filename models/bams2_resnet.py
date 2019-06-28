'''Bottleneck  Attention Module (with only spatial attnetion) Pre-activation ResNet in PyTorch.

1. relu(bn(Conv(c,c/r,1)))
2. relu(bn(DilateConv(c/r,c/r,3,d,d)))
3. relu(bn(DilateConv(c/r,c/r,3,d,d)))
4. conv(c/r,1,1)
5. x*sigmod(attention)


Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['BAMS2ResNet18', 'BAMS2ResNet34', 'BAMS2ResNet50', 'BAMS2ResNet101', 'BAMS2ResNet152']



class BAMSLayer(nn.Module):
    def __init__(self,in_channel,reduction_ratio=16,dilation_conv_num=2,dilation_val=2):
        super(BAMSLayer, self).__init__()
        self.bam_s = nn.Sequential()
        self.bam_s.add_module('bam_s_conv_reduce0',
                               nn.Conv2d(in_channel, in_channel // reduction_ratio, kernel_size=1))
        self.bam_s.add_module('bam_s_bn_reduce0', nn.BatchNorm2d(in_channel // reduction_ratio))
        self.bam_s.add_module('bam_s_relu_reduce0',nn.ReLU() )
        for i in range(dilation_conv_num):
            self.bam_s.add_module('bam_s_conv_di_%d' % i,
                                   nn.Conv2d(in_channel // reduction_ratio, in_channel // reduction_ratio,
                                             kernel_size=3, \
                                             padding=dilation_val, dilation=dilation_val))
            self.bam_s.add_module('bam_s_bn_di_%d' % i, nn.BatchNorm2d(in_channel // reduction_ratio))
            self.bam_s.add_module('bam_s_relu_di_%d' % i, nn.ReLU())
        self.bam_s.add_module('bam_s_conv_final', nn.Conv2d(in_channel // reduction_ratio, 1, kernel_size=1))


    def forward(self, x):
        att = torch.sigmoid(self.bam_s(x).expand_as(x))
        return att*x


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.sa = BAMSLayer(planes)
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.sa(out)
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.sa = BAMSLayer(planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out = self.sa(out)
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000,init_weights=True):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m, 'bias.data'):
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def BAMS2ResNet18(num_classes=1000):
    return PreActResNet(PreActBlock, [2,2,2,2],num_classes)

def BAMS2ResNet34(num_classes=1000):
    return PreActResNet(PreActBlock, [3,4,6,3],num_classes)

def BAMS2ResNet50(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,4,6,3],num_classes)

def BAMS2ResNet101(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,4,23,3],num_classes)

def BAMS2ResNet152(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,8,36,3],num_classes)


def test():
    # x=torch.randn(2,3,4,4)
    # sa = SALayer()
    # y=sa(x)
    # print(y.size())
    # print(y)
    net = BAMS2ResNet18(num_classes=100)
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()
