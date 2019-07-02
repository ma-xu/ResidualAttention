'''Pre-activation ResNet in PyTorch.


'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init

__all__ = ['BAMResNet50']


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        # super(ChannelGate, self).__init__()
        # self.gate_c = nn.Sequential()
        # self.gate_c.add_module('flatten', Flatten() )
        # gate_channels = [gate_channel]
        # gate_channels += [gate_channel // reduction_ratio] * num_layers
        # gate_channels += [gate_channel]
        # for i in range( len(gate_channels) - 2 ):
        #     self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
        #     self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
        #     self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        # self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )

        #Reimplemented by Xu Ma. Follows the code and paper
        super(ChannelGate, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential()
        self.fc.add_module('gate_c_fc_0', nn.Linear(gate_channel,gate_channel//reduction_ratio))
        self.fc.add_module('gate_c_relu_1',nn.ReLU(inplace=True))
        self.fc.add_module('gate_c_fc_final',nn.Linear(gate_channel // reduction_ratio, gate_channel))

    def forward(self, in_tensor):
        # avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        # return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)
        b, c, _, _ = in_tensor.size()
        out = self.avgpool(in_tensor).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        out = out.expand_as(in_tensor)
        return out


class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)
class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att = 1 + torch.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor



class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
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
        self.bam1 = BAM(64 * block.expansion)
        self.bam2 = BAM(128 * block.expansion)
        self.bam3 = BAM(256 * block.expansion)
        if init_weights:
            self._initialize_weights()
        # init for BAM and CBAM
        for key in self.state_dict():
            if key.split('.')[-1] == "weight":
                if "conv" in key:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                    # init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == 'bias':
                self.state_dict()[key][...] = 0

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
        out = self.bam1(out)
        out = self.layer2(out)
        out = self.bam2(out)
        out = self.layer3(out)
        out = self.bam3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def BAMResNet18(num_classes=1000):
    return PreActResNet(PreActBlock, [2,2,2,2],num_classes)

def BAMResNet34(num_classes=1000):
    return PreActResNet(PreActBlock, [3,4,6,3],num_classes)

def BAMResNet50(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,4,6,3],num_classes)

def BAMResNet101(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,4,23,3],num_classes)

def BAMResNet152(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,8,36,3],num_classes)


def test():
    net = BAMResNet18(num_classes=100)
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()
