'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['RCSAResNet50']


class ChannelAvgPool(nn.Module):
    def forward(self, x):
        return  torch.mean(x,1).unsqueeze(1)


class EnAvgPooling(nn.Module):
    def __init__(self):
        super(EnAvgPooling, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool3 = nn.AdaptiveAvgPool2d(3)

        self.channelAvgPool = ChannelAvgPool()
        self.weight_conv = nn.Conv2d(1, 1, kernel_size=3, groups=1, padding=1, stride=1, bias=False)

    def forward(self, x):
        b, _, _, _ = x.size()
        y1 = self.avg_pool1(x)
        y3 = self.avg_pool3(x)
        yw = self.channelAvgPool(y3)
        yw=self.weight_conv(yw)
        yw=torch.sigmoid(yw)
        yw = y3*yw
        return y1+self.avg_pool1(yw)


class ChannelATT(nn.Module):
    def __init__(self,in_channel, channel, reduction=16):
        super(ChannelATT, self).__init__()
        self.enAvgPooling = EnAvgPooling()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        if in_channel != channel:
            self.att_fc = nn.Linear(in_channel,channel,bias=False)

    def forward(self, x):
        b, c, _, _ = x[0].size()
        y = self.enAvgPooling(x[0]).view(b, c)
        if x[1] is None:
            all_att = self.fc(y)
        else:
            pre_att = self.att_fc(x[1]) if hasattr(self, 'att_fc') else x[1]
            all_att = self.fc(y) + pre_att.view(b,c)
        return all_att

class SpatialATT(nn.Module):
    def __init__(self,in_channel, channel):
        super(SpatialATT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gather = nn.Sequential(
            ChannelAvgPool(),
            nn.Conv2d(1, 1, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.excite = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        if in_channel != channel:
            self.att_pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self, x):
        y = self.gather(x[0])
        y = nn.functional.interpolate(y, scale_factor=2, mode='nearest')
        y = self.excite(y)
        if x[1] is None:
            all_att = y
        else:
            pre_att = self.att_pool(x[1]) if hasattr(self, 'att_pool') else x[1]
            all_att = y + pre_att
        return y


class RCSALayer(nn.Module):
    def __init__(self,in_channel, channel, reduction=16):
        super(RCSALayer, self).__init__()
        self.channelATT = ChannelATT(in_channel, channel, reduction)
        self.spatialATT = SpatialATT(in_channel, channel)

    def forward(self, x):
        # x[0]: feature map; x[1]: previous channel attention; x[2]: previous spatial attention.
        channel_att_val = self.channelATT({0:x[0], 1:x[1]})
        spatial_att_val = self.spatialATT({0:x[0], 1:x[2]})
        feature_map = x[0]*torch.sigmoid(channel_att_val.unsqueeze(-1).unsqueeze(-1))\
                      *torch.sigmoid(spatial_att_val)
        return {0:feature_map,1:channel_att_val,2:spatial_att_val}












class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.rcsa = RCSALayer(in_channel=in_planes, channel=planes*self.expansion)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):

        out = F.relu(self.bn1(x[0]))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x[0]
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        # Add RCSA block
        out = self.rcsa({0:out,1:x[1],2:x[2]})
        out_x = out[0]+shortcut
        return {0: out_x,1:out[1],2:out[2]}


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
        self.rcsa = RCSALayer(in_channel=in_planes, channel=planes*self.expansion)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x[0]))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x[0]
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Add RCSA block
        out = self.rcsa({0: out, 1: x[1], 2: x[2]})
        out_x = out[0] + shortcut
        return {0: out_x, 1: out[1], 2: out[2]}


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
        # n, c, _, _ = out.size()
        # att = torch.zeros(n,c)
        out = {0:out, 1:None,2:None}
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out[0], 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def RCSAResNet18(num_classes=1000):
    return PreActResNet(PreActBlock, [2,2,2,2],num_classes)

def RCSAResNet34(num_classes=1000):
    return PreActResNet(PreActBlock, [3,4,6,3],num_classes)

def RCSAResNet50(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,4,6,3],num_classes)

def RCSAResNet101(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,4,23,3],num_classes)

def RCSAResNet152(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,8,36,3],num_classes)


def test():
    net = RCSAResNet18(num_classes=100)
    y = net((torch.randn(10,3,32,32)))
    print(y.size())

# test()
