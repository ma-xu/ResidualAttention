'''Pre-activation ResNet in PyTorch.


'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['RCBAMResNet50']


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, in_channel,gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        if in_channel != gate_channels:
            self.att_fc = nn.Linear(in_channel,gate_channels,bias=False)

    def forward(self, input):
        x = input[0]
        pre_channel_att = input[1]
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        if pre_channel_att is None:
            channel_att_sum = channel_att_raw
        else:
            pre_channel_att = self.att_fc(pre_channel_att) if hasattr(self, 'att_fc') else pre_channel_att
            channel_att_sum = channel_att_sum + pre_channel_att

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return {0:x * scale,1:channel_att_sum}


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self,in_channel,channel):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        if in_channel != channel:
            self.att_pool = nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self, input):
        x = input[0]
        pre_spatial_att = input[1]
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        if pre_spatial_att is None:
            x_out = x_out
        else:
            pre_spatial_att = self.att_pool(pre_spatial_att) if hasattr(self, 'att_pool') else pre_spatial_att
            x_out = x_out + pre_spatial_att

        scale = torch.sigmoid(x_out) # broadcasting
        return {0:x * scale,1:x_out}


class RCBAM(nn.Module):
    def __init__(self,in_channel, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(RCBAM, self).__init__()
        self.ChannelGate = ChannelGate(in_channel,gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(in_channel,gate_channels)
    def forward(self, x):
        x_out = self.ChannelGate({0:x[0],1:x[1]})
        channel_att = x_out[1]
        if not self.no_spatial:
            x_out = self.SpatialGate({0:x_out[0],1:x[2]})
        return {0:x_out[0],1:channel_att,2:x_out[1]}


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.cbam = RCBAM(in_planes,planes, 16 )

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x[0]))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x[0]
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.cbam({0:out,1:x[1],2:x[2]})
        out[0] += shortcut
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
        self.cbam = RCBAM(in_planes,planes*self.expansion, 16)

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
        out = self.cbam({0:out,1:x[1],2:x[2]})
        out[0] += shortcut
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
        out ={0:out,1:None,2:None}
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out[0], 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def RCBAMResNet18(num_classes=1000):
    return PreActResNet(PreActBlock, [2,2,2,2],num_classes)

def RCBAMResNet34(num_classes=1000):
    return PreActResNet(PreActBlock, [3,4,6,3],num_classes)

def RCBAMResNet50(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,4,6,3],num_classes)

def RCBAMResNet101(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,4,23,3],num_classes)

def RCBAMResNet152(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,8,36,3],num_classes)


def test():
    net = RCBAMResNet50(num_classes=100)
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()
