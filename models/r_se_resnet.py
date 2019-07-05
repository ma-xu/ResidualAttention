'''Pre-activation ResNet in PyTorch.


'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.backends.cudnn as cudnn
import os

__all__ = ['RSEResNet50']

# class MultiPrmSequential(nn.Sequential):
#     def __init__(self, *args):
#         super(MultiPrmSequential, self).__init__(*args)
#
#     def forward(self, *input):
#         for module in self._modules.values():
#             input = module(*input)
#         return input


class RSELayer(nn.Module):
    def __init__(self, in_channel, channel, reduction=16):
        super(RSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        if in_channel != channel:
            self.att_fc = nn.Linear(in_channel,channel,bias=False)

    def forward(self, x, att=0):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        pre_att = self.att_fc(att) if hasattr(self, 'att_fc') else att
        all_att = self.fc(y)
        all_att += pre_att
        y=torch.sigmoid(all_att).view(b, c, 1, 1)


        return x * y.expand_as(x), all_att


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.se = RSELayer(in_channel=in_planes, channel=planes*self.expansion)
        # self.se = SELayer(planes*self.expansion)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self,x,att):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        # Add SE block
        out,att = self.se(out,att)
        out += shortcut
        return out,att


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
        self.se = RSELayer(in_channel=in_planes, channel=planes*self.expansion)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x,att):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Add SE block
        out,att = self.se(out,att)
        out += shortcut
        return out,att


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
        return layers

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
        n,c,_,_ = out.size()
        att = 0
        for layer1_block in self.layer1:
            layer1_block = layer1_block.cuda()
            out, att = layer1_block(out, att)
        for layer2_block in self.layer2:
            layer2_block = layer2_block.cuda()
            out, att = layer2_block(out, att)
        for layer3_block in self.layer3:
            layer3_block = layer3_block.cuda()
            out, att = layer3_block(out, att)
        for layer4_block in self.layer4:
            layer4_block = layer4_block.cuda()
            out, att = layer4_block(out, att)
        #
        # out,att = self.layer1(out,att)
        # out,att = self.layer2(out)
        # out,att = self.layer3(out)
        # out,_ = self.layer4(out,att)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def RSEResNet18(num_classes=1000):
    return PreActResNet(PreActBlock, [2,2,2,2],num_classes)

def RSEResNet34(num_classes=1000):
    return PreActResNet(PreActBlock, [3,4,6,3],num_classes)

def RSEResNet50(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,4,6,3],num_classes)

def RSEResNet101(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,4,23,3],num_classes)

def RSEResNet152(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,8,36,3],num_classes)


# def test():
#
#     net = RSEResNet18(num_classes=100)
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     net = net.to(device)
#     if device == 'cuda':
#         net = torch.nn.DataParallel(net)
#         cudnn.benchmark = True
#     y = net((torch.randn(2,3,32,32)))
#     print(y.size())
#
# test()
