'''
Different from normal ResNet, you could get several layers outputs by giving out_indices
from https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/fbresnet/resnet152_load.py
https://blog.csdn.net/frighting_ing/article/details/121324000 
'''

from audioop import bias
from itertools import dropwhile
from sqlite3 import adapt
from turtle import forward
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
import math
import collections
import sys

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



# 2 convolutions residual block, kernel size: 3,3
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 3 convolutions residual block, kernel size: 1, 3, 1
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# TODO add out_indices
class ResNet(nn.Module):

    def __init__(self, block, layers, out_indices=[-1], num_classes=1000, drop_rate=0.0):
        '''
        block: BasicBLock, Bottleneck
        layers: a list recording num of blocks in each stage
        out_indices: output from which stage
        available [0, 1, 2, 3, 4, -1], 0 means after conv1+pool, -1 means vector after fc
        BasicBlock  0:[h/2,w/2,64], 1:[h/4,w/4,64], 2:[h/8,w/8,128], 3:[h/16,w/16,256], 4:[h/32,w/32,512], -1:[1000]
        Bottleneck  0               1:[h/4,w/4,256],           512              1024                 2048   
        '''
        assert max(out_indices) <= 4
        drop_rate = drop_rate
        self.out_indices = out_indices
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        if 2 in out_indices:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        else:
            self.layer2 = nn.Identity()
        if 3 in out_indices:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        else:
            self.layer3 = nn.Identity()
        if 4 in out_indices:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if -1 in out_indices:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if 0 in self.out_indices:
            outs.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if 1 in self.out_indices:
            outs.append(x)
        x = self.layer2(x)
        if 2 in self.out_indices:
            outs.append(x)
        x = self.layer3(x)
        if 3 in self.out_indices:
            outs.append(x)
        if 4 in self.out_indices:
            x = self.layer4(x)
            outs.append(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)


        return outs


def load_pretrain(model, pre_s_dict):
    ''' Load state_dict in pre_model to model
    Solve the problem that model and pre_model have some different keys'''
    s_dict = model.state_dict()
    # remove fc weights and bias
    pre_s_dict.pop('fc.weight')
    pre_s_dict.pop('fc.bias')
    # use new dict to store states, record missing keys
    missing_keys = []
    new_state_dict = collections.OrderedDict()
    for key in s_dict.keys():
        if key in pre_s_dict.keys():
            new_state_dict[key] = pre_s_dict[key]
        else:
            new_state_dict[key] = s_dict[key]
            missing_keys.append(key)
    print('{} keys are not in the pretrain model:'.format(len(missing_keys)), missing_keys)
    # load new s_dict
    model.load_state_dict(new_state_dict)
    return model


# models
def resnet18(pretrained=False,**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pre_s_dict = model_zoo.load_url(model_urls['resnet18'])
        model = load_pretrain(model, pre_s_dict)
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        pre_s_dict = model_zoo.load_url(model_urls['resnet34'])
        model = load_pretrain(model, pre_s_dict)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        pre_s_dict = model_zoo.load_url(model_urls['resnet50'])
        model = load_pretrain(model, pre_s_dict)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        pre_s_dict = model_zoo.load_url(model_urls['resnet101'])
        model = load_pretrain(model, pre_s_dict)
    return model



if __name__=='__main__':
    model = resnet34(pretrained=True, out_indices=[0,1,2,3,4], adapt_method=None)  # 11m
    x = torch.randn(5,3,224,224)
    y = model(x)
    for item in y:
        print(item.shape)
    total_trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad)
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))