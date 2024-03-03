from __future__ import absolute_import

import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
import mindspore as ms
from msadapter.pytorch.nn import functional as F
from msadapter.pytorch.nn import init
import msadapter.torchvision

from msadapter.pytorch.hub import load_state_dict_from_url
import math

# from ..utils.serialization import load_checkpoint, copy_state_dict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2', 'A_2_net_backbone']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Generator(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        x = self.layer1(x[:,:,None, None])
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


class ResNet_Backbone(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Backbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def A_2_net_backbone(pretrained=True, progress=True, **kwargs):
    model = ResNet_Backbone(Bottleneck, [3, 4, 6], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=progress)
        for name in list(state_dict.keys()):
            if 'fc' in name or 'layer4' in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict)
    return model


class ResNet_Refine(nn.Module):

    def __init__(self, block, layer, is_local=True, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Refine, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 1024
        self.dilation = 1

        self.is_local = is_local
        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self._make_layer(block, 512, layer, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer4(x)

        pool_x = self.avgpool(x)
        pool_x = torch.flatten(pool_x, 1)
        if self.is_local:
            return x, pool_x
        else:
            return pool_x

    def forward(self, x):
        return self._forward_impl(x)


def A_2_net_refine(is_local=True, pretrained=True, progress=True, **kwargs):
    model = ResNet_Refine(Bottleneck, 3, is_local, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=progress)
        for name in list(state_dict.keys()):
            if not 'layer4' in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict)
    return model


class ResNet_Attention(nn.Module):

    def __init__(self, block, layer, att_size=4, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_Attention, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 1024
        self.dilation = 1
        self.att_size = att_size
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer4 = self._make_layer(block, 512, layer, stride=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        att_expansion = 0.25
        layers = []
        layers.append(block(self.inplanes, int(self.inplanes * att_expansion), stride,
                            downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                conv1x1(self.inplanes, int(self.inplanes * att_expansion)),
                nn.BatchNorm2d(int(self.inplanes * att_expansion))
            ))
            self.inplanes = int(self.inplanes * att_expansion)
            layers.append(block(self.inplanes, int(self.inplanes * att_expansion), groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        layers.append(nn.Sequential(
            conv1x1(self.inplanes, self.att_size),
            nn.BatchNorm2d(self.att_size),
            nn.ReLU(inplace=True)
        ))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer4(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def A_2_net_attention(att_size=4, pretrained=False, progress=True, **kwargs):
    model = ResNet_Attention(Bottleneck, 3, att_size=att_size, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=progress)
        for name in list(state_dict.keys()):
            if 'fc' in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict)
    return model


class A_2_net(nn.Module):
    def __init__(self, code_length=12, num_classes=200, att_size=4, feat_size=2048, device='cpu', pretrained=True):
        super(A_2_net, self).__init__()
        self.backbone = A_2_net_backbone(pretrained=pretrained)
        self.refine_global = A_2_net_refine(is_local=True, pretrained=pretrained)
        self.refine_local = A_2_net_refine(pretrained=pretrained)
        self.attention = A_2_net_attention(att_size)

        self.hash_layer_active = nn.Sequential(
            nn.Tanh(),
        )
        self.W = nn.Parameter(torch.Tensor(code_length, (att_size + 1) * feat_size))
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.encoder = nn.Linear((att_size + 1) * feat_size, 512)
        self.netG = Generator(512)
        self.encoder2 = nn.Linear((att_size + 1) * feat_size, 512)
        self.netG2 = Generator(512)
        self.device = device

    def forward(self, x, targets=None):
        out = self.backbone(x)#.detach()
        batch_size, channels, h, w = out.shape
        feat, global_f = self.refine_global(out)

        '''atts = feat.detach()
        for i in range(targets.size(0)):
            show_image(x[i], atts[i], i)'''

        att_map = self.attention(out)
        att_size = att_map.shape[1]
        att_map_rep = att_map.unsqueeze(axis=2).repeat(1, 1, channels, 1, 1)
        out_rep = out.unsqueeze(axis=1).repeat(1, att_size, 1, 1, 1)
        out_local = att_map_rep.mul(out_rep).view(batch_size * att_size, channels, h, w)
        local_f, avg_local_f = self.refine_local(out_local)
        _, channels, h, w = local_f.shape
        local_f = local_f.view(batch_size, att_size, channels, h, w)
        avg_local_f = avg_local_f.view(batch_size, att_size, channels)

        all_f = torch.cat([avg_local_f.view(batch_size, -1), global_f], dim=1)
        deep_S = F.linear(all_f, self.W)
        ret = self.hash_layer_active(deep_S)
        dret = F.linear(ret, self.W.t())
        
        if not self.training:
            return ret
        
        _img = self.netG(self.encoder(all_f))
        _img2 = self.netG2(self.encoder2(dret))

        return ret, dret, all_f, deep_S, {'_img': _img, '_img2': _img2}


import cv2
import matplotlib.pyplot as plt
@torch.no_grad()
def show_image(img, atts, i):
    mean, std = [0.466, 0.471, 0.380], [0.195, 0.194, 0.192]
    mean, std = torch.tensor(mean).view(1, 1, 3), torch.tensor(std).view(1, 1, 3)
    img = img.permute(1, 2, 0).cpu()
    img = (img * std + mean).numpy()#[:, :, ::-1]

    plt.subplot(2, 2, 1)
    plt.title(f'{i}')
    plt.axis('off')
    plt.imshow(img)

    att = atts[:].mean(0,True)
    att = att.permute(1, 2, 0).cpu()
    att = att.repeat(1, 1, 3).numpy()
    plt.subplot(2, 2, 2)
    plt.axis('off')
    att=att**0.5
    att=(att-att.min())/(att.max()-att.min())
    plt.imshow(att)

    att = atts[:2].mean(0,True)
    att = att.permute(1, 2, 0).cpu()
    att = att.repeat(1, 1, 3).numpy()
    plt.subplot(2, 2, 3)
    plt.axis('off')
    att=att**0.8
    att=(att-att.min())/(att.max()-att.min())
    plt.imshow(att)

    att = atts[-2:].mean(0,True)
    att = att.permute(1, 2, 0).cpu()
    att = att.repeat(1, 1, 3).numpy()
    plt.subplot(2, 2, 4)
    plt.axis('off')
    att=(att-att.min())/(att.max()-att.min())
    plt.imshow(att)


    #plt.draw()
    #plt.savefig(name)
    plt.show()
    plt.close()



def a_2_net(code_length, num_classes, att_size, feat_size, device, pretrained=False, progress=True, **kwargs):
    model = A_2_net(code_length, num_classes, att_size, feat_size, device, pretrained, **kwargs)
    return model


if __name__ == "__main__":
    import time
    device = torch.device('cuda')
    model = a_2_net(512, 101, 12, 2048, device).to(device)
    img = torch.rand(1, 3, 224, 224, device=device)
    for _ in range(5):
        output = model(img)
    start = time.time()
    for _ in range(100):
        output = model(img)
    end = time.time()
    print(end-start)
    