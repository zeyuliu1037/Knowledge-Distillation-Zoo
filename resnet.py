import torch
import torch.nn as nn
from utils import HoyerBiAct, customConv2
import torch.nn.functional as F


def conv3x3(inplanes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(inplanes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(inplanes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, spike_type='sum', \
         x_thr_scale=1.0, if_spike=True):
        super(BasicBlock, self).__init__()

        self.act = HoyerBiAct(num_features=inplanes, spike_type=spike_type, x_thr_scale=x_thr_scale, if_spike=if_spike)
        self.conv = conv3x3(inplanes, planes,stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x
        # always spike
        out = self.act(x)
        out = self.conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # print(out.shape, residual.shape)
        out += residual

        return out

class HoyerResNet(nn.Module):
    def __init__(self, block=None, num_blocks=None, num_class=10, net_type='ori', first_ch=64, loss_type='sum', spike_type = 'cw', start_spike_layer=0, x_thr_scale=1.0):
        
        super(HoyerResNet, self).__init__()
        self.inplanes = first_ch
        self.spike_type     = spike_type
        self.loss_type     = loss_type
        self.x_thr_scale    = x_thr_scale
        self.if_spike       = True if start_spike_layer == 0 else False 
        if num_class == 10:
            self.conv1 = nn.Sequential(
                                nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False) if net_type == 'ori' \
                                    else customConv2(in_channels=3, out_channels=self.inplanes, kernel_size=(3 ,3), stride = 1, padding = 1),
                                # customConv2(in_channels=3, out_channels=64, kernel_size=(3 ,3), stride = 1, padding = 1),
                                nn.BatchNorm2d(self.inplanes),
                                HoyerBiAct(num_features=self.inplanes, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),

                                nn.Conv2d(self.inplanes, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),

                                HoyerBiAct(64, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                )
        elif num_class == 1000:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) if net_type == 'ori' \
                else customConv2(in_channels=3, out_channels=self.inplanes, kernel_size=(7 ,7), stride = 2, padding = 3)
        else:
            raise RuntimeError('only for ciafar10 and imagenet now')
        self.bn1 = nn.BatchNorm2d(64)
        self.inplanes = 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc_act = HoyerBiAct(spike_type='sum', x_thr_scale=self.x_thr_scale, if_spike=self.if_spike)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_class)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 1.0 maxpool + bn + spike + conv1x1 for resnet18 with vgg, it is the best
            downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=stride),
                nn.BatchNorm2d(self.inplanes),
                HoyerBiAct(num_features=self.inplanes, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                # nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale))

        return nn.Sequential(*layers)

    def hoyer_loss(self, x):
        x[x<0]=0
        if torch.sum(torch.abs(x))>0: #  and l < self.start_spike_layer
            if self.loss_type == 'mean':
                return torch.mean(torch.sum(torch.abs(x), dim=(1,2,3))**2 / torch.sum((x)**2, dim=(1,2,3)))
            elif self.loss_type == 'sum':
                return  (torch.sum(torch.abs(x))**2 / torch.sum((x)**2))
            elif self.loss_type == 'cw':
                hoyer_thr = torch.sum((x)**2, dim=(0,2,3)) / torch.sum(torch.abs(x), dim=(0,2,3))
                # 1.0 is the max thr
                hoyer_thr = torch.nan_to_num(hoyer_thr, nan=1.0)
                return torch.mean(hoyer_thr)
        return 0.0


    def forward(self, x):
        act_out = 0.0
        # x = self.conv1(x)
        for i,layer in enumerate(self.conv1):
            x = layer(x)
            if i == 1 and isinstance(layer, nn.BatchNorm2d):
                stem_out = F.relu(x.clone())
        x = self.maxpool(x)
        x = self.bn1(x) 
        # stem_out = x.clone()
        act_out += self.hoyer_loss(x.clone())

        for i,layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for l in layers:
                x = l(x)
                act_out += self.hoyer_loss(x.clone())
     
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_act(x)
        x = self.fc(x)

        return stem_out, x, act_out

class HoyerResNet_S(nn.Module):
    def __init__(self, block=None, num_blocks=None, kd_ch=(16, 64), num_class=10, net_type='ori', first_ch=64, loss_type='sum', spike_type = 'cw', start_spike_layer=0, x_thr_scale=1.0):
        
        super(HoyerResNet_S, self).__init__()
        self.inplanes = first_ch
        self.spike_type     = spike_type
        self.loss_type     = loss_type
        self.x_thr_scale    = x_thr_scale
        self.if_spike       = True if start_spike_layer == 0 else False 
        if num_class == 10:
            self.conv1 = nn.Sequential(
                                nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False) if net_type == 'ori' \
                                    else customConv2(in_channels=3, out_channels=self.inplanes, kernel_size=(3 ,3), stride = 1, padding = 1),
                                # customConv2(in_channels=3, out_channels=64, kernel_size=(3 ,3), stride = 1, padding = 1),
                                nn.BatchNorm2d(self.inplanes),
                                HoyerBiAct(num_features=self.inplanes, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),

                                nn.Conv2d(self.inplanes, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),

                                HoyerBiAct(64, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                )
        elif num_class == 1000:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) if net_type == 'ori' \
                else customConv2(in_channels=3, out_channels=self.inplanes, kernel_size=(7 ,7), stride = 2, padding = 3)
        else:
            raise RuntimeError('only for ciafar10 and imagenet now')
        self.kd_conv = nn.Conv2d(kd_ch[0], kd_ch[1], kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.inplanes = 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc_act = HoyerBiAct(spike_type='sum', x_thr_scale=self.x_thr_scale, if_spike=self.if_spike)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_class)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 1.0 maxpool + bn + spike + conv1x1 for resnet18 with vgg, it is the best
            downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=stride),
                nn.BatchNorm2d(self.inplanes),
                HoyerBiAct(num_features=self.inplanes, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                # nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale))

        return nn.Sequential(*layers)

    def hoyer_loss(self, x):
        x[x<0]=0
        if torch.sum(torch.abs(x))>0: #  and l < self.start_spike_layer
            if self.loss_type == 'mean':
                return torch.mean(torch.sum(torch.abs(x), dim=(1,2,3))**2 / torch.sum((x)**2, dim=(1,2,3)))
            elif self.loss_type == 'sum':
                return  (torch.sum(torch.abs(x))**2 / torch.sum((x)**2))
            elif self.loss_type == 'cw':
                hoyer_thr = torch.sum((x)**2, dim=(0,2,3)) / torch.sum(torch.abs(x), dim=(0,2,3))
                # 1.0 is the max thr
                hoyer_thr = torch.nan_to_num(hoyer_thr, nan=1.0)
                return torch.mean(hoyer_thr)
        return 0.0


    def forward(self, x):
        act_out = 0.0
        stem_out = None
        for i,layer in enumerate(self.conv1):
            x = layer(x)
            if i == 1 and isinstance(layer, nn.BatchNorm2d):
                stem_out = self.kd_conv(F.relu(x.clone()))
        # x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x) 
        # stem_out = x.clone()
        act_out += self.hoyer_loss(x.clone())

        for i,layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for l in layers:
                x = l(x)
                act_out += self.hoyer_loss(x.clone())
     
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_act(x)
        x = self.fc(x)

        return stem_out, x, act_out

class HoyerResNet_Multi(HoyerResNet):
    def __init__(self, **kwargs):
        super(HoyerResNet_Multi, self).__init__(**kwargs)
    
    def forward(self, x):
        act_out = 0.0
        stem_out = []
        for i,layer in enumerate(self.conv1):
            x = layer(x)
            if i == 1 and isinstance(layer, nn.BatchNorm2d):
                stem_out.append(F.relu(x.clone()))
        # x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x) 
        # stem_out = x.clone()
        act_out += self.hoyer_loss(x.clone())

        for i,layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for l in layers:
                x = l(x)
                act_out += self.hoyer_loss(x.clone())
            stem_out.append(x.clone())
     
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_act(x)
        x = self.fc(x)

        return stem_out, x, act_out

class HoyerResNet_S_Multi(HoyerResNet_S):
    def __init__(self, **kwargs):
        super(HoyerResNet_S_Multi, self).__init__(**kwargs)
    
    def forward(self, x):
        act_out = 0.0
        stem_out = []
        for i,layer in enumerate(self.conv1):
            x = layer(x)
            if i == 1 and isinstance(layer, nn.BatchNorm2d):
                stem_out.append(self.kd_conv(F.relu(x.clone())))
        # x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x) 
        # stem_out = x.clone()
        act_out += self.hoyer_loss(x.clone())

        for i,layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for l in layers:
                x = l(x)
                act_out += self.hoyer_loss(x.clone())
            stem_out.append(x.clone())
     
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_act(x)
        x = self.fc(x)

        return stem_out, x, act_out

def resnet20(pretrained=False, **kwargs):
    """Constructs a BiRealNet-20 model. """
    model = HoyerResNet(block=BasicBlock, num_blocks=[4, 4, 4, 4], **kwargs)
    return model

def resnet20_s(pretrained=False, **kwargs):
    """Constructs a BiRealNet-20 model. """
    model = HoyerResNet_S(block=BasicBlock, num_blocks=[4, 4, 4, 4], **kwargs)
    return model

def resnet20_multi(pretrained=False, **kwargs):
    """Constructs a BiRealNet-20 model. """
    model = HoyerResNet_Multi(block=BasicBlock, num_blocks=[4, 4, 4, 4], **kwargs)
    return model

def resnet20_s_multi(pretrained=False, **kwargs):
    """Constructs a BiRealNet-20 model. """
    model = HoyerResNet_S_Multi(block=BasicBlock, num_blocks=[4, 4, 4, 4], **kwargs)
    return model