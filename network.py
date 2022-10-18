from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from utils import HoyerBiAct, customConv2
import torch.nn.functional as F



def define_tsnet(name, num_class, net_type='ori', first_ch=64, cuda=True, pretrained=None, resume=None):
    start_epoch = 1
    if name == 'resnet20':
        net = resnet20(num_class=num_class, net_type=net_type, first_ch=first_ch)
    elif name == 'resnet18':
        net = resnet18(num_class=num_class, net_type=net_type, first_ch=first_ch)
    elif name == 'vgg16':
        net = spike_vgg16(num_class=num_class, net_type=net_type, first_ch=first_ch)
    else:
        raise Exception('model name does not exist.')

    state = torch.load(pretrained, map_location='cpu') if pretrained else None
    if pretrained and 'state_dict' in state:
        state = state['state_dict']
        missing_keys, unexpected_keys = net.load_state_dict(state, strict=False)
        print('\n Missing keys : {}\n Unexpected Keys: {}'.format(missing_keys, unexpected_keys))  
    if pretrained and 'net' in state:
        start_epoch = state['epoch'] if resume else 1
        missing_keys, unexpected_keys = net.load_state_dict(state['net'], strict=False)
        print('\n Missing keys : {}\n Unexpected Keys: {}\n best accuracy: {}'.format(missing_keys, unexpected_keys, state['prec@1']))  

    # if cuda:
    #     net = torch.nn.DataParallel(net).cuda()
    # else:
    #     net = torch.nn.DataParallel(net)
    
    # if pretrained and 'net' in state:
    #     missing_keys, unexpected_keys = net.load_state_dict(state['net'], strict=False)
    #     print('\n Missing keys : {}\n Unexpected Keys: {}\n best accuracy: {}'.format(missing_keys, unexpected_keys, state['prec@1']))  

    return net, start_epoch

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
    def __init__(self, block, num_blocks, num_class=10, net_type='ori', first_ch=64, loss_type='sum', spike_type = 'cw', start_spike_layer=0, x_thr_scale=1.0):
        
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
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x) 
        stem_out = x.clone()
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

def resnet20(pretrained=False, **kwargs):
    """Constructs a BiRealNet-20 model. """
    model = HoyerResNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model

def resnet18(pretrained=False, **kwargs):
    """Constructs a BiRealNet-20 model. """
    model = HoyerResNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}
class spike_vgg16(nn.Module):
    def __init__(self, num_class=10, net_type='ori', first_ch=64, linear_dropout=0.1, conv_dropout=0.1, loss_type='sum', im_size=224, spike_type='cw'):
        super(spike_vgg16, self).__init__()
        self.if_spike = True
        self.conv_dropout = conv_dropout
        self.num_class = num_class
        self.loss_type = loss_type
        fc_spike_type = 'fixed' if spike_type == 'fixed' else 'sum'
        self.x_thr_scale = 1.0
        self.spike_type = spike_type
        self.net_type = net_type
        self.first_ch = first_ch
        self.features = self._make_layers(cfg['VGG16'])

        if num_class == 1000: # if data_name=='IMAGENET':
            self.classifier = nn.Sequential(
                        nn.Linear((im_size//32)**2*512, 4096, bias=False),
                        HoyerBiAct(spike_type=fc_spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                        nn.Dropout(linear_dropout),
                        nn.Linear(4096, 4096, bias=False),
                        HoyerBiAct(spike_type=fc_spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                        nn.Dropout(linear_dropout),
                        nn.Linear(4096, num_class, bias=False)
            )
        elif num_class == 10: # elif data_name=='CIFAR10':
            self.classifier = nn.Sequential(
                            nn.Linear(2048, 4096, bias=False),
                            HoyerBiAct(spike_type=fc_spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                            nn.Dropout(linear_dropout),
                            nn.Linear(4096, 4096, bias=False),
                            HoyerBiAct(spike_type=fc_spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                            nn.Dropout(linear_dropout),
                            nn.Linear(4096, num_class, bias=False))
        # self._initialize_weights2()
                            
    def hoyer_loss(self, x):
        # return torch.sum(x)
        x[x<0.0] = 0
        # x[x>thr] = 0
        if torch.sum(torch.abs(x))>0: #  and l < self.start_spike_layer
            # return  (torch.sum(torch.abs(x))**2 / torch.sum((x)**2))
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
        act_loss = 0.0
        first_layer = True
        out = x
        for l in self.features:
            out = l(out)
            if first_layer and isinstance(l, nn.BatchNorm2d):
                first_layer = False
                stem_out = out.clone()

            if isinstance(l, HoyerBiAct):
                act_loss += self.hoyer_loss(out.clone())
                
            # out = l(out)
        
        out = out.view(out.size(0), -1)
        
        for i,l in enumerate(self.classifier):
            out = l(out)
            if isinstance(l, HoyerBiAct):
                act_loss += self.hoyer_loss(out.clone())
            # out = l(out)
 
        return stem_out, out, act_loss

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        if self.num_class == 1000:
            cfg.append('M')
        for i,x in enumerate(cfg):
            
            if x == 'M':
                continue
            if i == 0 and self.net_type=='cus':
                x = self.first_ch
                conv = customConv2(in_channels=3, out_channels=x, kernel_size=(3 ,3), stride = 1, padding = 1)
                # conv = customConv2(in_channels=3, out_channels=16, kernel_size=(7, 7), stride = 6, padding = 1)
            else:
                conv = nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=1, bias=False)

            if i+1 < len(cfg) and cfg[i+1] == 'M':
                layers += [
                        conv,
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.BatchNorm2d(x),
                        HoyerBiAct(num_features=x, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                        nn.Dropout(self.conv_dropout)]
                # layers += [
                #         conv,
                #         nn.BatchNorm2d(x),
                #         HoyerBiAct(num_features=x, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                #         nn.Dropout(self.conv_dropout),
                #         nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                        conv,
                        nn.BatchNorm2d(x),
                        HoyerBiAct(num_features=x, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                        nn.Dropout(self.conv_dropout)]
            in_channels = x
        return nn.Sequential(*layers)

def define_paraphraser(in_channels_t, k, use_bn, cuda=True):
    net = paraphraser(in_channels_t, k, use_bn)
    if cuda:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = torch.nn.DataParallel(net)

    return net


class paraphraser(nn.Module):
    def __init__(self, in_channels_t, k, use_bn=True):
        super(paraphraser, self).__init__()
        factor_channels = int(in_channels_t*k)
        self.encoder = nn.Sequential(*[
                nn.Conv2d(in_channels_t, in_channels_t, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels_t, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
            ])
        self.decoder = nn.Sequential(*[
                nn.ConvTranspose2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
                nn.ConvTranspose2d(factor_channels, in_channels_t, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
                nn.ConvTranspose2d(in_channels_t, in_channels_t, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
            ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z   = self.encoder(x)
        out = self.decoder(z)
        return z, out


def define_translator(in_channels_s, in_channels_t, k, use_bn=True, cuda=True):
    net = translator(in_channels_s, in_channels_t, k, use_bn)
    if cuda:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = torch.nn.DataParallel(net)

    return net


class translator(nn.Module):
    def __init__(self, in_channels_s, in_channels_t, k, use_bn=True):
        super(translator, self).__init__()
        factor_channels = int(in_channels_t*k)
        self.encoder = nn.Sequential(*[
                nn.Conv2d(in_channels_s, in_channels_s, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(in_channels_s) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels_s, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
            ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z   = self.encoder(x)
        return z
