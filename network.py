from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from utils import HoyerBiAct, customConv2
from resnet import resnet20, resnet20_s, BasicBlock
import torch.nn.functional as F



def define_tsnet(name, num_class, net_type='ori', first_ch=64, flag='t', kd_ch=(16, 64), cuda=True, pretrained=None, resume=None):
    start_epoch = 1
    if flag == 't':
        if name == 'resnet20' or name == 'resnet18':
            net = resnet20(num_class=num_class, net_type=net_type, first_ch=first_ch)

        elif name == 'vgg16':
            net = spike_vgg16(num_class=num_class, net_type=net_type, first_ch=first_ch)
        else:
            raise Exception('model name does not exist.')
    elif flag == 's':
        if name == 'resnet20' or name == 'resnet18':
            net = resnet20_s(num_class=num_class, net_type=net_type, first_ch=first_ch, kd_ch=kd_ch)

        elif name == 'vgg16':
            net = spike_vgg16_s(num_class=num_class, net_type=net_type, first_ch=first_ch, kd_ch=kd_ch)
        else:
            raise Exception('model name does not exist.')
    # if cuda:
    #     net = torch.nn.DataParallel(net).cuda()
    # else:
    #     net = torch.nn.DataParallel(net)
    state = torch.load(pretrained, map_location='cpu') if pretrained else None
    if not pretrained:
        return net, start_epoch
    if 'state_dict' in state:
        state = state['state_dict']
        missing_keys, unexpected_keys = net.load_state_dict(state, strict=False)
        print('\n Missing keys : {}\n Unexpected Keys: {}'.format(missing_keys, unexpected_keys))  
    elif pretrained and 'net' in state:
        start_epoch = state['epoch'] if resume else 1
        missing_keys, unexpected_keys = net.load_state_dict(state['net'], strict=False)
        print('\n Missing keys : {}\n Unexpected Keys: {}\n best accuracy: {}'.format(missing_keys, unexpected_keys, state['prec@1']))  
    elif pretrained and 'snet' in state:
        start_epoch = state['epoch'] if resume else 1
        missing_keys, unexpected_keys = net.load_state_dict(state['snet'], strict=False)
        print('\n Missing keys : {}\n Unexpected Keys: {}\n best accuracy: {}'.format(missing_keys, unexpected_keys, state['prec@1']))

    # if cuda:
    #     net = torch.nn.DataParallel(net).cuda()
    # else:
    #     net = torch.nn.DataParallel(net)
    
    # if pretrained and 'net' in state:
    #     missing_keys, unexpected_keys = net.load_state_dict(state['net'], strict=False)
    #     print('\n Missing keys : {}\n Unexpected Keys: {}\n best accuracy: {}'.format(missing_keys, unexpected_keys, state['prec@1']))  

    return net, start_epoch


def resnet20(pretrained=False, **kwargs):
    """Constructs a BiRealNet-20 model. """
    model = HoyerResNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model

def resnet20_s(pretrained=False, **kwargs):
    """Constructs a BiRealNet-20 model. """
    model = HoyerResNet_S(BasicBlock, [4, 4, 4, 4], **kwargs)
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

class spike_vgg16_s(nn.Module):
    def __init__(self, num_class=10, net_type='ori', first_ch=64, kd_ch=(16, 64), linear_dropout=0.1, conv_dropout=0.1, loss_type='sum', im_size=224, spike_type='cw'):
        super(spike_vgg16_s, self).__init__()
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

        self.kd_conv = nn.Conv2d(kd_ch[0], kd_ch[1], kernel_size=1, stride=1, bias=False)
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
                stem_out = self.kd_conv(F.relu((out.clone()))
)
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
