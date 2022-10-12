from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from utils import HoyerBiAct, customConv2



def define_tsnet(name, num_class, cuda=True):
    if name == 'resnet20':
        net = resnet20(num_class=num_class)
    elif name == 'resnet110':
        net = resnet110(num_class=num_class)
    elif name == 'vgg16':
        net = spike_vgg16(num_class=num_class)
    else:
        raise Exception('model name does not exist.')

    if cuda:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = torch.nn.DataParallel(net)

    return net


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels, return_before_act):
        super(resblock, self).__init__()
        self.return_before_act = return_before_act
        self.downsample = (in_channels != out_channels)
        if self.downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.ds    = nn.Sequential(*[
                            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(out_channels)
                            ])
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.ds    = None
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        pout = self.conv1(x) # pout: pre out before activation
        pout = self.bn1(pout)
        pout = self.relu(pout)

        pout = self.conv2(pout)
        pout = self.bn2(pout)

        if self.downsample:
            residual = self.ds(x)

        pout += residual
        out  = self.relu(pout)

        if not self.return_before_act:
            return out
        else:
            return pout, out


class resnet20(nn.Module):
    def __init__(self, num_class):
        super(resnet20, self).__init__()
        self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)
        self.relu    = nn.ReLU()

        self.res1 = self.make_layer(resblock, 3, 16, 16)
        self.res2 = self.make_layer(resblock, 3, 16, 32)
        self.res3 = self.make_layer(resblock, 3, 32, 64)

        self.avgpool = nn.AvgPool2d(8)
        self.fc      = nn.Linear(64, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_class = num_class

    def make_layer(self, block, num, in_channels, out_channels): # num must >=2
        layers = [block(in_channels, out_channels, False)]
        for i in range(num-2):
            layers.append(block(out_channels, out_channels, False))
        layers.append(block(out_channels, out_channels, True))
        return nn.Sequential(*layers)

    def forward(self, x):
        pstem = self.conv1(x) # pstem: pre stem before activation
        pstem = self.bn1(pstem)
        stem  = self.relu(pstem)
        stem  = (pstem, stem)

        rb1 = self.res1(stem[1])
        rb2 = self.res2(rb1[1])
        rb3 = self.res3(rb2[1])

        feat = self.avgpool(rb3[1])
        feat = feat.view(feat.size(0), -1)
        out  = self.fc(feat)

        return stem, rb1, rb2, rb3, feat, out

    def get_channel_num(self):
        return [16, 16, 32, 64, 64, self.num_class]

    def get_chw_num(self):
        return [(16, 32, 32),
                (16, 32, 32),
                (32, 16, 16),
                (64, 8 , 8 ),
                (64,),
                (self.num_class,)]


class resnet110(nn.Module):
    def __init__(self, num_class):
        super(resnet110, self).__init__()
        self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)
        self.relu    = nn.ReLU()

        self.res1 = self.make_layer(resblock, 18, 16, 16)
        self.res2 = self.make_layer(resblock, 18, 16, 32)
        self.res3 = self.make_layer(resblock, 18, 32, 64)

        self.avgpool = nn.AvgPool2d(8)
        self.fc      = nn.Linear(64, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_class = num_class

    def make_layer(self, block, num, in_channels, out_channels):  # num must >=2
        layers = [block(in_channels, out_channels, False)]
        for i in range(num-2):
            layers.append(block(out_channels, out_channels, False))
        layers.append(block(out_channels, out_channels, True))
        return nn.Sequential(*layers)

    def forward(self, x):
        pstem = self.conv1(x) # pstem: pre stem before activation
        pstem = self.bn1(pstem)
        stem  = self.relu(pstem)
        stem  = (pstem, stem)

        rb1 = self.res1(stem[1])
        rb2 = self.res2(rb1[1])
        rb3 = self.res3(rb2[1])

        feat = self.avgpool(rb3[1])
        feat = feat.view(feat.size(0), -1)
        out  = self.fc(feat)

        return stem, rb1, rb2, rb3, feat, out

    def get_channel_num(self):
        return [16, 16, 32, 64, 64, self.num_class]

    def get_chw_num(self):
        return [(16, 32, 32),
                (16, 32, 32),
                (32, 16, 16),
                (64, 8 , 8 ),
                (64,),
                (self.num_class,)]
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}
class spike_vgg16(nn.Module):
    def __init__(self, num_class=10, linear_dropout=0.1, conv_dropout=0.1, loss_type='sum', im_size=224, spike_type='cw'):
        super(spike_vgg16, self).__init__()
        self.if_spike = True
        self.conv_dropout = conv_dropout
        self.num_class = num_class
        self.loss_type = loss_type
        fc_spike_type = 'fixed' if spike_type == 'fixed' else 'sum'
        self.x_thr_scale = 1.0
        self.if_spike = True
        self.spike_type = spike_type
        self.features = self._make_layers(cfg['VGG16'])

        if num_class == 1000: # if dataset=='IMAGENET':
            self.classifier = nn.Sequential(
                        nn.Linear((im_size//32)**2*512, 4096, bias=False),
                        HoyerBiAct(spike_type=fc_spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                        nn.Dropout(linear_dropout),
                        nn.Linear(4096, 4096, bias=False),
                        HoyerBiAct(spike_type=fc_spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                        nn.Dropout(linear_dropout),
                        nn.Linear(4096, num_class, bias=False)
            )
        elif num_class == 10: # elif dataset=='CIFAR10':
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
            if isinstance(l, nn.BatchNorm2d) and first_layer:
                first_layer = False
                out_first = out.clone()

            if isinstance(l, HoyerBiAct):
                act_loss += self.hoyer_loss(out.clone())
                
            # out = l(out)
        
        out = out.view(out.size(0), -1)
        
        for i,l in enumerate(self.classifier):
            out = l(out)
            if isinstance(l, HoyerBiAct):
                act_loss += self.hoyer_loss(out.clone())
            # out = l(out)
 
        return out_first, out, act_loss

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        if self.num_class == 1000:
            cfg.append('M')
        for i,x in enumerate(cfg):
            
            if x == 'M':
                continue
            if i == 0:
                x=64
                conv = nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=1, bias=False)
                # conv = customConv2(in_channels=3, out_channels=x, kernel_size=(3 ,3), stride = 1, padding = 1)
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
