from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst

from utils import AverageMeter, accuracy, transform_time
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB
from network import define_tsnet
from kd_losses import *
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser(description='train kd')

# various path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')
parser.add_argument('--s_init', type=str, required=True, help='initial parameters of student model')
parser.add_argument('--t_model', type=str, required=True, help='path name of teacher model')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--optimizer', type=str, default='SGD', help='The type of optimizer')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')

# net and dataset choosen
parser.add_argument('--data_name', type=str, required=True, help='name of dataset') # cifar10/cifar100
parser.add_argument('--t_name', type=str, required=True, help='name of teacher')    # resnet20/resnet110
parser.add_argument('--s_name', type=str, required=True, help='name of student')    # resnet20/resnet110
parser.add_argument('--t_type', type=str, required=True, help='type of teacher')    # resnet20/resnet110
parser.add_argument('--s_type', type=str, required=True, help='type of student')    # resnet20/resnet110
parser.add_argument('--t_ch', type=int, default=64, help='channel  of teacher')    # resnet20/resnet110
parser.add_argument('--s_ch', type=int, default=64, help='channel of student')    # resnet20/resnet110

# hyperparameter
parser.add_argument('--kd_mode', type=str, required=True, help='mode of kd, which can be:'
                                                               'logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/'
                                                               'sp/sobolev/cc/lwm/irg/vid/ofd/afd')
parser.add_argument('--lambda_kd', type=float, default=1.0, help='trade-off parameter for kd loss')
parser.add_argument('--T', type=float, default=4.0, help='temperature for ST')
parser.add_argument('--p', type=float, default=2.0, help='power for AT')
parser.add_argument('--w_dist', type=float, default=25.0, help='weight for RKD distance')
parser.add_argument('--w_angle', type=float, default=50.0, help='weight for RKD angle')
parser.add_argument('--m', type=float, default=2.0, help='margin for AB')
parser.add_argument('--gamma', type=float, default=0.4, help='gamma in Gaussian RBF for CC')
parser.add_argument('--P_order', type=int, default=2, help='P-order Taylor series of Gaussian RBF for CC')
parser.add_argument('--w_irg_vert', type=float, default=0.1, help='weight for IRG vertex')
parser.add_argument('--w_irg_edge', type=float, default=5.0, help='weight for IRG edge')
parser.add_argument('--w_irg_tran', type=float, default=5.0, help='weight for IRG transformation')
parser.add_argument('--sf', type=float, default=1.0, help='scale factor for VID, i.e. mid_channels = sf * out_channels')
parser.add_argument('--init_var', type=float, default=5.0, help='initial variance for VID')
parser.add_argument('--att_f', type=float, default=1.0, help='attention factor of mid_channels for AFD')


args, unparsed = parser.parse_known_args()

args.save_root = os.path.join(f'results/kd/{args.s_name}', args.note)
# args.save_root = os.path.join(f'/root/autodl-tmp/results/kd_{args.s_name}', args.note)
create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    if args.cuda > 1:
        # distubition initialization
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        print('local rank: {}'.format(local_rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0

    logging.info('----------- Network Initialization --------------')
    snet, _ = define_tsnet(name=args.s_name, num_class=args.num_class, net_type=args.s_type, flag='s', kd_ch=(args.s_ch, args.t_ch), first_ch=args.s_ch, cuda=args.cuda)
    checkpoint = torch.load(args.s_init)
    load_pretrained_model(snet, checkpoint['net'])
    if local_rank == 0:
        logging.info('Student: %s', snet)
        logging.info('Student param size = %fMB', count_parameters_in_MB(snet))

    tnet, _ = define_tsnet(name=args.t_name, num_class=args.num_class, net_type=args.t_type, first_ch=args.t_ch, cuda=args.cuda)
    checkpoint = torch.load(args.t_model)
    if 'net' in checkpoint:
        load_pretrained_model(tnet, checkpoint['net'])
    else:
        load_pretrained_model(tnet, checkpoint)
    tnet.eval()
    for param in tnet.parameters():
        param.requires_grad = False
    if local_rank == 0:
        logging.info('Teacher: %s', tnet)
        logging.info('Teacher param size = %fMB', count_parameters_in_MB(tnet))
        logging.info('-----------------------------------------------')

    # define loss functions
    if args.kd_mode == 'logits':
        criterionKD = Logits()
        criterionKD2 = Logits()

    elif args.kd_mode == 'at':
        criterionKD = AT(args.p)
        criterionKD2 = Logits()

    else:
        raise Exception('Invalid kd mode...')
    if args.cuda:
        criterionKD = criterionKD.cuda()
        criterionKD2 = criterionKD2.cuda()
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = torch.nn.CrossEntropyLoss()

    # initialize optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(snet.parameters(),
                                lr = args.lr, 
                                momentum = args.momentum, 
                                weight_decay = args.weight_decay,
                                nesterov = True)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(snet.parameters(),
                                lr = args.lr,
                                amsgrad = True, 
                                weight_decay = args.weight_decay,
                                )

     # define transforms
    if args.data_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        # std  = (0.2470, 0.2435, 0.2616)
        std = (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])
        test_transform = transforms.Compose([
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std)
        ])
        train_dataset   = dst.CIFAR10(root='./cifar_data', train=True, download=True, transform=train_transform)
        test_dataset    = dst.CIFAR10(root='./cifar_data', train=False, download=True, transform=test_transform)
    elif args.data_name == 'imagenet':
        traindir    = os.path.join('/home/ubuntu/imagenet', 'train')
        valdir      = os.path.join('/home/ubuntu/imagenet', 'val')
        normalize   = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_dataset    = dst.ImageFolder(
                                traindir,
                                transforms.Compose([
                                    # new data augmentation
                                    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))
        test_dataset     = dst.ImageFolder(
                            valdir,
                            transforms.Compose([
                                transforms.Resize(224+32), # 256
                                transforms.CenterCrop(224), # 224
                                transforms.ToTensor(),
                                normalize,
                            ]))
    else:
        raise Exception('Invalid dataset name...') 

    # define data loader
    if args.cuda == 1:
        train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True)
        test_loader     = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=2, shuffle=False)
    else:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        # num_workers actually need to be set accroding to the cpu
        train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=2*args.cuda, sampler=train_sampler, pin_memory=True)
        test_loader     = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=2*args.cuda, sampler=test_sampler, pin_memory=True)

    # first initilizing the student nets

    best_top1 = 0
    best_top5 = 0
    if args.cuda > 1:
        snet = snet.cuda()
        tnet = tnet.cuda()
        snet = torch.nn.parallel.DistributedDataParallel(snet)
    else:
        snet = torch.nn.DataParallel(snet).cuda()
        tnet = torch.nn.DataParallel(tnet).cuda()
    # warp nets and criterions for train and test
    nets = {'snet':snet, 'tnet':tnet}
    criterions = {'criterionCls':criterionCls, 'criterionKD':criterionKD, 'criterionKD2':criterionKD2}

    for epoch in range(1, args.epochs+1):
        adjust_lr(optimizer, epoch)

        # train one epoch
        epoch_start_time = time.time()
        log_str = train(train_loader, nets, optimizer, criterions, epoch)

        # evaluate on testing set
        # logging.info('Testing the models......')
        if local_rank == 0:
            logging.info(log_str)
            test_top1, test_top5 = test(test_loader, nets, criterions, epoch)

            epoch_duration = time.time() - epoch_start_time
            logging.info('Epoch time: {}s'.format(int(epoch_duration)))
            # save model
            is_best = False
            if test_top1 > best_top1:
                best_top1 = test_top1
                best_top5 = test_top5
                is_best = True
                logging.info('Saving models, the best accuracy is {} ......'.format(best_top1))
                save_checkpoint({
                    'epoch': epoch,
                    'snet': snet.module.state_dict(),
                    'tnet': tnet.state_dict(),
                    'prec@1': test_top1,
                    'prec@5': test_top5,
                }, is_best, args.save_root)
    if local_rank == 0:        
        logging.info('the best accuracy: top1: {}, top5: {}, saving in: {}'.format(best_top1, best_top5, args.save_root))



def train(train_loader, nets, optimizer, criterions, epoch):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    cls_losses = AverageMeter()
    kd_losses  = AverageMeter()
    kd1_losses  = AverageMeter()
    kd2_losses  = AverageMeter()
    act_losses = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionKD  = criterions['criterionKD']
    criterionKD2  = criterions['criterionKD2']

    snet.train()
    if args.kd_mode in ['vid', 'ofd']:
        for i in range(1,4):
            criterionKD[i].train()

    end = time.time()
    for i, (img, target) in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)

        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        if args.kd_mode in ['sobolev', 'lwm']:
            img.requires_grad = True

        stem_s, out_s, act_out = snet(img)
        stem_t, out_t, _ = tnet(img)

        cls_loss = criterionCls(out_s, target)
        # print('kd_mode: {}, bool: {}, bool: {}'.format(args.kd_mode, (args.kd_mode == 'logits'), args.kd_mode in ['logits', 'st']))
        if args.kd_mode in ['logits', 'st']:
            # kd_loss = criterionKD(out_s, out_t.detach()) * args.lambda_kd
            if not isinstance(stem_s, list):
                kd_loss = (criterionKD(out_s, out_t.detach()) + criterionKD(stem_s, stem_t.detach()))/2.0 * args.lambda_kd
            else:
                out_kd_loss = criterionKD(out_s, out_t.detach())
                stem_kd_loss = []
                for j in range(len(stem_s)):
                    stem_kd_loss.append(criterionKD(stem_s[j], stem_t[j].detach()))
                print('out_kd_loss: {}, stem_kd_loss: {}'.format(out_kd_loss, stem_kd_loss))
                kd_loss = (out_kd_loss + sum(stem_kd_loss)/len(stem_kd_loss))/2.0 * args.lambda_kd

        elif args.kd_mode in ['at', 'sp']:
            kd_loss1 = criterionKD(stem_s, stem_t.detach()) * args.lambda_kd
            kd_loss2 = criterionKD2(out_s, out_t.detach()) * 0.1
            kd_loss = (kd_loss1 + kd_loss2) / 2.0
        else:
            raise Exception(f'Invalid kd mode...{args.kd_mode}')
        act_loss = act_out*0
        loss = cls_loss + kd_loss + act_loss

        prec1, prec5 = accuracy(out_s, target, topk=(1,5))
        cls_losses.update(cls_loss.item(), img.size(0))
        kd_losses.update(kd_loss.item(), img.size(0))
        if args.kd_mode == 'at':
            kd1_losses.update(kd_loss1.item(), img.size(0))
            kd2_losses.update(kd_loss2.item(), img.size(0))
        act_losses.update(act_loss, img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    log_str = ('Epoch[{0}]: '
                'Cls:{cls_losses.avg:.4f}  '
                'KD:{kd_losses.avg:.4f}  '
                'act+loss:{act_losses.avg:.4f}'
                'prec@1:{top1.avg:.2f}  '
                'prec@5:{top5.avg:.2f}'.format(
                epoch, cls_losses=cls_losses, kd_losses=kd_losses, act_losses=act_losses, top1=top1, top5=top5))
    if args.kd_mode == 'at':
        log_str = ('Epoch[{0}]: '
                'Cls:{cls_losses.avg:.4f}  '
                'KD1:{kd1_losses.avg:.4f}  '
                'KD2:{kd2_losses.avg:.4f}  '
                'KD:{kd_losses.avg:.4f}  '
                'act_loss:{act_losses.avg:.4f}'
                'prec@1:{top1.avg:.2f}  '
                'prec@5:{top5.avg:.2f}'.format(
                epoch, cls_losses=cls_losses, kd1_losses=kd1_losses, kd2_losses=kd2_losses, kd_losses=kd_losses, act_losses=act_losses, top1=top1, top5=top5))
    return log_str


def test(test_loader, nets, criterions, epoch):
    cls_losses = AverageMeter()
    kd_losses  = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionKD  = criterions['criterionKD']
    criterionKD2  = criterions['criterionKD2']

    snet.eval()
    if args.kd_mode in ['vid', 'ofd']:
        for i in range(1,4):
            criterionKD[i].eval()

    end = time.time()
    for i, (img, target) in enumerate(test_loader, start=1):
        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        with torch.no_grad():
            stem_s, out_s, _ = snet(img)
            stem_t, out_t, _ = tnet(img)

        cls_loss = criterionCls(out_s, target)
        if args.kd_mode in ['logits', 'st']:
            kd_loss  = criterionKD(out_s, out_t.detach()) * args.lambda_kd
        elif args.kd_mode in ['at', 'sp']:
            kd_loss1 = criterionKD(stem_s, stem_t.detach()) * args.lambda_kd
            kd_loss2 = criterionKD2(out_s, out_t.detach()) * 0.1
            kd_loss = (kd_loss1 + kd_loss2) / 2.0
        else:
            raise Exception('Invalid kd mode...')

        prec1, prec5 = accuracy(out_s, target, topk=(1,5))
        cls_losses.update(cls_loss.item(), img.size(0))
        kd_losses.update(kd_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    f_l = [cls_losses.avg, kd_losses.avg, top1.avg, top5.avg]
    logging.info('Testing: Cls: {:.4f}, KD: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

    return top1.avg, top5.avg


def adjust_lr_init(optimizer, epoch):
    scale   = 0.1
    lr_list = [args.lr*scale] * 30
    lr_list += [args.lr*scale*scale] * 10
    lr_list += [args.lr*scale*scale*scale] * 10

    lr = lr_list[epoch-1]
    logging.info('Epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lr(optimizer, epoch):
    # scale   = 0.1
    # lr_list =  [args.lr] * 100
    # lr_list += [args.lr*scale] * 50
    # lr_list += [args.lr*scale*scale] * 50
    lr_interval = [360, 120, 60, 60] if args.epochs == 600 else [30, 15, 8, 7]
    scale   = 0.2
    lr_list =  [args.lr] * lr_interval[0]
    lr_list += [args.lr*scale] * lr_interval[1]
    lr_list += [args.lr*scale*scale] * lr_interval[2]
    lr_list += [args.lr*scale*scale*scale] * lr_interval[3]

    lr = lr_list[epoch-1]
    logging.info('Epoch: {}  lr: {:.1e}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()