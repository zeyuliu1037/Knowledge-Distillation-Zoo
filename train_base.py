from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np

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

parser = argparse.ArgumentParser(description='train base net')

# various path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--optimizer', type=str, default='SGD', help='The type of optimizer')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')

# net and dataset choosen
parser.add_argument('--data_name', type=str, required=True, help='name of dataset') # cifar10/cifar100
parser.add_argument('--net_name', type=str, required=True, help='name of basenet')  # resnet20/resnet110


args, unparsed = parser.parse_known_args()

args.save_root = os.path.join(args.save_root, args.note)
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
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    logging.info('----------- Network Initialization --------------')
    net = define_tsnet(name=args.net_name, num_class=args.num_class, cuda=args.cuda)
    logging.info('%s', net)
    logging.info("param size = %fMB", count_parameters_in_MB(net))
    logging.info('-----------------------------------------------')

    # save initial parameters
    logging.info('Saving initial parameters......') 
    save_path = os.path.join(args.save_root, 'initial_r{}.pth.tar'.format(args.net_name[6:]))
    torch.save({
        'epoch': 0,
        'net': net.state_dict(),
        'prec@1': 0.0,
        'prec@5': 0.0,
    }, save_path)

    # initialize optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),
                                lr = args.lr, 
                                momentum = args.momentum, 
                                weight_decay = args.weight_decay,
                                nesterov = True)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(),
                                lr = args.lr,
                                amsgrad = True, 
                                weight_decay = args.weight_decay,
                                )

    # define loss functions
    if args.cuda:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # define transforms
    if args.data_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)
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
        train_dataset   = dst.CIFAR10(root='./cifar_data', train=True, download=True, transform =train_transform, pin_memory=True)
        test_dataset    = dst.CIFAR10(root='./cifar_data', train=False, download=True, transform=test_transform, pin_memory=True)
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
    
    train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    test_loader     = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    best_top1 = 0
    best_top5 = 0
    for epoch in range(1, args.epochs+1):
        adjust_lr(optimizer, epoch)

        # train one epoch
        epoch_start_time = time.time()
        train(train_loader, net, optimizer, criterion, epoch)

        # evaluate on testing set
        logging.info('Testing the models......')
        test_top1, test_top5 = test(test_loader, net, criterion)

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
            'net': net.state_dict(),
            'prec@1': test_top1,
            'prec@5': test_top5,
        }, is_best, args.save_root)


def train(train_loader, net, optimizer, criterion, epoch, hoyer_decay=1e-8):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    act_losses   = AverageMeter()
    total_losses = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    net.train()

    end = time.time()
    for i, (img, target) in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)

        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # _, _, _, _, _, out = net(img)
        _, out, act_loss = net(img)
        loss = criterion(out, target)
        act_loss = hoyer_decay*act_loss
        total_loss = loss + act_loss
        # print('out: {}, target: {}'.format(out.shape, target.shape))
        prec1, prec5 = accuracy(out, target, topk=(1,5))
        losses.update(loss.item(), img.size(0))
        act_losses.update(act_loss, img.size(0))
        total_losses.update(total_loss, img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
                       'Time:{batch_time.val:.4f} '
                       'Data:{data_time.val:.4f}  '
                       'loss:{losses.val:.4f}({losses.avg:.4f})  '
                       'act_loss:{act_losses.val:.4f}({act_losses.avg:.4f})  '
                       'total_loss:{total_losses.val:.4f}({total_losses.avg:.4f})  '
                       'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                       'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                       losses=losses, act_losses=act_losses, total_losses=total_losses, top1=top1, top5=top5))
            logging.info(log_str)


def test(test_loader, net, criterion):
    losses = AverageMeter()
    top1   = AverageMeter()
    top5   = AverageMeter()

    net.eval()

    end = time.time()
    for i, (img, target) in enumerate(test_loader, start=1):
        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        with torch.no_grad():
            # _, _, _, _, _, out = net(img)
            _, out, _ = net(img)
            loss = criterion(out, target)

        prec1, prec5 = accuracy(out, target, topk=(1,5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    f_l = [losses.avg, top1.avg, top5.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

    return top1.avg, top5.avg


def adjust_lr(optimizer, epoch):
    #  [360, 480, 540]
    scale   = 0.2
    lr_list =  [args.lr] * 360
    lr_list += [args.lr*scale] * 120
    lr_list += [args.lr*scale*scale] * 60
    lr_list += [args.lr*scale*scale*scale] * 60

    lr = lr_list[epoch-1]
    logging.info('Epoch: {}  lr: {:.1e}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()