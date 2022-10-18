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
from torch.utils.data.distributed import DistributedSampler

from utils import AverageMeter, accuracy, transform_time
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB
from network import define_tsnet

parser = argparse.ArgumentParser(description='train base net')

# various path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')

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

parser.add_argument('--net_type', type=str, default='ori')
parser.add_argument('--first_ch', type=int, default=64)
# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--test_only', action='store_true', help='perform only inference')
parser.add_argument('--resume', action='store_true', help='perform only inference')
parser.add_argument('--pretrained', default='', type=str, help='pretrained model to initialize ANN')

# net and dataset choosen
parser.add_argument('--data_name', type=str, required=True, help='name of dataset') # cifar10/cifar100
parser.add_argument('--net_name', type=str, required=True, help='name of basenet')  # resnet20/resnet110


args, unparsed = parser.parse_known_args()

args.save_root = os.path.join(f'results/{args.net_name}', args.note)
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
    net, start_epoch = define_tsnet(name=args.net_name, num_class=args.num_class, net_type=args.net_type, first_ch=args.first_ch, \
                        cuda=args.cuda, pretrained=args.pretrained, resume=args.resume)
    if local_rank == 0:
        logging.info('%s', net)
        logging.info("param size = %fMB", count_parameters_in_MB(net))
        logging.info('-----------------------------------------------')

    if not args.test_only and local_rank == 0:
        # save initial parameters
        logging.info('Saving initial parameters......') 
        save_path = os.path.join(args.save_root, 'initial_r{}.pth.tar'.format(args.net_name[6:]))
        torch.save({
            'epoch': 0,
            'net': net.state_dict(),
            'prec@1': 0.0,
            'prec@5': 0.0,
        }, save_path)
        

    # define loss functions
    if args.cuda:
        net = net.cuda()
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
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

    best_top1 = 0
    best_top5 = 0
    if args.cuda == 1:
        net = net.cuda()
    else:
        net = net.cuda()
        net = torch.nn.parallel.DistributedDataParallel(net)
    for epoch in range(start_epoch, args.epochs+1):
        adjust_lr(optimizer, epoch)

        # train one epoch
        epoch_start_time = time.time()
        if not args.test_only:
            log_str = train(train_loader, net, optimizer, criterion, epoch)

        # evaluate on testing set
        # logging.info('Testing the models......')
        if local_rank == 0:
            logging.info(log_str)
            test_top1, test_top5 = test(test_loader, net, criterion)
            epoch_duration = time.time() - epoch_start_time
            logging.info('Epoch time: {}s'.format(int(epoch_duration)))
            if args.test_only:
                break
            # save model
            is_best = False
            if test_top1 > best_top1:
                best_top1 = test_top1
                best_top5 = test_top5
                is_best = True
                logging.info('Saving models, the best accuracy is {} ......'.format(best_top1))
                save_checkpoint({
                    'epoch': epoch,
                    'net': net.state_dict() if args.cuda == 1 else net.module.state_dict(),
                    'prec@1': test_top1,
                    'prec@5': test_top5,
                }, is_best, args.save_root)
    if local_rank == 0:
        logging.info('the best accuracy: top1: {}, top5: {}, saving in: {}'.format(best_top1, best_top5, args.save_root))
        


def train(train_loader, net, optimizer, criterion, epoch, hoyer_decay=1e-8):
    losses     = AverageMeter()
    act_losses   = AverageMeter()
    total_losses = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    net.train()

    for i, (img, target) in enumerate(train_loader, start=1):

        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # _, _, _, _, _, out = net(img)
        _, out, act_out = net(img)
        loss = criterion(out, target)
        act_loss = hoyer_decay*act_out
        # print('loss: {}, act_loss: {}'.format(loss, act_loss))
        total_loss = loss + act_loss 
        # print('out: {}, target: {}'.format(out.shape, target.shape))
        # print('loss: ', total_loss)
        prec1, prec5 = accuracy(out, target, topk=(1,5))
        losses.update(loss.item(), img.size(0))
        act_losses.update(act_loss, img.size(0))
        total_losses.update(total_loss, img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    log_str = ('Epoch[{0}]:  '
                'loss:{losses.avg:.4f}  '
                'act_loss:{act_losses.avg:.4f}  '
                'total_loss:{total_losses.avg:.4f}  '
                'prec@1:{top1.avg:.2f}  '
                'prec@5:{top5.avg:.2f} '.format(
                epoch, losses=losses, act_losses=act_losses, total_losses=total_losses, top1=top1, top5=top5))
    return log_str


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
    logging.info('Testing: Loss: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

    return top1.avg, top5.avg


def adjust_lr(optimizer, epoch):
    #  [360, 480, 540]
    if args.epochs == 600:
        lr_interval = [360, 120, 60, 60]
    elif args.epochs == 120:
        lr_interval = [60, 30, 15, 15]
    elif args.epochs == 30:
        lr_interval = [15, 7, 3, 3]
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