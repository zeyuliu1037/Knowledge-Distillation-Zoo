from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import shutil
import numpy as np
import torch
from typing import Optional
import torch.nn as nn


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val   = 0
		self.avg   = 0
		self.sum   = 0
		self.count = 0

	def update(self, val, n=1):
		self.val   = val
		self.sum   += val * n
		self.count += n
		self.avg   = self.sum / self.count


def count_parameters_in_MB(model):
	return sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6


def create_exp_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	print('Experiment dir : {}'.format(path))


def load_pretrained_model(model, pretrained_dict):
	model_dict = model.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict) 
	# 3. load the new state dict
	model.load_state_dict(model_dict)


def transform_time(s):
	m, s = divmod(int(s), 60)
	h, m = divmod(m, 60)
	return h,m,s


def save_checkpoint(state, is_best, save_root):
	save_path = os.path.join(save_root, 'checkpoint.pth.tar')
	torch.save(state, save_path)
	if is_best:
		best_save_path = os.path.join(save_root, 'model_best.pth.tar')
		shutil.copyfile(save_path, best_save_path)


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred    = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		# correct_k = correct[:k].view(-1).float().sum(0)
		correct_k = correct[:k].reshape(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


# try combined threshold in hoyerBiAct
class HoyerBiAct(nn.Module):
    """
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """
    _version = 2
    __constants__ = ["num_features", "eps", "momentum", "spike_type", "x_thr_scale", "if_spike", "track_running_stats"]
    num_features: int
    eps: float
    momentum: float
    spike_type: str
    x_thr_scale: float
    if_spike: bool
    track_running_stats: bool
    # spike_type is args.act_mode
    def __init__(self, num_features=1, eps=1e-05, momentum=0.1, spike_type='sum', track_running_stats: bool = True, device=None, dtype=None, \
        min_thr_scale=0.0, max_thr_scale=1.0, x_thr_scale=1.0, if_spike=True):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HoyerBiAct, self).__init__()
        self.num_features   = num_features if spike_type == 'cw' else 1
        self.eps            = eps
        self.momentum       = momentum
        self.spike_type     = spike_type
        self.track_running_stats = track_running_stats
        self.threshold      = nn.Parameter(torch.tensor(1.0))
        self.min_thr_scale  = min_thr_scale
        self.max_thr_scale  = max_thr_scale
        self.x_thr_scale    = x_thr_scale
        self.if_spike       = if_spike  
        # self.register_buffer('x_thr_scale', torch.tensor(x_thr_scale))
        # self.register_buffer('if_spike', torch.tensor(if_spike))
             

        # self.running_hoyer_thr = 0.0 if spike_type != 'cw' else torch.zeros(num_features).cuda()
        if self.track_running_stats:
            self.register_buffer('running_hoyer_thr', torch.zeros(self.num_features, **factory_kwargs))
            self.running_hoyer_thr: Optional[torch.Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        else:
            self.register_buffer("running_hoyer_thr", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_running_stats()
    
    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_hoyer_thr/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_hoyer_thr.zero_()  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def forward(self, input):
        # calculate running estimates
        input = input / torch.abs(self.threshold)
        # input = torch.clamp(input, min=0.0, max=1.0)
        if self.training:
            clamped_input = torch.clamp((input).clone().detach(), min=0.0, max=1.0)
            # clamped_input[clamped_input >= 1.0] = 0.0
            # clamped_input = input.clone().detach()
            if self.spike_type == 'sum':
                hoyer_thr = torch.sum((clamped_input)**2) / torch.sum(torch.abs(clamped_input))
                # if torch.sum(torch.abs(clamped_input)) > 0:
                #     hoyer_thr = torch.sum((clamped_input)**2) / torch.sum(torch.abs(clamped_input))
                # else:
                #     print('Warning: the output is all zero!!!')

                #     hoyer_thr = self.running_hoyer_thr
            elif self.spike_type == 'fixed':
                hoyer_thr = 1.0                
            elif self.spike_type == 'cw':
                hoyer_thr = torch.sum((clamped_input)**2, dim=(0,2,3)) / torch.sum(torch.abs(clamped_input), dim=(0,2,3))
                # 1.0 is the max thr
                hoyer_thr = torch.nan_to_num(hoyer_thr, nan=1.0)
                # hoyer_thr = torch.mean(hoyer_cw, dim=0)
            
            with torch.no_grad():
                self.running_hoyer_thr = self.momentum * hoyer_thr\
                    + (1 - self.momentum) * self.running_hoyer_thr
        else:
            hoyer_thr = self.running_hoyer_thr
      
        # 
        out = Spike_func.apply(input, hoyer_thr, self.x_thr_scale, self.spike_type, self.if_spike)

        return out

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, spike_type={spike_type}, x_thr_scale={x_thr_scale}, if_spike={if_spike}, track_running_stats={track_running_stats}".format(**self.__dict__)
        )
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(HoyerBiAct, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

class Spike_func(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    @staticmethod
    def forward(ctx, input, hoyer_thr, x_thr_scale=1.0, spike_type='sum', if_spike=True):
        ctx.save_for_backward(input)
        out = torch.clamp(input, min=0.0, max=1.0)
        ctx.if_spike = if_spike
        # print('input shape: {}, hoyer thr shape: {}, x_thr_scale: {}'.format(input.shape, hoyer_thr, x_thr_scale))
        if spike_type != 'cw':
            if if_spike:
                out[out < x_thr_scale*hoyer_thr] = 0.0
            # print('out shape: {}, x scale: {}, hoyer_thr: {}'.format(out.shape, x_thr_scale, hoyer_thr))
            out[out >= x_thr_scale*hoyer_thr] = 1.0
        else:
            if if_spike:
                out[out<x_thr_scale*hoyer_thr[None, :, None, None]] = 0.0
            out[out>=x_thr_scale*hoyer_thr[None, :, None, None]] = 1.0 
            # out[out<0.1*x_thr_scale*hoyer_thr[None, :, None, None]] = 0.0
            # out[out>=0.9*x_thr_scale*hoyer_thr[None, :, None, None]] = 1.0 
                    
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,  = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_inp = torch.zeros_like(input).cuda()

        grad_inp[input > 0] = 1.0
        # only for
        grad_inp[input > 2.0] = 0.0

        # grad_scale = 0.5 if ctx.if_spike else 1.0
        grad_scale = 0.5
    

        return grad_scale*grad_inp*grad_input, None, None, None, None

class customConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.register_buffer('identity_kernel', torch.ones(out_channels, in_channels, *kernel_size))
        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size), requires_grad=True)
        with torch.no_grad():
            self.weights.data.normal_(0.0, 0.8)

    def forward(self, img):
        
        b, c, h, w = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        p00 = 0.0
        p01 = -0.000287
        p10 = 0.0
        p11 = 0.266
        p20 = 0.0
        p21 = -0.1097
        p30 = 0.0

        img_unf = nn.functional.unfold(img, kernel_size=self.kernel_size,
                                       stride=self.stride, padding=self.padding).transpose(1, 2).contiguous()
        self.identity_kernel = self.identity_kernel.contiguous()
        identity_weights = self.identity_kernel.view(self.identity_kernel.size(0), -1).contiguous()
        self.weights = self.weights.contiguous()
        weights = self.weights.view(self.weights.size(0), -1).contiguous()

        # f0 = (p00 + torch.zeros_like(img_unf)).matmul(identity_weights.t())
        # f1 = (p10 * (img_unf - 0.5)).matmul(identity_weights.t())
        # f2 = (p01 * torch.ones_like(img_unf)).matmul(weights.t())
        # f3 = (p20 * torch.pow(img_unf - 0.5, 2)).matmul(identity_weights.t())
        # f4 = (p11 * (img_unf - 0.5)).matmul(weights.t())
        # f5 = (p30 * torch.pow(img_unf - 0.5, 3)).matmul(identity_weights.t())
        # f6 = (p21 * torch.pow(img_unf - 0.5, 2)).matmul(weights.t())
        # f = (f0 + f1 + f2 + f3 + f4 + f5 + f6).transpose(1, 2)

        f = ((p00 + torch.zeros_like(img_unf) +
             p10 * (img_unf) +
             p20 * torch.pow(img_unf, 2) +
             p30 * torch.pow(img_unf, 3)).matmul(identity_weights.t()) + \
            (p01 * torch.ones_like(img_unf) +
             p11 * (img_unf) +
             p21 * torch.pow(img_unf, 2)
             ).matmul(weights.t().contiguous())).transpose(1, 2).contiguous()

        
        out_xshape = int((h-self.kernel_size[0]+2*self.padding)/self.stride) + 1
        out_yshape = int((w-self.kernel_size[1]+2*self.padding)/self.stride) + 1
        #out = f.contiguous()
        out = f.view(b, self.out_channels, out_xshape, out_yshape)#.contiguous()
        out = out/(3*self.kernel_size[0]*self.kernel_size[1])
        return out
    def extra_repr(self):
        return (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}".format(**self.__dict__)
        )