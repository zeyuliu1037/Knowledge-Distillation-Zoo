import torch

model_name = '/root/autodl-tmp/results/vgg16/base-imagenet-cus64-pretrained2/model_best_6382.pth.tar'
model = torch.load(model_name)

# print(model['net'].keys())
change_params = ['features.0.weights', 'features.0.identity_kernel', 'features.1.weight', 'features.1.bias', 'features.1.running_mean', 'features.1.running_var', 'features.2.running_hoyer_thr']
for key in model['net'].keys():
    if key in change_params:
        change_param = model['net'][key]
        change_param_parts = torch.split(change_param, 4, dim=0)
        change_param_d4 = torch.cat([torch.mean(a, dim=0, keepdim=True) for a in change_param_parts], dim=0)
        model['net'][key] = change_param_d4
    elif key == 'features.4.weight':
        change_param = model['net'][key]
        change_param_parts = torch.split(change_param, 4, dim=1)
        change_param_d4 = torch.cat([torch.mean(a, dim=1, keepdim=True) for a in change_param_parts], dim=1)
        model['net'][key] = change_param_d4

torch.save(model, '/root/autodl-tmp/results/vgg16/base-imagenet-cus64-pretrained2/model_best_mean.pth.tar')