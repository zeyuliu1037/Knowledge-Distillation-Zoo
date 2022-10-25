import torch
import copy

model_name = '/root/autodl-tmp/results/vgg_cus64_9208.pth.tar'
# model_name = '/root/autodl-tmp/results/vgg_cus16_91.27.pth.tar'
model = torch.load(model_name)

# for key in model['snet'].keys():
#     print(key, model['snet'][key].shape)
#     break


# # print(model['net'].keys())
# change_params = ['features.0.weights', 'features.0.identity_kernel', 'features.1.weight', 'features.1.bias', 'features.1.running_mean', 'features.1.running_var', 'features.2.running_hoyer_thr']
# for key in model['net'].keys():
#     if key in change_params:
#         change_param = model['net'][key]
#         change_param_parts = torch.split(change_param, 4, dim=0)
#         change_param_d4 = torch.cat([torch.mean(a, dim=0, keepdim=True) for a in change_param_parts], dim=0)
#         model['net'][key] = change_param_d4
#     elif key == 'features.4.weight':
#         change_param = model['net'][key]
#         change_param_parts = torch.split(change_param, 4, dim=1)
#         change_param_d4 = torch.cat([torch.mean(a, dim=1, keepdim=True) for a in change_param_parts], dim=1)
#         model['net'][key] = change_param_d4

# torch.save(model, '/root/autodl-tmp/results/vgg16/base-imagenet-cus64-pretrained2/model_best_mean.pth.tar')

# print(model.keys()) # ['epoch', 'snet', 'tnet', 'prec@1', 'prec@5']
model_new = copy.deepcopy(model)
model_new['net'] = {}
for key in model['snet'].keys():
    if key[:7] == 'module.':
        update_key = key[7:]
        model_new['net'][update_key] = model_new['snet'].pop(key)
model_new.pop('snet')
model_new.pop('tnet')
print(model_new.keys())
torch.save(model_new, '/root/autodl-tmp/results/without_module_vgg_cus64_9208.pth.tar')