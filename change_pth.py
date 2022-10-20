import torch

model_name = '/root/autodl-tmp/results/vgg16/base-imagenet-cus64-pretrained2/model_best.pth.tar'
model = torch.load(model_name)

# print(model['net'].keys())
for key in model['net'].keys():
    if key == 'features.0.weights':
        features_0_weights = model['net'][key]
        print(key,features_0_weights.shape, torch.sum(features_0_weights))
    if key == 'features.0.identity_kernel':
        features_0_identity_kernel = model['net'][key]
        print(key,features_0_weights.shape, torch.mean(features_0_identity_kernel))

features_0_weights_parts = torch.split(features_0_weights, 4, dim=0)
features_0_weights_d4 = torch.cat([torch.sum(a, dim=0, keepdim=True) for a in features_0_weights_parts], dim=0)
print(features_0_weights_d4.shape, torch.sum(features_0_weights_d4))

features_0_identity_kernel_parts = torch.split(features_0_identity_kernel, 4, dim=0)
features_0_identity_kernel_d4 = torch.cat([torch.mean(a, dim=0, keepdim=True) for a in features_0_identity_kernel_parts], dim=0)
print(features_0_identity_kernel_d4.shape, torch.mean(features_0_identity_kernel_d4))

model_mean = model['net']
model_mean['features.0.weights'] = features_0_weights_d4
model_mean['features.0.identity_kernel'] = features_0_identity_kernel_d4
model['net'] = model_mean
torch.save(model, '/root/autodl-tmp/results/vgg16/base-imagenet-cus64-pretrained2/model_best_sum.pth.tar')