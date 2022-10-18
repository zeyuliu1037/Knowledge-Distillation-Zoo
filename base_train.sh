CUDA_VISIBLE_DEVICES=0 python -u train_base.py \
                           --data_name cifar10 \
                           --epochs 600 \
                           --lr 0.1 \
                           --optimizer SGD \
                           --print_freq 200 \
                           --num_class 10 \
                           --net_name resnet18 \
                           --net_type 'cus' \
                           --first_ch 8 \
                           --note base-c10-resnet18_cus8_1018
                           # --pretrained 'results/resnet18/base-c10-resnet18_cus64_1017/model_best_9122.pth.tar'
                        #    --pretrained 'model_t/ann_vgg16_light_cifar10_202209290044.pth' \

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -u train_base.py \
#                            --save_root "./results/resnet/base/" \
#                            --epochs 120 \
#                            --lr 0.0001 \
#                            --optimizer Adam \
#                            --weight_decay 5e-6 \
#                            --num_class 1000 \
#                            --net_name 'vgg16' \
#                            --data_name 'imagenet' \
#                            --net_type 'cus' \
#                            --first_ch 64 \
#                            --note base-imagenet-resnet20_ori \
#                            --cuda 4