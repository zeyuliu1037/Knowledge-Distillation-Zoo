CUDA_VISIBLE_DEVICES=1 python -u train_base.py \
                           --save_root "./vgg16/base/" \
                           --data_name cifar10 \
                           --epochs 600 \
                           --lr 0.0001 \
                           --optimizer Adam \
                           --print_freq 200 \
                           --num_class 10 \
                           --net_name vgg16 \
                           --note base-c10-vgg16_ori

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -u train_base.py \
#                            --save_root "./result/resnet/base/" \
#                            --epochs 120 \
#                            --lr 0.0001 \
#                            --optimizer Adam \
#                            --weight_decay 5e-6 \
#                            --num_class 1000 \
#                            --net_name 'vgg16' \
#                            --data_name 'imagenet' \
#                            --note base-imagenet-resnet20_ori \
#                            --cuda 4