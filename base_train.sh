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