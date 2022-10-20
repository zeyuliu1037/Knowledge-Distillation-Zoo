# CUDA_VISIBLE_DEVICES=1 python -u train_kd.py \
#                            --save_root "./results/logits/" \
#                            --t_model "vgg16/base/base-c10-vgg16_cus_1014/model_best_9185.pth.tar" \
#                            --s_init "vgg16/base/base-c10-vgg16_cus16_1014/model_best_9056.pth.tar" \
#                            --data_name cifar10 \
#                            --num_class 10 \
#                            --t_name vgg16 \
#                            --s_name vgg16 \
#                            --t_type 'cus' \
#                            --s_type 'cus' \
#                            --t_ch 64 \
#                            --s_ch 16 \
#                            --optimizer 'Adam' \
#                            --lr 1e-4 \
#                            --epochs 600 \
#                            --kd_mode logits \
#                            --lambda_kd 0.1 \
#                            --note 'logits-c10-vgg16_ch16-kd_logits_with_act'

CUDA_VISIBLE_DEVICES=3 python -u train_kd.py \
                           --t_model "results/kd/vgg16/at-c10-vgg16-ch64-without-act-loss/model_best_9208.pth.tar" \
                           --s_init "vgg16/base/base-c10-vgg16_cus16_1014/model_best_9056.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name vgg16 \
                           --s_name vgg16 \
                           --t_type 'cus' \
                           --s_type 'cus' \
                           --t_ch 64 \
                           --s_ch 16 \
                           --optimizer 'Adam' \
                           --weight_decay 0 \ 
                           --lr 1e-4 \
                           --epochs 600 \
                           --kd_mode logits \
                           --lambda_kd 0.1 \
                           --note 'at-c10-vgg16-ch64-cus64_16-kd_logits'

# CUDA_VISIBLE_DEVICES=5 python -u train_kd.py \
#                            --t_model "/root/autodl-tmp/results/resnet18/base-c10-resnet18_cus64_1017/model_best_9138.pth.tar" \
#                            --s_init "/root/autodl-tmp/results/resnet18/base-c10-resnet18_cus8_1018/model_best_8986.pth.tar" \
#                            --data_name cifar10 \
#                            --num_class 10 \
#                            --t_name resnet20 \
#                            --s_name resnet20 \
#                            --t_type 'cus' \
#                            --s_type 'cus' \
#                            --t_ch 64 \
#                            --s_ch 8 \
#                            --optimizer 'SGD' \
#                            --lr 0.01 \
#                            --epochs 600 \
#                            --kd_mode logits \
#                            --lambda_kd 0.1 \
#                            --note 'at-c10-resnet20-ch64_8-without-act-loss'