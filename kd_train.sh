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

# CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
#                             --save_root '/root/autodl-tmp/results' \
#                             --t_model "/root/autodl-tmp/results/without_module_vgg_cus16_91.27.pth.tar" \
#                             --s_init "/root/autodl-tmp/results/vgg_cus8_8912.pth.tar" \
#                             --data_name cifar10 \
#                             --num_class 10 \
#                             --t_name vgg16 \
#                             --s_name vgg16 \
#                             --t_type 'cus' \
#                             --s_type 'cus' \
#                             --t_ch 16 \
#                             --s_ch 8 \
#                             --optimizer 'Adam' \
#                             --batch_size 128 \
#                             --weight_decay 0 \
#                             --lr 1e-5 \
#                             --epochs 600 \
#                             --kd_mode logits \
#                             --lambda_kd 0.1 \
#                             --note 'at-c10-vgg16-ch64-cus16_cus4-kd_logits_small_lr'

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

# CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
#                            --t_model "results/resnet18/base-c10-resnet18_cus64_1017/model_best_9138.pth.tar" \
#                            --s_init "results/resnet18/base-c10-resnet18_cus8/model_best_8969.pth.tar" \
#                            --data_name cifar10 \
#                            --num_class 10 \
#                            --t_name resnet20_multi \
#                            --s_name resnet20_multi \
#                            --t_type 'cus' \
#                            --s_type 'cus' \
#                            --t_ch 64 \
#                            --s_ch 8 \
#                            --optimizer 'SGD' \
#                            --lr 0.001 \
#                            --epochs 300 \
#                            --kd_mode at \
#                            --lambda_kd 1000.0 \
#                            --p 2.0 \
#                            --note at-c10-r20_cus64_cus8_at_10kd1

CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root '/root/autodl-tmp/results' \
                           --t_model "/root/autodl-tmp/results/resnet_cus64_9138.pth.tar" \
                           --s_init "/root/autodl-tmp/results/resnet_cus8_8986.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet20_multi \
                           --s_name resnet20_multi \
                           --t_type 'cus' \
                           --s_type 'cus' \
                           --t_ch 64 \
                           --s_ch 8 \
                           --optimizer 'SGD' \
                           --lr 0.1 \
                           --epochs 600 \
                           --kd_mode at \
                           --lambda_kd 1000.0 \
                           --p 2.0 \
                           --note at-c10-r20_cus64_cus8_at_100kd1_1000kd_large_lr_01