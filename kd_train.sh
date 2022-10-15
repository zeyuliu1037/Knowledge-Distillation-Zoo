CUDA_VISIBLE_DEVICES=1 python -u train_kd.py \
                           --save_root "./results/logits/" \
                           --t_model "vgg16/base/base-c10-vgg16_ori_finetune/model_best_9316.pth.tar" \
                           --s_init "vgg16/base/base-c10-vgg16_ori_1014/model_best_9185.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name vgg16 \
                           --s_name vgg16 \
                           --t_type 'ori' \
                           --s_type 'cus' \
                           --t_ch 64 \
                           --s_ch 64 \
                           --optimizer 'Adam' \
                           --lr 1e-5 \
                           --kd_mode logits \
                           --lambda_kd 0.1 \
                           --note logits-c10-vgg16-kd_logits