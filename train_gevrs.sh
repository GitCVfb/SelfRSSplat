#!/bin/bash

# !! Updata the path to the dataset and directory to 
# !! save your trained models with your own local path !!
gevsc_dataset_type=GevRSReal
gevsc_root_path_training_data=/data/local_userdata/fanbin/raw_data/Gev-RS-Real/01/
#01_001435_032300: (20, 121-seq_len+1)        0.67        346*260

log_dir_pretrained_GS=/home/fanbin/fan/SelfSoftSplat/deep_unroll_weights/Pretrained/pretrain_vfi/
log_dir=/home/fanbin/fan/SelfSoftSplat/deep_unroll_weights/
#
cd deep_unroll_net

python train_SelfRSSR.py \
          --dataset_type=$gevsc_dataset_type \
          --dataset_root_dir=$gevsc_root_path_training_data \
          --log_dir_pretrained_GS=$log_dir_pretrained_GS \
          --log_dir=$log_dir \
          --lamda_L1=10 \
          --lamda_perceptual=1 \
          --lamda_flow_smoothness=0.1 \
          --gamma=0.67 \
          --img_H=260 \
          #--continue_train=True \
          #--start_epoch=201 \
          #--model_label=200