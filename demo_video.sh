#!/bin/bash

# create an empty folder for experimental results
mkdir -p experiments/results_demo_carla_video
mkdir -p experiments/results_demo_faster_video
mkdir -p experiments/results_demo_bsrsc_video
mkdir -p experiments/results_demo_gevrs_video_1

cd deep_unroll_net

###
python inference_demo_video.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_carla_video \
            --data_dir='../demo_video/Carla' \
            --img_H=448 \
            --gamma=1.0 \
            --log_dir=../deep_unroll_weights/pre_carla_ft

:<<!
python inference_demo_video.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_faster_video \
            --data_dir='../demo_video/Fastec' \
            --img_H=480 \
            --gamma=1.0 \
            --log_dir=../deep_unroll_weights/pre_fastec_ft
!

:<<!
python inference_demo_video.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_bsrsc_video \
            --data_dir='../demo_video/BSRSC' \
            --img_H=768 \
            --gamma=0.45 \
            --log_dir=../deep_unroll_weights/pre_bsrsc_ft
!

:<<!
python inference_demo_video_gevrs.py \
            --results_dir=../experiments/results_demo_gevrs_video_1 \
            --model_label='pre' \
            --data_dir='../demo_video/GevRS/01' \
            --is_GevRS=1 \
            --gamma=0.67 \
            --log_dir=../deep_unroll_weights/pre_gevrs_ft
!