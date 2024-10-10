#--load_1st_GS=0 ==> The middle scanline
#--load_1st_GS=1 ==> The first scanline

# !! Updata the path to the dataset and directory to 
# !! save your trained models with your own local path !!
carla_dataset_type=Carla
carla_root_path_test_data=/data/local_userdata/fanbin/raw_data/carla/data_test/test/

fastec_dataset_type=Fastec
fastec_root_path_test_data=/data/local_userdata/fanbin/raw_data/faster/data_test/test/

bsrsc_dataset_type=BSRSC
bsrsc_root_path_test_data=/data/local_userdata/fanbin/raw_data/BSRSC/test/

dir_pretrained_GS=../deep_unroll_weights/Pretrained/pretrain_vfi/
results_dir=/home/fanbin/fan/SelfRSSplat/deep_unroll_results/

model_dir_carla=../deep_unroll_weights/pre_carla_ft/
model_dir_faste=../deep_unroll_weights/pre_fastec_ft/
model_dir_bsrsc=../deep_unroll_weights/pre_bsrsc_ft/

cd deep_unroll_net


python inference.py \
          --dataset_type=$carla_dataset_type \
          --dataset_root_dir=$carla_root_path_test_data \
          --log_dir=$model_dir_carla \
          --results_dir=$results_dir \
          --img_H=448 \
          --gamma=1.0 \
          --compute_metrics \
          --model_label=pre \
          --load_1st_GS=1  


:<<!
python inference.py \
          --dataset_type=$fastec_dataset_type \
          --dataset_root_dir=$fastec_root_path_test_data \
          --log_dir=$model_dir_faste \
          --results_dir=$results_dir \
          --img_H=480 \
          --gamma=1.0 \
          --compute_metrics \
          --model_label=pre \
          --load_1st_GS=1
!

:<<!
python inference.py \
          --dataset_type=$bsrsc_dataset_type \
          --dataset_root_dir=$bsrsc_root_path_test_data \
          --log_dir=$model_dir_bsrsc \
          --results_dir=$results_dir \
          --img_H=768 \
          --gamma=0.45 \
          --compute_metrics \
          --model_label=pre \
          --load_1st_GS=0
!




