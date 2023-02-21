# !/bin/sh
source activate yyx_ishen
group='0221-for-lower-aoi-1'
for uav_num in 5;
do
nohup python -u main_DPPO.py --mute_wandb --group $group --dataset NCSU --poi_num 48 --uav_num $uav_num --aoith 45 --algo G2ANet --use_snrmap --knn_coefficient 0.1 --user yyx --device cuda:3 &
nohup python -u main_DPPO.py --mute_wandb --group $group --dataset NCSU --poi_num 48 --uav_num $uav_num --aoith 30 --algo G2ANet --use_snrmap --knn_coefficient 0.1 --user yyx --device cuda:4 &
nohup python -u main_DPPO.py --mute_wandb --group $group --dataset NCSU --poi_num 48 --uav_num $uav_num --aoith 15 --algo G2ANet --use_snrmap --knn_coefficient 0.1 --user yyx --device cuda:5 &
done