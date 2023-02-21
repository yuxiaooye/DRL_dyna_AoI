# !/bin/sh
source activate yyx_adept
group='0221-night-ablation'

nohup python -u main_DPPO.py --group $group --dataset NCSU --uav_num 5 --aoith 30 --w_noise -104 --algo G2ANet --use_snrmap --knn_coefficient 0.1 --user yyx --device cuda:0 &
nohup python -u main_DPPO.py --group $group --dataset NCSU --uav_num 5 --aoith 30 --w_noise -104 --algo G2ANet                                    --user yyx --device cuda:1 &
nohup python -u main_DPPO.py --group $group --dataset NCSU --uav_num 5 --aoith 30 --w_noise -104 --algo DPPO --use_snrmap                         --user yyx --device cuda:2 &
nohup python -u main_DPPO.py --group $group --dataset NCSU --uav_num 5 --aoith 30 --w_noise -104 --algo DPPO                                      --user yyx --device cuda:3 &

nohup python -u main_DPPO.py --group $group --dataset KAIST --uav_num 5 --aoith 30 --w_noise -104 --algo G2ANet --use_snrmap --knn_coefficient 0.5 --user yyx --device cuda:0 &
nohup python -u main_DPPO.py --group $group --dataset KAIST --uav_num 5 --aoith 30 --w_noise -104 --algo G2ANet                                    --user yyx --device cuda:1 &
nohup python -u main_DPPO.py --group $group --dataset KAIST --uav_num 5 --aoith 30 --w_noise -104 --algo DPPO --use_snrmap                         --user yyx --device cuda:2 &
nohup python -u main_DPPO.py --group $group --dataset KAIST --uav_num 5 --aoith 30 --w_noise -104 --algo DPPO                                      --user yyx --device cuda:3 &