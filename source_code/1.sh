# !/bin/sh
source activate yyx_adept
group='0222-morning-hyper'
nohup python -u main_DPPO.py --group $group --dataset NCSU --uav_num 5 --algo G2ANet --use_snrmap --knn_coefficient 0.1 --g2a_hops 0 --user yyx --device cuda:0 &
nohup python -u main_DPPO.py --group $group --dataset NCSU --uav_num 5 --algo G2ANet --use_snrmap --knn_coefficient 0.1 --g2a_hops 1 --user yyx --device cuda:0 &
nohup python -u main_DPPO.py --group $group --dataset NCSU --uav_num 5 --n_iter 8000 --algo G2ANet --use_snrmap --knn_coefficient 0.1 --g2a_hops 2 --user yyx --device cuda:0 &
nohup python -u main_DPPO.py --group $group --dataset NCSU --uav_num 5 --n_iter 8000 --algo G2ANet --use_snrmap --knn_coefficient 0.1 --g2a_hops 3 --user yyx --device cuda:0 &
