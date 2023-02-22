# !/bin/sh
source activate yyx_adept
nohup python -u main_DPPO.py --group 0221-five-uavnum --uav_num 2 --algo Random --user yyx --device cuda:0 &
nohup python -u main_DPPO.py --group 0221-five-uavnum --uav_num 3 --algo Random --user yyx --device cuda:0 &
nohup python -u main_DPPO.py --group 0221-five-uavnum --uav_num 4 --algo Random --user yyx --device cuda:0 &
nohup python -u main_DPPO.py --group 0221-five-uavnum --uav_num 5 --algo Random --user yyx --device cuda:0 &
nohup python -u main_DPPO.py --group 0221-five-uavnum --uav_num 7 --algo Random --user yyx --device cuda:0 &
nohup python -u main_DPPO.py --group 0221-five-uavnum --uav_num 10 --algo Random --user yyx --device cuda:0 &



