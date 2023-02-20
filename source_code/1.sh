# !/bin/sh
source activate yyx_ishen
group='0220-snr500'
nohup python -u main_DPPO.py --group $group --poi_num 50 --agent_field 750 --txth 1 --user_data_amount 0.75 --w_noise -110 --algo G2ANet --use_snrmap --knn_coefficient 0.25 --user yyx --device cuda:2 &
nohup python -u main_DPPO.py --group $group --poi_num 50 --agent_field 750 --txth 3 --user_data_amount 0.75 --w_noise -110 --algo G2ANet --use_snrmap --knn_coefficient 0.25 --user yyx --device cuda:2 &
nohup python -u main_DPPO.py --group $group --poi_num 50 --agent_field 750 --txth 1 --user_data_amount 0.75 --w_noise -110 --algo G2ANet --use_snrmap --knn_coefficient 0.5 --user yyx --device cuda:2 &
nohup python -u main_DPPO.py --group $group --poi_num 50 --agent_field 750 --txth 3 --user_data_amount 0.75 --w_noise -110 --algo G2ANet --use_snrmap --knn_coefficient 0.5 --user yyx --device cuda:2 &