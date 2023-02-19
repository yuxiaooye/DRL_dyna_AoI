# !/bin/sh
source activate yyx_ishen
group='0219-hao02191630'
for update_num in 5 10;
do
nohup python -u main_DPPO.py --group $group --poi_num 50 --update_num $update_num --algo G2ANet --use_snrmap --knn_coefficient 0.5 --user yyx --device cuda:5 &
done