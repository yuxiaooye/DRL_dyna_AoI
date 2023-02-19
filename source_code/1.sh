# !/bin/sh
source activate yyx_ishen
group='0219-adjust-VPS-knn'
for aVPS in 0.2 0.3;
do
for tVPS in 0.05 0.1;
do
nohup python -u main_DPPO.py --group ${group} --poi_num 50 --algo G2ANet --aVPS $aVPS --tVPS $tVPS --knn_coefficient 0.5 --use_snrmap --user yyx --device cuda:7 &
done
done