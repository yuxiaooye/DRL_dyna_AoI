# !/bin/sh
source activate yyx_adept
group='0221-hyper'

for g2a_hops in 1 2 3;
do
for map_size in 4 6 8 10;
do
nohup python -u main_DPPO.py --group $group --dataset NCSU --uav_num 5 --aoith 30 --w_noise -107 --algo G2ANet --use_snrmap --knn_coefficient 0.1 --g2a_hops $g2a_hops --map_size $map_size --user yyx --device cuda:1 &
nohup python -u main_DPPO.py --group $group --dataset KAIST --uav_num 5 --aoith 30 --w_noise -107 --algo G2ANet --use_snrmap --knn_coefficient 0.1 --g2a_hops $g2a_hops --map_size $map_size --user yyx --device cuda:2 &
done
done
