# !/bin/sh
source activate yyx_ishen
group='0217-adjust-uavheight'
for uav_height in 100 80 60;
do
nohup python -u main_DPPO.py --group ${group} --n_thread 16 --uav_height ${uav_height} --device cuda:7 &
done