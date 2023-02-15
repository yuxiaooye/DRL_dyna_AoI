# !/bin/sh
source activate yyx_ishen
group='0215-SSTSS'
nohup_dir='../nohup_log'
mkdir -p ${nohup_dir}
for uav_num in 3 4 5;
do
for update_num in 5 10;
do
nohup python -u main_DPPO.py --group ${group} --use_extended_value --dyna_level SSTSS --update_num ${update_num} --uav_num ${uav_num} --device cuda:3 >> ${nohup_dir}/0215.log 2>&1 &
nohup python -u main_DPPO.py --group ${group} --use_extended_value --dyna_level SSTSS --update_num ${update_num} --uav_num ${uav_num} --amount_prop_to_SNRth --device cuda:4 >> ${nohup_dir}/0215.log 2>&1 &
done
done