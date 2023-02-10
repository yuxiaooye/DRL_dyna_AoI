# !/bin/sh
source activate yyx_ishen
group_postfix='baselines-0210'
nohup_dir='../nohup_log'
mkdir -p ${nohup_dir}
for algo in 'DMPO' 'DPPO' 'CPPO' 'IC3Net' 'IA2C';
do
nohup python -u main_DPPO.py --group_postfix ${group_postfix} --user_data_amount 5 --algo ${algo} --device cuda:5 >> ${nohup_dir}/0210.log 2>&1 &
done