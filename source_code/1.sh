# !/bin/sh
source activate yyx_ishen
group='baselines-0212'
nohup_dir='../nohup_log'
mkdir -p ${nohup_dir}
for algo in 'DMPO' 'DPPO' 'CPPO' 'IC3Net' 'IA2C';
do
nohup python -u main_DPPO.py --group ${group} --algo ${algo} --device cuda:5 >> ${nohup_dir}/0212.log 2>&1 &
done