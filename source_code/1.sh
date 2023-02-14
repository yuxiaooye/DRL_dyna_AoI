# !/bin/sh
source activate yyx_ishen
group='extendV-ablation-0214'
nohup_dir='../nohup_log'
mkdir -p ${nohup_dir}
nohup python -u main_DPPO.py --group ${group} --device cuda:6 >> ${nohup_dir}/0214.log 2>&1 &
nohup python -u main_DPPO.py --group ${group} --use-extended-value --device cuda:6 >> ${nohup_dir}/0214.log 2>&1 &