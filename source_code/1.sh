# !/bin/sh
source activate yyx_ishen
group='dmpo-mlp-0212'
nohup_dir='../nohup_log'
mkdir -p ${nohup_dir}
nohup python -u main_DPPO.py --algo DMPO --group ${group} --n_thread 4 --use-mlp-model --device cuda:3 >> ${nohup_dir}/0212.log 2>&1 &
nohup python -u main_DPPO.py --algo DMPO --group ${group} --n_thread 4 --device cuda:3 >> ${nohup_dir}/0212.log 2>&1 &