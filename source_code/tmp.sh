# !/bin/sh
source activate yyx_ishen
group='check-no-vital-debug-0210'
nohup_dir='../nohup_log'
mkdir -p ${nohup_dir}
nohup python -u main_DPPO.py --group ${group} --device cuda:5 >> ${nohup_dir}/0210.log 2>&1 &
