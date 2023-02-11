# !/bin/sh
source activate yyx_ishen
group='tune-env-dynaQoS-0211'
nohup_dir='../nohup_log'
mkdir -p ${nohup_dir}
for dyna_level in 1 2 3;
do
nohup python -u main_DPPO.py --group ${group} --dyna_level ${dyna_level} --device cuda:6 >> ${nohup_dir}/0211.log 2>&1 &
done