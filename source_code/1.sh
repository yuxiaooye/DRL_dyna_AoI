# !/bin/sh
source activate yyx_ishen
group='vec-try-0212'
nohup_dir='../nohup_log'
mkdir -p ${nohup_dir}
for n_thread in 4 8 16;
do
nohup python -u main_DPPO.py --group ${group} --n_thread ${n_thread} --device cuda:6 >> ${nohup_dir}/0212.log 2>&1 &
done