# !/bin/sh
source activate yyx_ishen
group='0217-change-aoith'
for aoith in 100 80 60 40;
do
nohup python -u main_DPPO.py --group ${group} --aoith ${aoith} --n_thread 32 --device cuda:3 &
done