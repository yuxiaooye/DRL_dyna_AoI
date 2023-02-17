# !/bin/sh
source activate yyx_ishen
group='0217-adjust-aoith'
for aoith in 60 50 40 30;
do
nohup python -u main_DPPO.py --group ${group} --aoith ${aoith} --n_thread 16 --device cuda:7 &
done