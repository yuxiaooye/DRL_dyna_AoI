# !/bin/bash
source activate yyx_adept

group_postfix='stack-frame-0209'
nohup_dir='../nohup_log'
mkdir -p ${nohup_dir}
for SF in 0 1;
do
if [ $SF == 1 ]
then stub="--debug_use_stack_frame"
else stub=""
fi
nohup python -u main_DPPO.py --group_postfix ${group_postfix} --dyna_level 3 ${stub} >> ${nohup_dir}/0209.log 2>&1 &
done


