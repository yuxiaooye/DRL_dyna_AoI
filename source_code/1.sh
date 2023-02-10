# !/bin/sh
source activate yyx_adept
group_postfix='tune-env-0210-second'
nohup_dir='../nohup_log'
mkdir -p ${nohup_dir}
for user_data_amount in 1 5 10;
do
for update_num in 10 4;
do
if [ $update_num == 10 ]
then device="cuda:6"
else [ $update_num == 4 ]
device="cuda:7"
fi
nohup python -u main_DPPO.py --group_postfix ${group_postfix} --dyna_level 3 --user_data_amount ${user_data_amount} --update_num ${update_num} --device ${device} >> ${nohup_dir}/0210.log 2>&1 &
done
done