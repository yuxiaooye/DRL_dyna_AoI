# !/bin/sh
source activate yyx_ishen
group='tune-env-0211'
nohup_dir='../nohup_log'
mkdir -p ${nohup_dir}
for user_data_amount in 1 5 10;  # 可以只提高user_data_amount，但不能只降低update_num
do
for update_num in 10 5;
do
if [ $update_num == 10 ]
then device="cuda:3"
else
device="cuda:4"
fi
nohup python -u main_DPPO.py --group ${group} --user_data_amount ${user_data_amount} --update_num ${update_num} --device ${device} >> ${nohup_dir}/0211.log 2>&1 &
done
done