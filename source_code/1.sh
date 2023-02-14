# !/bin/sh
source activate yyx_ishen
group='snrmap-0214'
nohup_dir='../nohup_log'
mkdir -p ${nohup_dir}
for algo in DPPO G2ANet;
do
nohup python -u main_DPPO.py --group ${group} --algo ${algo} --dyna_level 4 --update_num 5 --user_data_amount 5 --device cuda:4 >> ${nohup_dir}/0214.log 2>&1 &
nohup python -u main_DPPO.py --group ${group} --algo ${algo} --use_snrmap --dyna_level 4 --update_num 5 --user_data_amount 5 --device cuda:4 >> ${nohup_dir}/0214.log 2>&1 &
done