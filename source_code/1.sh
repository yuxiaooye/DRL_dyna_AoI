# !/bin/sh
source activate yyx_ishen
group='0215-snrmap-shortcut'
nohup_dir='../nohup_log'
mkdir -p ${nohup_dir}
for algo in IPPO DPPO;
do
nohup python -u main_DPPO.py --group ${group} --use-old-env --use_snrmap --algo ${algo} --dyna_level 4 --update_num 5 --user_data_amount 5 --device cuda:4 >> ${nohup_dir}/0216.log 2>&1 &
nohup python -u main_DPPO.py --group ${group} --use-old-env --use_snrmap --use_snrmap_shortcut --algo ${algo} --dyna_level 4 --update_num 5 --user_data_amount 5 --device cuda:4 >> ${nohup_dir}/0216.log 2>&1 &
done