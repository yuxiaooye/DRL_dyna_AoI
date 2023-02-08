# !/bin/sh
source activate yyx_adept
group_postfix='dynaQoS-0208-level2'
nohup_dir='nohup_log'
mkdir -p ${nohup_dir}
for algo in 'DMPO' 'DPPO' 'CPPO' 'IC3Net' 'IA2C';
do
nohup python -u yyx_DMPO_sourcecode/main_DPPO.py --group_postfix ${group_postfix} --dyna_level 2 --algo ${algo} >> ${nohup_dir}/0208.log 2>&1 &
done