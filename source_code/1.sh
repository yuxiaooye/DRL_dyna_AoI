# !/bin/sh
source activate yyx_adept
group_postfix='dynaQoS-0208-level2and3'
nohup_dir='../nohup_log'
mkdir -p ${nohup_dir}
for dyna_level in 2 3;
do
for algo in 'DMPO' 'DPPO' 'CPPO' 'IC3Net' 'IA2C';
do
nohup python -u main_DPPO.py --group_postfix ${group_postfix} --dyna_level ${dyna_level} --algo ${algo} >> ${nohup_dir}/0208.log 2>&1 &
done
done