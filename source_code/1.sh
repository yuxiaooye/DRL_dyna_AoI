# !/bin/sh
source activate yyx_ishen
group='0218-algos'
for algo in CPPO;
do
nohup python -u main_DPPO.py --group ${group} --n_thread 16 --algo ${algo} --device cuda:7 &
done