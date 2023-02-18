# !/bin/sh
source activate yyx_adept
group='0218-NCSU50user-ablation'
nohup python -u main_DPPO.py --poi_num 50 --algo IPPO --group ${group} --user yyx --device cuda:1 &
nohup python -u main_DPPO.py --poi_num 50 --algo IPPO --use_snrmap --group ${group} --user yyx --device cuda:1 &
nohup python -u main_DPPO.py --poi_num 50 --algo G2ANet --group ${group} --user yyx --device cuda:2 &
nohup python -u main_DPPO.py --poi_num 50 --algo G2ANet --use_snrmap --group ${group} --user yyx --device cuda:2 &