# !/bin/sh
source activate yyx_adept
group='0218-NCSUmoreuser-ablation'
nohup python -u main_DPPO.py --algo IPPO --group ${group} --user yyx --device cuda:2 &
nohup python -u main_DPPO.py --algo IPPO --use_snrmap --group ${group} --user yyx --device cuda:2 &
nohup python -u main_DPPO.py --algo G2ANet --group ${group} --user yyx --device cuda:2 &
nohup python -u main_DPPO.py --algo G2ANet --use_snrmap --group ${group} --user yyx --device cuda:2 &