# !/bin/sh
source activate yyx_ishen
group='0218-NCSU50user-tuneG2A'
for g2a_hidden_dim in 32 64 96 128;
do
nohup python -u main_DPPO.py --group ${group} --poi_num 50 --algo G2ANet --g2a_hidden_dim ${g2a_hidden_dim}  --user yyx --device cuda:1 &
done