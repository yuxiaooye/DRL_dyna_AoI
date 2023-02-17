source activate yyx_ishen
group='0217-first'
for algo in IPPO DPPO G2ANet;
do
nohup python -u main_DPPO.py --group ${group} --dataset KAIST --algo ${algo} --device cuda:2 &
done