source activate yyx_adept
group='0217-first-bonus-ratio=0'
for algo in IPPO;
do
nohup python -u main_DPPO.py --group ${group} --algo ${algo} --n_thread 16 --device cuda:4 &
done