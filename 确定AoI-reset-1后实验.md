- [x] [0217下午 tune是否使用snrmap]

```sh
# !/bin/sh
source activate yyx_adept
group='0217-local-ummap'
nohup python -u main_DPPO.py --group ${group} --aoith 60 --n_thread 16 --device cuda:3 &
nohup python -u main_DPPO.py --group ${group} --aoith 60 --use_snrmap --n_thread 16 --device cuda:3 &
nohup python -u main_DPPO.py --group ${group} --aoith 60 --use_snrmap --use_snrmap_shortcut --n_thread 16 --device cuda:3 &
```



- [ ] [0217晚上 不同算法]

aoith=60 bonus-ratio=0

```shell
# !/bin/sh
source activate yyx_ishen
group='0217-algos'
for algo in IPPO DPPO G2ANet;
do
nohup python -u main_DPPO.py --group ${group} --algo ${algo} --n_thread 16 --device cuda:4 &
done
```



- [ ] [0217晚上 不同的AoIth取值（类似五点图）]

aoith=60 bonus-ratio=0

```shell
# !/bin/sh
source activate yyx_ishen
group='0217-adjust-aoith'
for aoith in 60 50 40 30;
do
nohup python -u main_DPPO.py --group ${group} --aoith ${aoith} --n_thread 16 --device cuda:7 &
done
```



[0217晚上 现在episodic-aoi太大了 把uav高度调低]

aoith=60 这个实验开始加上了tx satis ratio的奖励scale









