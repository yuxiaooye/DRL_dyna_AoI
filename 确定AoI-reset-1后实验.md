【2.17下午 在最新的场景上对比是否使用snrmap shortcut】

G2ANet的shortcut暂时没跑通 有个维度的问题 问题不大

```sh
# !/bin/sh
source activate yyx_ishen
group='0217-snrmap-shortcut'
nohup_dir='../nohup_log'
mkdir -p ${nohup_dir}
for algo in IPPO DPPO;
do
nohup python -u main_DPPO.py --group ${group} --use-old-env --use_snrmap --algo ${algo} --dyna_level 4 --update_num 5 --user_data_amount 5 --device cuda:4 >> ${nohup_dir}/0216.log 2>&1 &
nohup python -u main_DPPO.py --group ${group} --use-old-env --use_snrmap --use_snrmap_shortcut --algo ${algo} --dyna_level 4 --update_num 5 --user_data_amount 5 --device cuda:4 >> ${nohup_dir}/0216.log 2>&1 &
done
```





![image-20230217154345883](https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202302171543967.png)