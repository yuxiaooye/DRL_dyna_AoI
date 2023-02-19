# !/bin/sh
source activate yyx_ishen
group='0219-tune-mapsize'
for dataset in 'NCSU' 'KAIST';
do
for map_size in 4 6 8 10;
do
if [ $dataset == 'NCSU' ]
then
nohup python -u main_DPPO.py --group ${group} --poi_num 50 --use_snrmap --map_size ${map_size} --user yyx --device cuda:0 &
else
nohup python -u main_DPPO.py --group ${group} --dataset ${dataset} --poi_num 122 --use_snrmap --map_size ${map_size} --user yyx --device cuda:1 &
fi
done
done