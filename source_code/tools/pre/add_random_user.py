import pandas as pd
from env_configs.roadmap_env.roadmap_utils import Roadmap
import osmnx as ox
import numpy as np
from numpy.random import rand, uniform

poi_num = 50
default_poi_num = 33
T = 121

rm = Roadmap('NCSU', poi_num=default_poi_num)
poi_df = pd.read_csv(r'F:\PycharmProjects\jsac\DRL_dyna_AoI\source_code\envs\NCSU\human120.csv')

G = ox.load_graphml(r'F:\PycharmProjects\ICDE\roadmap\data\NCSU\walk\map.graphml')
X, Y = rm.max_dis_x, rm.max_dis_y
for i in range(default_poi_num, poi_num):  # 新生成的user
    src = np.array([rand() * X, rand() * Y])
    src_node = ox.distance.nearest_nodes(G, *rm.pygamexy2lonlat(*src))
    lons, lats = [], []
    r = 24  # 人步行速度1.2m/s
    theta = uniform(0, 2*np.pi)
    for t in range(T):

        # dst = src + [uniform(-1, 1)*20, uniform(-1, 1)*20]  # 太二了，原地周围随机打转
        dst = src + [r*np.cos(theta), r*np.sin(theta)]
        dst = np.clip(dst, [0, 0], [X, Y])  # 出界检测
        dst_node = ox.distance.nearest_nodes(G, *rm.pygamexy2lonlat(*dst))
        lon, lat = G.nodes[dst_node]['x'], G.nodes[dst_node]['y']
        lons.append(lon)
        lats.append(lat)
        src = dst
        if rand() < 0.2:  # 以一定概率改变运动朝向
            theta = uniform(0, 2 * np.pi)
    pxs, pys = rm.lonlat2pygamexy(np.array(lons), np.array(lats))
    print(1)

    subdf = pd.DataFrame()
    subdf['id'] = [i for _ in range(T)]
    subdf['px'], subdf['py'] = pxs, pys
    poi_df = poi_df.append(subdf)

assert poi_df.shape[0] == poi_num * T
poi_df.to_csv(rf'F:\PycharmProjects\jsac\DRL_dyna_AoI\source_code\envs\NCSU\human120-user{poi_num}.csv')



