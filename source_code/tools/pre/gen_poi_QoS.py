import numpy as np
import os
import argparse

assert os.getcwd().endswith('source_code'), '请将工作路径设为source_code，否则无法将结果存入正确路径'

from env_configs.roadmap_env.roadmap_utils import Roadmap

parser = argparse.ArgumentParser()
parser.add_argument('--dyna_level', type=str, default='')
args = parser.parse_args()

def gen(POI_NUM, T):
    '''目前episode_steps硬编码为120，poi数量硬编码为NCSU'''
    # 分为三部分，一部分逐渐400 -> 200且权重增加，一部分逐渐200 -> 400且权重降低，另一部分固定300不变，权重不变
    # 预期结果：无人机前期重点关注第二类，后期重点关注第一类
    # if args.dyna_level == '1':
    #     case1 = np.linspace(400, 200, T)
    #     case2 = np.linspace(200, 400, T)
    #     case3 = np.ones((T,)) * 300
    # elif args.dyna_level == '2':
    #     case1 = np.linspace(300, 100, T)
    #     case2 = np.linspace(100, 300, T)
    #     case3 = np.ones((T,)) * 200
    # elif args.dyna_level == '3':
    #     case1 = np.linspace(200, 100, T)
    #     case2 = np.linspace(100, 200, T)
    #     case3 = np.ones((T,)) * 150
    # elif args.dyna_level == '4':
    #     case1 = np.linspace(300, 100, T)
    #     case2 = np.linspace(100, 300, T)
    #     case3 = np.ones((T,)) * 100
    raise NotImplementedError

    poi_QoS = np.vstack(
        [np.tile(case1, (10, 1)), np.tile(case2, (10, 1)), np.tile(case3, (13, 1))]
    )  # shape = (POI_NUM, T)
    assert poi_QoS.shape == (POI_NUM, T)
    return poi_QoS


def gen_according_to_cluster(POI_NUM, T):
    rm = Roadmap(dataset='NCSU', poi_num=POI_NUM)
    poi_mat = rm.init_pois(max_episode_step=T)

    # 根据user的初始位置划分case，具体划分规则参见草稿纸
    init_poses = poi_mat[:,0,:]
    case1 = np.linspace(300, 100, T)
    case2 = np.linspace(100, 300, T)
    case3 = np.ones((T,)) * 100
    poi_QoS = []
    count1, count2, count3 = 0, 0, 0
    for id, pos in enumerate(init_poses):
        if pos[0] < rm.max_dis_x/2:
            poi_QoS.append(case1)
            count1 += 1
        elif pos[1] < rm.max_dis_y/2:
            poi_QoS.append(case2)
            count2 += 1
        else:
            poi_QoS.append(case3)
            count3 += 1
    print(count1, count2, count3)
    poi_QoS = np.vstack(poi_QoS)
    assert poi_QoS.shape == (POI_NUM, T)
    return poi_QoS


def gen_spatial_temporal_SNRth(POI_NUM, T):
    # 右下角(case1)从100涨到500，其他地方(case2)从500降到100
    # 预期结果：任务前半部分主要去外面采，任务后半部分主要在右下角采
    case1 = np.linspace(500, 100, T)
    case2 = np.linspace(100, 500, T)

    rm = Roadmap(dataset='NCSU', poi_num=POI_NUM)
    poi_mat = rm.init_pois(max_episode_step=T)  # shape = (33, 121, 2)
    assert poi_mat.shape == (POI_NUM, T+1, 2)

    poi_QoS = np.zeros((POI_NUM, T))
    count1, count2 = 0, 0
    for poi_id in range(POI_NUM):
        for t in range(T):
            pos = poi_mat[poi_id][t]
            if rm.max_dis_x / 2 < pos[0] < rm.max_dis_x and rm.max_dis_y / 2 < pos[1] < rm.max_dis_y:
                count1 += 1
                poi_QoS[poi_id][t] = case1[t]
            else:
                count2 += 1
                poi_QoS[poi_id][t] = case2[t]

    print(count1, count2)
    return poi_QoS

POI_NUM = 33
T = 120
# 将结果存入外存
save_dir = f'envs/NCSU/QoS{T}/poi_QoS{args.dyna_level}.npy'

if args.dyna_level in (1, 2, 3, 4):
    result = gen(POI_NUM, T)
elif args.dyna_level == 'cluster':
    result = gen_according_to_cluster(POI_NUM, T)
elif args.dyna_level == 'SSTSS':  # Same Space Time Same SNRth
    result = gen_spatial_temporal_SNRth(POI_NUM, T)
else:
    raise NotImplementedError
np.save(save_dir, result)
#
# ans = np.load(save_dir)
# print(ans.shape)
# print(ans.mean())
# print(ans.var())
