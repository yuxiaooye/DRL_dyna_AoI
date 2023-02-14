import numpy as np
import os
import argparse

assert os.getcwd().endswith('source_code'), '请将工作路径设为source_code，否则无法将结果存入正确路径'

parser = argparse.ArgumentParser()
parser.add_argument('--dyna_level', type=str, default='')
args = parser.parse_args()

def gen(POI_NUM, T):
    '''目前episode_steps硬编码为120，poi数量硬编码为NCSU'''
    # 分为三部分，一部分逐渐400 -> 200且权重增加，一部分逐渐200 -> 400且权重降低，另一部分固定300不变，权重不变
    # 预期结果：无人机前期重点关注第二类，后期重点关注第一类
    if args.dyna_level == '1':
        case1 = np.linspace(400, 200, T)
        case2 = np.linspace(200, 400, T)
        case3 = np.ones((T,)) * 300
    elif args.dyna_level == '2':
        case1 = np.linspace(300, 100, T)
        case2 = np.linspace(100, 300, T)
        case3 = np.ones((T,)) * 200
    elif args.dyna_level == '3':
        case1 = np.linspace(200, 100, T)
        case2 = np.linspace(100, 200, T)
        case3 = np.ones((T,)) * 150
    elif args.dyna_level == '4':
        case1 = np.linspace(300, 100, T)
        case2 = np.linspace(100, 300, T)
        case3 = np.ones((T,)) * 100
    else:
        raise NotImplementedError

    poi_QoS = np.vstack(
        [np.tile(case1, (10, 1)), np.tile(case2, (10, 1)), np.tile(case3, (13, 1))]
    )  # shape = (POI_NUM, T)
    assert poi_QoS.shape == (POI_NUM, T)
    return poi_QoS


POI_NUM = 33
T = 120
# 将结果存入外存
save_dir = f'envs/NCSU/QoS{T}/poi_QoS{args.dyna_level}.npy'
np.save(save_dir, gen(POI_NUM, T))

ans = np.load(save_dir)
print(ans.shape)
print(ans.mean())
print(ans.var())
