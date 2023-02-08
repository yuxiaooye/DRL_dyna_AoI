import numpy as np
import os
import argparse

proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# print(proj_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--dyna_level', type=str, default='')
args = parser.parse_args()

def gen():
    '''目前episode_steps硬编码为120，poi数量硬编码为NCSU'''
    POI_NUM = 33
    T = 120
    # 分为三部分，一部分逐渐400 -> 200且权重增加，一部分逐渐200 -> 400且权重降低，另一部分固定300不变，权重不变
    # 权重改为不读_poi_weight而是就正比于QoS的倒数
    # 预期结果：无人机前期重点关注第二类，后期重点关注第一类
    if args.dyna_level == '':
        case1 = np.linspace(400, 200, T)
        case2 = np.linspace(200, 400, T)
        case3 = np.ones((T,)) * 300
    elif args.dyna_level == '2':
        case1 = np.linspace(300, 100, T)
        case2 = np.linspace(100, 300, T)
        case3 = np.ones((T,)) * 200

    # poi_QoS = np.ones((POI_NUM, T)) * 100
    poi_QoS = np.vstack(
        [np.tile(case1, (10, 1)), np.tile(case2, (10, 1)), np.tile(case3, (13, 1))]
    )  # shape = (33, 120)
    assert poi_QoS.shape == (POI_NUM, T)
    return poi_QoS
r'''在source_code\adept\env\env_ucs\util\Beijing路径下生成poi_QoS.npy，shape=(POI_NUM, T)'''
# print(os.getcwd())
save_dir = os.path.join(proj_dir, f'adept/env/env_ucs/util/NCSU/poi_QoS{args.dyna_level}.npy')
# print(save_dir)
np.save(save_dir, gen())

ans = np.load(save_dir)
print(ans.shape)
print(ans.mean())
print(ans.var())
