from operator import truediv
import sndhdr

'''为了导包的目标是本项目中覆写的adept，而不是~/wh/adeptRL'''
import os
import sys

# print('-----------------')
add_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
# print(add_dir)
# print('-----------------')
sys.path = [add_dir] + sys.path

from adept.env.env_ucs.util.utils import *
# from adept.env.env_ucs.env_ucs import EnvUCS
from adept.env.envs.envs import EnvMobile
import gym
import numpy as np
import pandas as pd
import time
import copy
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=False, default='tethered', help="environment(eight/ring/catchup/slowdown/Grid/Monaco/)")
    parser.add_argument('--algo', type=str, required=False, default='DPPO', help="algorithm(DMPO/IC3Net/CPPO/DPPO/IA2C) ")
    parser.add_argument('--device', type=str, required=False, default='cuda:0', help="device(cpu/cuda:0/cuda:1/...) ")
    parser.add_argument('--name', type=str, required=False, default='', help="the additional name for logger")
    parser.add_argument('--para', type=str, required=False, default='{}', help="the hyperparameter json string")

    from arguments import add_args
    parser = add_args(parser)
    args = parser.parse_args()
    args.para = json.loads(args.para.replace('\'', '\"'))

    # == yyx add ==
    if args.dataset == 'purdue':
        args.setting_dir = 'purdue59move'
    elif args.dataset == 'NCSU':
        args.setting_dir = 'NCSU33move'

    if args.debug:
        args.output_dir = 'runs/debug'
        args.Max_train_steps = 100
        args.T_horizon = 10
        args.eval_interval = 50
        args.save_interval = 20
        args.n_rollout_threads = 3

    return args

def yyx_get_env_args(args):
    '''
    这里替换为适配我自己环境的env_config
    '''
    from env_configs.utils.my_utils import fillin_lazy_args, import_env_config, env_config_wrapper
    from env_configs.config.main_config import rllib_env_config as yyx_args
    args = fillin_lazy_args(args, dataset_str=args.dataset)
    my_env_config = import_env_config(args.dataset, args)
    my_env_config = env_config_wrapper(my_env_config, args.num_uv, args.sinr_demand, args.num_serviced_pois, args.uav_height)
    yyx_args['my_env_config'] = my_env_config
    yyx_args['args'] = args

    # hook the params
    tmp_dict = copy.deepcopy(yyx_args)
    tmp_dict['args'] = vars(tmp_dict['args'])
    tmp_dict['setting_dir'] = args.setting_dir
    with open(os.path.join(args.output_dir, 'params.json'), 'w') as f:
        f.write(json.dumps(tmp_dict))
    return yyx_args

def test(data_amount=1, uav_num=3, bar=100, snr=500):
    print('uav_num:{},snr:{},data_amount:{},bar:{}'.format(uav_num, snr, data_amount, bar))
    yyx_args = yyx_get_env_args(parse_args())
    hao_args = {
        'test_mode': True,
        'save_path': '.',
        "controller_mode": True,
        "seed": 1,
        "action_mode": 3,
        "weighted_mode": True,
        "mip_mode": False,
        "noisy_power": -90,
        "tx_power": 20,
        "render_mode": True,
        #
        "user_data_amount": data_amount,
        "uav_num": uav_num,
        "emergency_threshold": bar,
        "collect_range": snr,
        # yyx add
        "max_episode_step": yyx_args['my_env_config']['num_timestep'],
    }
    
    env = EnvMobile(
        hao_args=hao_args, yyx_args=yyx_args
    )
    # env.args.output_dir = run_args.output_dir  # 统一输出路径
    env.phase = 'train'
    new_obs_n = env.reset()
    total = []
    iteration = 0

    done_count = 0
    poi_collect_list = []
    retdict = {'collection_ratio': [], 'violation_ratio': [], 'episodic_aoi': [], 'T-AoI': [], 'consumption_ratio': []}

    for i in range(1):

        episode_step = 0
        episode_rewards = [0.0]  # sum of rewards for all agents
        episode_action = []
        while True:
            actions = []
            for i in range(uav_num):
                a = np.random.randint(9)
                # while new_obs_n['available_actions'][i][a] != 1:  # 这个while根本不会进入~~就是离散的9个动作
                #    a = np.random.randint(13)
                actions.append(a)

            action = {'Discrete': [actions[i] for i in range(uav_num)]}
            new_obs_n, rew_n, done_n, info_n = env.step(action)
            episode_action.append(action)
            obs_n = new_obs_n
            done = done_n
            episode_rewards[-1] += rew_n[0]  # 每一个step的总reward
            episode_step += 1
            if done:
                done_count += 1
                total.append(np.sum(episode_rewards))

                poi_collect_list.append(info_n['a_poi_collect_ratio'])
                retdict['collection_ratio'] = info_n['a_poi_collect_ratio']
                retdict['violation_ratio'] = info_n['b_emergency_violation_ratio']
                retdict['episodic_aoi'] = info_n['e_weighted_aoi']
                retdict['T-AoI'] = info_n['f_weighted_bar_aoi']
                retdict['consumption_ratio'] = info_n['h_energy_consuming_ratio']

                obs_n = env.reset()
                iteration += 1
                break

    print('collection_ratio:', np.mean(retdict['collection_ratio']), 'violation_ratio:', np.mean(retdict['violation_ratio']),
          '\nepisodic_aoi:', np.mean(retdict['episodic_aoi']), 'T-AoI:', np.mean(retdict['T-AoI']), 'consumption_ratio:', np.mean(retdict['consumption_ratio']))
    print('\n')


if __name__ == '__main__':
    np.random.seed(1)  # yyx添加，保证对环境的调试可以复现
    test()
