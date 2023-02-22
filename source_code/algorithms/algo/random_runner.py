import os
import os.path as osp
from datetime import datetime
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from algorithms.utils import collect, mem_report
from algorithms.models import GraphConvolutionalModel, MLP, CategoricalActor
from tqdm.std import trange
# from algorithms.algorithm import ReplayBuffer
from gym.spaces.box import Box
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import pickle
from copy import deepcopy as dp
from algorithms.models import CategoricalActor
import random
import multiprocessing as mp
# import torch.multiprocessing as mp
from torch import distributed as dist
import argparse
from algorithms.algo.buffer import MultiCollect, Trajectory, TrajectoryBuffer, ModelBuffer
from algorithms.algo.main import OnPolicyRunner


def write_output(info, output_dir, tag='train'):
    logging_path = osp.join(output_dir, f'{tag}_output.txt')
    with open(logging_path, 'a') as f:
        f.write('[' + datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S') + ']\n')
        f.write(f""
                # f"best_{tag}_reward: {'%.3f' % self.best_eval_reward if tag == 'eval' else '%.3f' % self.best_train_reward} "
                f"QoI: {'%.3f' % info['QoI']} "
                f"episodic_aoi: {'%.3f' % info['episodic_aoi']} "
                f"aoi_satis_ratio: {'%.3f' % info['aoi_satis_ratio']} "
                f"data_satis_ratio: {'%.3f' % info['data_satis_ratio']} "
                # f"tx_satis_ratio: {'%.3f' % info['tx_satis_ratio']} "
                # f"soft_tx_satis_ratio: {'%.3f' % info['soft_tx_satis_ratio']} "
                f"energy_consuming: {'%.3f' % info['energy_consuming']} "
                + '\n'
                )


class RandomRunner(OnPolicyRunner):
    def __init__(self, logger, agent, envs_learn, envs_test, dummy_env,
                 run_args, alg_args, input_args, **kwargs):
        super().__init__(logger, agent, envs_learn, envs_test, dummy_env,
                 run_args, alg_args, input_args, **kwargs)

    def test(self):
        pass

    def rollout_env(self, iter):
        """
        The environment should return sth like [n_agent, dim] or [batch_size, n_agent, dim] in either numpy or torch.
        """
        self.routine_count += 1


        envs = self.envs_learn
        for t in range(int(self.rollout_length / self.input_args.n_thread)):  # 加入向量环境后，控制总训练步数不变
            a = self.agent.act()
            _, r, done, env_info = envs.step(a.tolist())
            done = done.any()

            episode_r = r
            assert episode_r.ndim > 1
            episode_r = episode_r.sum(axis=-1)  # 对各agent奖励求和
            self.episode_reward += episode_r
            self.episode_len += 1
            self.logger.log(interaction=None)

            if done:
                ep_r = self.episode_reward
                print('train episode reward:', ep_r)
                self.logger.log(mean_episode_reward=ep_r.mean(), episode_len=self.episode_len, episode=None)
                self.logger.log(max_episode_reward=ep_r.max(), episode_len=self.episode_len, episode=None)
                if ep_r.max() > self.best_episode_reward:
                    max_id = ep_r.argmax()
                    self.best_episode_reward = ep_r.max()
                    self.agent.save_nets(dir_name=self.run_args.output_dir, is_newbest=True)
                    best_train_trajs = self.envs_learn.get_saved_trajs()
                    poi_aoi_history = self.envs_learn.get_poi_aoi_history()
                    serves = self.envs_learn.get_serves()
                    write_output(env_info[max_id], self.run_args.output_dir)
                    self.dummy_env.save_trajs_2(
                        best_train_trajs[max_id], poi_aoi_history[max_id], serves[max_id], phase='train', is_newbest=True)

                '''执行env的reset'''
                try:
                    _, self.episode_len = self.envs_learn.reset(), 0
                    self.episode_reward = np.zeros((self.input_args.n_thread))
                except Exception as e:
                    raise NotImplementedError


        return None

