import os
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from algorithms.utils import collect, mem_report
from algorithms.models import GaussianActor, GraphConvolutionalModel, MLP, CategoricalActor
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
from algorithms.models import CategoricalActor, EnsembledModel, SquashedGaussianActor, ParameterizedModel_MBPPO
import random
import multiprocessing as mp
# import torch.multiprocessing as mp
from torch import distributed as dist
import argparse
from algorithms.algo.buffer import MultiCollect, Trajectory, TrajectoryBuffer, ModelBuffer


class OnPolicyRunner:
    def __init__(self, logger, agent, envs_learn, envs_test, dummy_env,
                 run_args, alg_args, input_args, **kwargs):
        self.run_args = run_args
        self.input_args = input_args
        self.debug = self.run_args.debug
        self.logger = logger
        self.name = run_args.name
        # agent initialization
        self.agent = agent
        self.num_agent = agent.n_agent
        self.device = self.agent.device if hasattr(self.agent, "device") else "cpu"

        if run_args.init_checkpoint is not None:  # not train from scratch
            # agent.load(run_args.init_checkpoint)
            self.agent.actors.load_state_dict(torch.load(run_args.init_checkpoint))
            logger.log(interaction=run_args.start_step)
        self.start_step = run_args.start_step
        self.env_name = input_args.env
        self.algo_name = input_args.algo
        self.n_thread = input_args.n_thread

        # yyx add
        self.best_episode_reward = float('-inf')
        self.best_test_episode_reward = float('-inf')

        # algorithm arguments
        self.n_iter = alg_args.n_iter
        self.n_inner_iter = alg_args.n_inner_iter
        self.n_warmup = alg_args.n_warmup if not self.run_args.debug else 1
        self.n_model_update = alg_args.n_model_update
        self.n_model_update_warmup = alg_args.n_model_update_warmup if not self.run_args.debug else 1
        self.n_test = alg_args.n_test
        self.test_interval = alg_args.test_interval
        self.rollout_length = alg_args.rollout_length
        self.test_length = alg_args.test_length
        self.use_stack_frame = alg_args.use_stack_frame

        # environment initialization
        self.envs_learn = envs_learn
        self.envs_test = envs_test
        self.dummy_env = dummy_env

        # buffer initialization
        self.model_based = alg_args.model_based
        self.model_batch_size = alg_args.model_batch_size
        if self.model_based:
            self.n_traj = alg_args.n_traj
            self.model_traj_length = alg_args.model_traj_length
            self.model_error_thres = alg_args.model_error_thres
            self.model_buffer = ModelBuffer(alg_args.model_buffer_size)
            self.model_update_length = alg_args.model_update_length
            self.model_validate_interval = alg_args.model_validate_interval
            self.model_prob = alg_args.model_prob
        # 一定注意，PPO并不是在每次调用rollout时reset，一次rollout和是否reset没有直接对应关系
        _, self.episode_len = self.envs_learn.reset(), 0
        # 每个环境分别记录episode_reward
        self.episode_reward = np.zeros((self.input_args.n_thread))

        # load pretrained env model when model-based
        self.load_pretrained_model = alg_args.load_pretrained_model
        if self.model_based and self.load_pretrained_model:
            self.agent.load_model(alg_args.pretrained_model)

    def run(self):  # 被launcher.py调用的主循环
        if self.model_based and not self.load_pretrained_model:  # warm up the model
            for _ in trange(self.n_warmup, desc='warmup'):  # 50
                trajs = self.rollout_env()  # 这些trajs用于warm up更新model后就扔了~
                self.model_buffer.storeTrajs(trajs)
            self.updateModel(self.n_model_update_warmup)

        if self.run_args.test:
            ret = self.test()
            print('test episode reward: ', ret)
            return

        for iter in trange(self.n_iter, desc='rollout env'):
            if iter % 50 == 0:
                mean_return = self.test()
                self.agent.save(info=mean_return)  # yyx的保存模型
            if iter % 1000 == 0:
                self.agent.save_nets(dir_name=self.run_args.output_dir, episode=iter)

            trajs = self.rollout_env()  # model-based三部曲之一
            if self.model_based:
                self.model_buffer.storeTrajs(trajs)
                if iter % 10 == 0:
                    self.updateModel()  # model-based三部曲之二

            agentInfo = []
            for inner in trange(self.n_inner_iter, desc='inner-iter updateAgent'):
                if self.model_based and np.random.uniform() < self.model_prob:  # Use the model with a certain probability
                    trajs = self.rollout_model(trajs)
                info = self.agent.updateAgent(trajs)  # model-based三部曲之三
                agentInfo.append(info)
                if self.agent.checkConverged(agentInfo):
                    break
            self.logger.log(inner_iter=inner + 1, iter=iter)

    def test(self):
        """
        The environment should return sth like [n_agent, dim] or [batch_size, n_agent, dim] in either numpy or torch.
        """
        length = self.test_length
        returns = []
        lengths = []
        # episodes = []
        for i in trange(self.n_test, desc='test'):
            # episode = []
            done, ep_ret, ep_len = False, np.zeros((1,)), 0  # ep_ret改为分threads存
            envs = self.envs_test
            envs.reset()
            while not (done or (ep_len == length)):  # 测试时限定一个episode最大为length步
                s = envs.get_obs_from_outside()
                a = self.agent.act(s).sample()  # shape = (-1, 3)
                # if len(a.shape) == 2 and a.shape[0] == 1:  # for IA2C and IC3Net TODO 向量环境下这个需要改！
                #     a = a.squeeze(0)
                a = a.detach().cpu().numpy()  # # shape should be (UAV_NUM, )
                s1, r, done, _ = envs.step(a.tolist())
                done = done.any()
                ep_ret += r.sum(axis=-1)  # 对各agent的奖励求和
                ep_len += 1
                self.logger.log(interaction=None)
            if ep_ret.max() > self.best_test_episode_reward:
                self.best_test_episode_reward = ep_ret.max()
                best_eval_trajs = self.envs_test.get_saved_trajs()
                self.dummy_env.save_trajs_2(
                    best_eval_trajs[ep_ret.argmax()], phase='test', is_newbest=True)
            returns += [ep_ret.sum()]
            lengths += [ep_len]
        returns = np.stack(returns, axis=0)
        lengths = np.stack(lengths, axis=0)
        self.logger.log(test_episode_reward=returns.mean(),
                        test_episode_len=lengths.mean(), test_round=None)
        # print(returns)
        print(f"{self.n_test} episodes average accumulated reward: {returns.mean()}")

        # with open(f"checkpoints/{self.name}/test.pickle", "wb") as f:
        #     pickle.dump(episodes, f)
        # with open(f"checkpoints/{self.name}/test.txt", "w") as f:
        #     for episode in episodes:
        #         for step in episode:
        #             f.write(f"{step[0]}, {step[1]}, {step[2]}\n")
        #         f.write("\n")
        return returns.mean()

    def rollout_env(self):  # 与环境交互得到trajs
        """
        The environment should return sth like [n_agent, dim] or [batch_size, n_agent, dim] in either numpy or torch.
        """
        trajBuffer = TrajectoryBuffer(device=self.device)
        envs = self.envs_learn
        for t in range(int(self.rollout_length/self.input_args.n_thread)):  # 加入向量环境后，控制总训练步数不变
            s = envs.get_obs_from_outside()
            dist = self.agent.act(s)  # TODO rollout时还是有问题
            a = dist.sample()
            logp = dist.log_prob(a)
            # if len(a.shape) == 2 and a.shape[0] == 1:  # for IA2C and IC3Net  # TODO 向量环境下要改~ a.shape[0]已经不是IA2C和IC3Net会额外添加的batch的维度了，我猜需要维度从0改成1
            #     a = a.squeeze(0)
            #     logp = logp.squeeze(0)
            a = a.detach().cpu().numpy()
            s1, r, done, env_info = envs.step(a.tolist())
            done = done.any()
            trajBuffer.store(s, a, r, s1,
                       np.full((self.n_thread, self.num_agent), done),
                       logp)
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
                    self.best_episode_reward = ep_r.max()
                    self.agent.save_nets(dir_name=self.run_args.output_dir, is_newbest=True)
                    best_train_trajs = self.envs_learn.get_saved_trajs()
                    self.dummy_env.save_trajs_2(
                        best_train_trajs[self.episode_reward.argmax()], phase='train', is_newbest=True)

                self.logger.log(collect_ratio=sum(d['a_poi_collect_ratio'] for d in env_info) / len(env_info),
                                violation_ratio=sum(d['b_emergency_violation_ratio'] for d in env_info) / len(env_info),
                                episodic_aoi=sum(d['e_weighted_aoi'] for d in env_info) / len(env_info),
                                threshold_aoi=sum(d['f_weighted_bar_aoi'] for d in env_info) / len(env_info),
                                energy_consuming_ratio=sum(d['h_energy_consuming_ratio'] for d in env_info) / len(env_info),
                                )
                '''执行env的reset'''
                try:
                    _, self.episode_len = self.envs_learn.reset(), 0
                    self.episode_reward = np.zeros((self.input_args.n_thread))
                except Exception as e:
                    raise NotImplementedError

        return trajBuffer.retrieve()

    # Use the environment model to collect data
    def rollout_model(self, trajs):  # 与model交互
        '''
        # input: trajs, len(trajs) = n_thread
        # output: 选择trajs中一些s为起始状态，与model交互model_traj_length步得到n_traj条新经验
        总结：1.选择与model交互时，完全扔掉与真实环境交互的经验
             2.与model交互得到的经验的维度(b,T)和与真实环境交互时不同
        '''
        s = [traj['s'] for traj in trajs]
        s = torch.stack(s, dim=0)  # shape = (b, T, agent, dim)

        ## 选择trajs中的s
        b, T, n, dim = s.shape
        s = s.view(-1, n, dim)  # shape = (b*T, agent, dim)
        idxs = torch.randint(low=0, high=b * T, size=(self.n_traj,), device=self.device)
        s = s.index_select(dim=0, index=idxs)  # 选择后 shape = (n_traj, agent, dim)

        ## 以s为起始状态与model交互
        trajBuffer = TrajectoryBuffer(device=self.device)
        for _ in range(self.model_traj_length):  # 与环境做交互
            dist = self.agent.act(s)
            a = dist.sample()
            logp = dist.log_prob(a)
            r, s1, done, _ = self.agent.model_step(s, a)
            trajBuffer.store(s, a, r, s1, done, logp)
            s = s1
        return trajBuffer.retrieve()

    def updateModel(self, n=0):  # 一层封装，在内部调用self.agent的同名函数
        if n <= 0:
            n = self.n_model_update
        for i_model_update in trange(n):
            trajs = self.model_buffer.sampleTrajs(self.model_batch_size)
            trajs = [traj.getFraction(length=self.model_update_length) for traj in trajs]

            rel_state_error = self.agent.updateModel(trajs, length=self.model_update_length)
            self.logger.log(state_error=rel_state_error)  # 记录论文中fig5的state error

            if i_model_update % self.model_validate_interval == 0:
                validate_trajs = self.model_buffer.sampleTrajs(self.model_batch_size)
                validate_trajs = [traj.getFraction(length=self.model_update_length) for traj in validate_trajs]
                rel_error = self.agent.validateModel(validate_trajs, length=self.model_update_length)
                if rel_error < self.model_error_thres:  # model拟合的很好的话就不需要再更新model了~
                    break
        self.logger.log(model_update=i_model_update + 1)

    def testModel(self, n=0):  # 一层封装，在内部调用self.agent.validateModel
        trajs = self.model_buffer.sampleTrajs(self.model_batch_size)
        trajs = [traj.getFraction(length=self.model_update_length) for traj in trajs]
        return self.agent.validateModel(trajs, length=self.model_update_length)
