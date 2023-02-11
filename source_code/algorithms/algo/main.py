import time
import os
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from algorithms.utils import collect, mem_report
from algorithms.models import GaussianActor, GraphConvolutionalModel, MLP, CategoricalActor
from tqdm.std import trange
# from algorithms.algorithm import ReplayBuffer
from ray.state import actors
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
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
    def __init__(self, logger, run_args, alg_args, agent, env_learn, env_test, env_args, **kwargs):
        self.run_args = run_args
        self.debug = self.run_args.debug
        self.logger = logger
        self.name = run_args.name
        # agent initialization
        self.agent = agent
        self.device = self.agent.device if hasattr(self.agent, "device") else "cpu"

        if run_args.init_checkpoint is not None:  # not train from scratch
            # agent.load(run_args.init_checkpoint)
            self.agent.actors.load_state_dict(torch.load(run_args.init_checkpoint))
            logger.log(interaction=run_args.start_step)
        self.start_step = run_args.start_step
        self.env_name = env_args.env
        self.algo_name = env_args.algo

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
        self.max_episode_len = alg_args.max_episode_len
        self.clip_scheme = None if (not hasattr(alg_args, "clip_scheme")) else alg_args.clip_scheme
        self.debug_use_stack_frame = alg_args.debug_use_stack_frame

        # environment initialization
        self.env_learn = env_learn
        self.env_test = env_test

        # buffer initialization
        self.discrete = agent.discrete
        action_dtype = torch.long if self.discrete else torch.float
        self.model_based = alg_args.model_based
        self.model_batch_size = alg_args.model_batch_size
        if self.model_based:
            self.n_traj = alg_args.n_traj
            self.model_traj_length = alg_args.model_traj_length
            self.model_error_thres = alg_args.model_error_thres
            self.model_buffer = ModelBuffer(alg_args.model_buffer_size)
            self.model_update_length = alg_args.model_update_length
            self.model_validate_interval = alg_args.model_validate_interval
            self.model_length_schedule = alg_args.model_length_schedule
            self.model_prob = alg_args.model_prob
        # 一定注意，PPO并不是在每次调用rollout时reset，一次rollout和是否reset没有直接对应关系
        self.env_learn.reset()
        self.episode_len, self.episode_reward = 0, 0

        # load pretrained env model when model-based
        self.load_pretrained_model = alg_args.load_pretrained_model
        if self.model_based and self.load_pretrained_model:
            self.agent.load_model(alg_args.pretrained_model)

    def run(self):  # 被launcher.py调用的主循环，在内部调用rollout_env或rollout_model
        # 如果model-based且train from scratch的话，先warmup
        if self.model_based and not self.load_pretrained_model:
            for _ in trange(self.n_warmup, desc='warmup'):  # 50
                trajs = self.rollout_env()
                self.model_buffer.storeTrajs(trajs)
            # 参数是更新多少次，每次更新时，从traj采样一个batchsize更新
            self.updateModel(self.n_model_update_warmup)  # Sample trajectories, then shorten them.

        if self.run_args.test:
            ret = self.test()
            print('test episode reward: ', ret)
            return

        for iter in trange(self.n_iter, desc='rollout env'):
            if iter % 50 == 0:
                mean_return = self.test()
                self.agent.save(info=mean_return)  # 这个是保存模型么？
            if iter % 1000 == 0:  # call torch.save to save model
                self.agent.save_nets(dir_name=self.run_args.output_dir, episode=iter)

            trajs = self.rollout_env()  # TO cheak: rollout n_step, maybe multi trajs
            t1 = time.time()
            if self.model_based:
                self.model_buffer.storeTrajs(trajs)
                # train the environment model
                if iter % 10 == 0:
                    self.updateModel()
            t2 = time.time()
            # print('t=', t2 - t1)

            agentInfo = []
            real_trajs = trajs
            for inner in trange(self.n_inner_iter, desc='inner-iter updateAgent'):
                if self.model_based:
                    ## Use the model with a certain probability                  
                    use_model = np.random.uniform() < self.model_prob
                    if use_model:
                        if self.model_length_schedule is not None:
                            trajs = self.rollout_model(real_trajs, self.model_length_schedule(iter))
                        else:
                            trajs = self.rollout_model(real_trajs)
                    else:
                        trajs = trajs
                if self.clip_scheme is not None:
                    info = self.agent.updateAgent(trajs, self.clip_scheme(iter))  # TO cheak: updata
                else:
                    info = self.agent.updateAgent(trajs)
                agentInfo.append(info)
                if self.agent.checkConverged(agentInfo):
                    break
            self.logger.log(inner_iter=inner + 1, iter=iter)

    def test(self):
        """
        The environment should return sth like [n_agent, dim] or [batch_size, n_agent, dim] in either numpy or torch.
        """
        time_t = time.time()
        length = self.test_length
        returns = []
        lengths = []
        episodes = []
        for i in trange(self.n_test, desc='test'):
            episode = []
            done, ep_ret, ep_len = np.array([False]), 0, 0  # done就是把标量包装成array
            env = self.env_test
            env.reset()
            while not (done.any() or (ep_len == length)):  # 测试时限定一个episode最大为length步
                s = env.get_obs_from_outside()
                a = self.agent.act(s).sample()
                if len(a.shape) == 2 and a.shape[0] == 1:  # for IA2C and IC3Net
                    a = a.squeeze(0)
                a = a.detach().cpu().numpy()  # # shape should be (UAV_NUM, )
                s1, r, done, _ = env.step({'Discrete': a.tolist()})
                episode += [(s.tolist(), a.tolist(), r)]
                done = np.array(done)
                ep_ret += sum(r)
                ep_len += 1
                self.logger.log(interaction=None)
            if ep_ret > self.best_test_episode_reward:
                self.best_test_episode_reward = ep_ret
                self.env_test.callback_write_trajs_to_storage(is_newbest=True)
            returns += [ep_ret]
            lengths += [ep_len]
            episodes += [episode]
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
        self.logger.log(test_time=time.time() - time_t)
        return returns.mean()

    def rollout_env(self, length=0):  # 与环境交互得到trajs
        """
        # yyx i了i了，这代码的注释太清晰了！
        The environment should return sth like [n_agent, dim] or [batch_size, n_agent, dim] in either numpy or torch.
        """
        time_t = time.time()
        if length <= 0:
            length = self.rollout_length
        trajs = []
        traj = TrajectoryBuffer(device=self.device)
        start = time.time()
        env = self.env_learn
        for t in range(length):
            s = env.get_obs_from_outside()
            dist = self.agent.act(s)
            a = dist.sample()
            logp = dist.log_prob(a)
            if len(a.shape) == 2 and a.shape[0] == 1:  # for IA2C and IC3Net
                a = a.squeeze(0)
                logp = logp.squeeze(0)
            a = a.detach().cpu().numpy()
            s1, r, done, env_info = env.step({'Discrete': a.tolist()})
            traj.store(s, a, r, s1, [done for _ in range(s.shape[0])], logp)
            episode_r = r
            # if hasattr(env, '_comparable_reward'):
            #     episode_r = env._comparable_reward()
            if episode_r.ndim > 1:
                episode_r = episode_r.mean(axis=0)
            self.episode_reward += episode_r  # episode_r也是个列表 每个uav的奖励先相加
            if self.debug: print('在main中的一个step, r=', episode_r)
            self.episode_len += 1
            self.logger.log(interaction=None)
            if self.episode_len == self.max_episode_len:
                done = np.zeros(done.shape, dtype=np.float32)
            done = np.array(done)
            # 当episode结束或达到规定的最大步长，reset
            if done.any() or (self.episode_len == self.max_episode_len):
                ep_r = self.episode_reward.sum()
                print('train episode reward:', ep_r)
                self.logger.log(episode_reward=ep_r, episode_len=self.episode_len, episode=None)
                if ep_r > self.best_episode_reward:
                    self.best_episode_reward = ep_r
                    self.agent.save_nets(dir_name=self.run_args.output_dir, is_newbest=True)
                    self.env_learn.callback_write_trajs_to_storage(is_newbest=True)
                self.logger.log(collect_ratio=env_info['a_poi_collect_ratio'],
                                violation_ratio=env_info['b_emergency_violation_ratio'],
                                episodic_aoi=env_info['e_weighted_aoi'],
                                threshold_aoi=env_info['f_weighted_bar_aoi'],
                                energy_consuming_ratio=env_info['h_energy_consuming_ratio'],
                                )
                '''执行env的reset'''
                try:
                    _, self.episode_reward, self.episode_len = self.env_learn.reset(), 0, 0
                except Exception as e:
                    raise NotImplementedError
                    # print('reset error!:', e)
                    # _, self.episode_reward, self.episode_len = self.env_learn.reset(), 0, 0
                    # if self.model_based == False:
                    #     trajs += traj.retrieve()
                    #     traj = TrajectoryBuffer(device=self.device)

            if self.model_based and self.episode_len == self.max_episode_len:
                trajs += traj.retrieve()
                traj = TrajectoryBuffer(device=self.device)
        # --------------------------------------------------------------------------------------
        end = time.time()
        # print('time in 1 episode is ', end - start)
        trajs += traj.retrieve(length=self.max_episode_len)
        self.logger.log(env_rollout_time=time.time() - time_t)
        return trajs

    # Use the environment model to collect data
    def rollout_model(self, trajs, length=0):  # 与learned model交互得到rollout
        # input: trajs, len(trajs) = 1, trajs[0]['s'].shape = (5, 3, 238) 3意为num_agent 238意为obs_dim
        # output: 以trajs中s作为起始状态，与learned model进行交互得到的新经验
        # len(trajs) = 4, trajs[0]['s'].shape = (25, 8, 5)
        time_t = time.time()
        n_traj = self.n_traj
        if length <= 0:
            length = self.model_traj_length
        s = [traj['s'] for traj in trajs]

        s = torch.stack(s, dim=0)
        b, T, n, depth = s.shape  # T 意为rollout的T_horizon
        s = s.view(-1, n, depth)
        idxs = torch.randint(low=0, high=b * T, size=(n_traj,), device=self.device)
        s = s.index_select(dim=0, index=idxs)

        trajs = TrajectoryBuffer(device=self.device)
        for _ in range(length):
            # a, logp = self.agent.act(s, requires_log=True)
            dist = self.agent.act(s)
            a = dist.sample()
            logp = dist.log_prob(a)
            r, s1, done, _ = self.agent.model_step(s, a)
            trajs.store(s, a, r, s1, done, logp)
            s = s1
        trajs = trajs.retrieve()
        self.logger.log(model_rollout_time=time.time() - time_t)
        return trajs

    def updateModel(self, n=0):  # 一层封装，在内部调用self.agent.updateModel
        if n <= 0:
            n = self.n_model_update
        for i_model_update in trange(n):
            trajs = self.model_buffer.sampleTrajs(self.model_batch_size)
            trajs = [traj.getFraction(length=self.model_update_length) for traj in trajs]

            # == yyx change 记录论文中fig5的state error ==
            # self.agent.updateModel(trajs, length=self.model_update_length)
            rel_state_error = self.agent.updateModel(trajs, length=self.model_update_length)
            self.logger.log(state_error=rel_state_error)
            # ===========================================

            if i_model_update % self.model_validate_interval == 0:
                validate_trajs = self.model_buffer.sampleTrajs(self.model_batch_size)
                validate_trajs = [traj.getFraction(length=self.model_update_length) for traj in validate_trajs]
                rel_error = self.agent.validateModel(validate_trajs, length=self.model_update_length)
                if rel_error < self.model_error_thres:
                    break
        self.logger.log(model_update=i_model_update + 1)

    def testModel(self, n=0):  # 一层封装，在内部调用self.agent.validateModel
        trajs = self.model_buffer.sampleTrajs(self.model_batch_size)
        trajs = [traj.getFraction(length=self.model_update_length) for traj in trajs]
        return self.agent.validateModel(trajs, length=self.model_update_length)
