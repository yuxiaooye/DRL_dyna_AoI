# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 02:09:45 2022

@author: 86153
"""

import time
import os
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from algorithms.utils import collect, mem_report
from algorithms.models import GaussianActor, GraphConvolutionalModel, MLP, CategoricalActor, YyxMLPModel
from tqdm.std import trange
# from algorithms.algorithm import ReplayBuffer

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
from torch import distributed as dist
import argparse
from algorithms.algo.agent.DPPO import DPPOAgent
from algorithms.algo.buffer import MultiCollect, Trajectory, TrajectoryBuffer, ModelBuffer


class ModelBasedAgent(nn.ModuleList):
    def __init__(self, logger, device, agent_args, input_args, **kwargs):
        super().__init__(logger, device, agent_args, input_args, **kwargs)
        self.input_args = input_args
        self.logger = logger
        self.device = device
        self.lr_p = agent_args.lr_p
        self.p_args = agent_args.p_args

        if self.input_args.use_mlp_model:
            self.ps = YyxMLPModel(self.logger, self.observation_dim,
                                  self.p_args, multi_mlp=input_args.multi_mlp).to(self.device)
        else:
            self.ps = GraphConvolutionalModel(self.logger, self.adj, self.observation_dim, self.action_dim, self.n_agent, self.p_args).to(self.device)

        # 2.12凌晨做的mlp-model实验，根本没优化model！
        self.optimizer_p = Adam(self.ps.parameters(), lr=self.lr)

    def updateModel(self, trajs, length=1):
        """
        Input dim: 
        s: [[T, n_agent, state_dim]]
        a: [[T, n_agent, action_dim]]
        """

        time_t = time.time()
        ss, actions, rs, s1s, ds = [], [], [], [], []
        for traj in trajs:
            s, a, r, s1, d = traj["s"], traj["a"], traj["r"], traj["s1"], traj["d"]
            s, a, r, s1, d = [torch.as_tensor(item, device=self.device) for item in [s, a, r, s1, d]]
            ss.append(s)
            actions.append(a)
            rs.append(r)
            s1s.append(s1)
            ds.append(d)

        ss, actions, rs, s1s, ds = [torch.stack(item, dim=0) for item in [ss, actions, rs, s1s, ds]]
        if self.input_args.use_mlp_model:
            loss, rel_state_error = self.ps.train(ss, actions, rs, s1s, length)  # [n_traj, T, n_agent, dim]
        else:
            loss, rel_state_error = self.ps.train(ss, actions, rs, s1s, ds, length)  # [n_traj, T, n_agent, dim]
        self.optimizer_p.zero_grad()
        loss.sum().backward()
        # torch.nn.utils.clip_grad_norm_(parameters=self.ps.parameters(), max_norm=5, norm_type=2)
        self.optimizer_p.step()
        # ———————————————————————————————————————————————————————————————————————————————————
        self.logger.log(p_loss_total=loss.sum(), p_update=None)
        self.logger.log(model_update_time=time.time() - time_t)
        # ——————————————————————————————————————————————————————————————————————————————————
        return rel_state_error.item()

    def validateModel(self, trajs, length=1):
        with torch.no_grad():
            ss, actions, rs, s1s, ds = [], [], [], [], []
            for traj in trajs:
                s, a, r, s1, d = traj["s"], traj["a"], traj["r"], traj["s1"], traj["d"]
                s, a, r, s1, d = [torch.as_tensor(item, device=self.device) for item in [s, a, r, s1, d]]
                ss.append(s)
                actions.append(a)
                rs.append(r)
                s1s.append(s1)
                ds.append(d)
            ss, actions, rs, s1s, ds = [torch.stack(item, dim=0) for item in [ss, actions, rs, s1s, ds]]
            if self.input_args.use_mlp_model:
                _, rel_state_error = self.ps.train(ss, actions, rs, s1s, length)
            else:
                _, rel_state_error = self.ps.train(ss, actions, rs, s1s, ds, length)  # [n_traj, T, n_agent, dim]
            return rel_state_error.item()

    def model_step(self, s, a):
        """
        Input dim: 
        s: [batch_size, n_agent, state_dim]
        a: [batch_size, n_agent] (discrete) or [batch_size, n_agent, action_dim] (continuous)

        Return dim == 3.
        """
        with torch.no_grad():
            while s.dim() <= 2:
                s = s.unsqueeze(0)
                a = a.unsqueeze(0)
            while a.dim() <= 2:
                a = a.unsqueeze(-1)
            s = s.to(self.device)
            a = a.to(self.device)
            if self.input_args.use_mlp_model:
                rs, s1s = self.ps(s, a)
                ds = torch.full((rs.shape), False)  # MLP不预测done，先硬编码为False
            else:
                rs, s1s, ds = self.ps.predict(s, a)

            return rs.detach(), s1s.detach(), ds.detach(), s.detach()

    def load_model(self, pretrained_model):
        # pretrained_model = 
        # 'D:/A_RL/MB-MARL/checkpoints/standard _CACC_slowdown_MB_DPPOAgent_23243/2572680_-2813.359473800659.pt'
        dic = torch.load(pretrained_model)
        self.load_state_dict(dic[''])


# yyx wow, 继承自两个父类，有的学习了
class DMPOAgent(ModelBasedAgent, DPPOAgent):
    def __init__(self, logger, device, agent_args, input_args, **kwargs):
        super().__init__(logger, device, agent_args, input_args, **kwargs)

    def checkConverged(self, ls_info):
        # DMPO的这个函数只有意义的，跟DPPO不一样~
        rs = [info[0] for info in ls_info]
        r_converged = len(rs) > 8 and np.mean(rs[-3:]) < np.mean(rs[:-5])
        entropies = [info[1] for info in ls_info]
        entropy_converged = len(entropies) > 8 and np.abs(np.mean(entropies[-3:]) / np.mean(entropies[:-5]) - 1) < 1e-2
        kls = [info[2] for info in ls_info]
        kl_exceeded = False
        if self.target_kl is not None:
            kls = [kl > 1.5 * self.target_kl for kl in kls]
            kl_exceeded = any(kls)
        return kl_exceeded or r_converged and entropy_converged


