# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 02:04:27 2022

@author: 86153
"""

import time
import os
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from algorithms.utils import collect, mem_report
from algorithms.models import MLP
from algorithms.algo.yyx_agent_base import YyxAgentBase
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
from torch import distributed as dist
import argparse
from algorithms.algo.buffer import MultiCollect, Trajectory, TrajectoryBuffer, ModelBuffer


class DPPOAgent(nn.ModuleList, YyxAgentBase):
    """Everything in and out is torch Tensor."""

    def __init__(self, logger, device, agent_args, input_args):
        super().__init__()
        self.input_args = input_args

        self.logger = logger  # LogClient类对象
        self.device = device
        self.n_agent = agent_args.n_agent
        self.gamma = agent_args.gamma
        self.lamda = agent_args.lamda
        self.clip = agent_args.clip
        self.target_kl = agent_args.target_kl
        self.v_coeff = agent_args.v_coeff
        self.v_thres = agent_args.v_thres
        self.entropy_coeff = agent_args.entropy_coeff
        self.lr = agent_args.lr  # 5e-5
        self.lr_v = agent_args.lr_v  # 5e-4
        self.n_update_v = agent_args.n_update_v
        self.n_update_pi = agent_args.n_update_pi
        self.n_minibatch = agent_args.n_minibatch
        self.use_reduced_v = agent_args.use_reduced_v

        self.use_rtg = agent_args.use_rtg
        self.use_gae_returns = agent_args.use_gae_returns
        self.env_name = input_args.env
        self.algo_name = input_args.algo
        self.advantage_norm = agent_args.advantage_norm
        self.observation_dim = agent_args.observation_dim
        if input_args.use_stack_frame:
            self.observation_dim *= 4
        self.action_space = agent_args.action_space
        # self.action_dim = sum([dim.n for dim in self.action_space])

        self.adj = torch.as_tensor(agent_args.adj, device=self.device, dtype=torch.float)
        self.radius_v = agent_args.radius_v
        self.radius_pi = agent_args.radius_pi
        self.pi_args = agent_args.pi_args
        self.v_args = agent_args.v_args

        if input_args.algo not in ('G2ANet', 'G2ANet2', 'IPPO'):

            self.collect_pi, self.actors = self._init_actors()  # collect_pi和collect_v应该一样吧？
            self.collect_v, self.vs = self._init_vs()
            self.optimizer_v = Adam(self.vs.parameters(), lr=self.lr_v)
            self.optimizer_pi = Adam(self.actors.parameters(), lr=self.lr)

    def get_networked_s(self, s, which_net):
        # s [1,3,106]
        if self.input_args.algo == 'IPPO':
            return s
        if self.input_args.algo == 'G2ANet2':
            return self.g2a_embed_net(s)
        if self.input_args.algo == 'G2ANet':  # 拿到hard_att的邻居图后，根据邻居图对obs取并集
            hard_atts = self.g2a_embed_hard_net(s)
            hard_atts = hard_atts.squeeze(-2)  # shape = (agent, n_thread, agent-1)
            n_thread = hard_atts.shape[1]
            ans_s = []
            for i in range(self.n_agent):
                hard_att = hard_atts[i]
                # 添加对自己的hard-att一定是1的维度
                hard_att = torch.cat([hard_att[:, :i], torch.ones(n_thread, 1).to(self.device), hard_att[:, i:]], dim=-1) #[1,3]
                shared_s = s * hard_att.unsqueeze(-1)  # mask后仅邻居的状态可见
                shared_s = torch.sum(shared_s, dim=1)
                shared_s = torch.where(s[:,i] > 0, s[:, i], shared_s)  # 取并集
                ans_s.append(shared_s)
            return ans_s

        assert which_net in ('pi', 'v')
        s = self.collect_pi.gather(s) if which_net == 'pi' else self.collect_v.gather(s)

        return s

    def s_for_agent(self, s, i):
        if self.input_args.algo in ('G2ANet2', 'IPPO'):
            return s[:, i, :]
        return s[i]

    def act(self, s):
        """
        非向量环境：Requires input of [batch_size, n_agent, dim] or [n_agent, dim].
        向量环境：Requires input of [batch_size*n_thread, n_agent, dim] or [n_thread, n_agent, dim].
        其中第一维度的值在后续用-1表示
        This method is gradient-free. To get the gradient-enabled probability information, use get_logp().
        Returns a distribution with the same dimensions of input.
        """
        with torch.no_grad():
            assert s.dim() == 3
            s = s.to(self.device)
            s = self.get_networked_s(s, which_net='pi')
            # Now s[i].dim() == ([-1, dim]) 注意不同agent的dim不同，由它的邻居数量决定
            probs = []
            for i in range(self.n_agent):
                probs.append(self.actors[i](self.s_for_agent(s, i)))
            
            probs = torch.stack(probs, dim=1)  # shape = (-1, NUM_AGENT, act_dim1+act_dim2)    
            return {
                'branch1':Categorical(probs[:,:,0:9]),
                'branch2':Categorical(probs[:,:,9:])
            }

    def get_logp(self, s, a):
        """
        Requires input of [batch_size, n_agent, dim] or [n_agent, dim].
        Returns a tensor whose dim() == 3.
        """
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)

        while s.dim() <= 2:
            s = s.unsqueeze(0)
            a = a.unsqueeze(0)
        while a.dim() < s.dim():
            a = a.unsqueeze(-1)

        s = self.get_networked_s(s, which_net='pi')
        # Now s[i].dim() == 2, a.dim() == 3
        log_prob = []
        for i in range(self.n_agent):
            probs = self.actors[i](self.s_for_agent(s, i)) # [320,2,9]

            index1 = torch.select(a, dim=1, index=i).long()[:,0]
            ans1 = torch.log(torch.gather(probs[:,:9], dim=-1, index=index1.unsqueeze(-1))) # [320,2,1]
            index2 = torch.select(a, dim=1, index=i).long()[:,1]
            ans2 = torch.log(torch.gather(probs[:,9:], dim=-1, index=index2.unsqueeze(-1))) # [320,2,1]
            ans = torch.cat([ans1,ans2],dim=-1)
            
            log_prob.append(ans.squeeze(-1))
        log_prob = torch.stack(log_prob, dim=1)
        while log_prob.dim() < 3:
            log_prob = log_prob.unsqueeze(-1)
        return log_prob

    def _evalV(self, s):
        # yyx：在这个函数中没找到DMPO论文的式9呀？在MultiCollect中~
        # Requires input in shape [-1, n_agent, dim]
        s = s.to(self.device)
        # s变为有n_agent个元素的列表 且元素.shape = [-1, 邻域agent数量*dim]
        # 各个agent的元素.shape不相同！因为邻域规模不同
        # 得到的是s_{N_j}
        s = self.get_networked_s(s, which_net='v')
        values = []
        for i in range(self.n_agent):
            values.append(self.vs[i](self.s_for_agent(s, i)))
        # 填充后，values是有n_agent个元素的列表 元素.shape = (-1, 1)
        # 得到的是V_j(s_{N_j}) 也即式(6)
        return torch.stack(values, dim=1)

    def updateAgent(self, trajs, clip=None):
        time_t = time.time()
        if clip is None:
            clip = self.clip
        n_minibatch = self.n_minibatch

        names = Trajectory.names()  # ['s', 'a', 'r', 's1', 'd', 'logp']
        traj_all = {name: [] for name in names}
        for traj in trajs:  # len(trajs) == n_thread
            for name in names:
                traj_all[name].append(traj[name])
        traj = {name: torch.stack(value, dim=0) for name, value in traj_all.items()}

        for i_update in range(self.n_update_pi):
            s, a, r, s1, d, logp = traj['s'], traj['a'], traj['r'], traj['s1'], traj['d'], traj['logp']
            s, a, r, s1, d, logp = [item.to(self.device) for item in [s, a, r, s1, d, logp]]
            # 关键：all in shape [n_thread, T, n_agent, dim]
            value_old, returns, advantages, reduced_advantages = self._process_traj(**traj)  # 不同traj分开计算adv和return
            advantages_old = reduced_advantages if self.use_reduced_v else advantages

            _, T, n, d_s = s.size()
            d_a = a.size()[-1]
            s = s.view(-1, n, d_s)
            a = a.view(-1, n, d_a)
            logp = logp.view(-1, n, d_a)
            advantages_old = advantages_old.view(-1, n, 1)
            returns = returns.view(-1, n, 1)
            value_old = value_old.view(-1, n, 1)
            # 关键：s, a, logp, adv, ret, v are now all in shape [-1, n_agent, dim] 因为计算完adv和return后可以揉在一起做mini_batch训练
            batch_total = logp.size()[0]
            batch_size = int(batch_total / n_minibatch)

            kl_all = []
            i_pi = 0
            for i_pi in range(1):
                batch_state, batch_action, batch_logp, batch_advantages_old = [s, a, logp, advantages_old]
                if n_minibatch > 1:
                    idxs = np.random.choice(range(batch_total), size=batch_size, replace=False)
                    [batch_state, batch_action, batch_logp, batch_advantages_old] = [item[idxs] for item in [batch_state, batch_action, batch_logp, batch_advantages_old]]
                batch_logp_new = self.get_logp(batch_state, batch_action)

                logp_diff = batch_logp_new.sum(-1, keepdim=True) - batch_logp.sum(-1, keepdim=True)
                kl = logp_diff.mean()  # 这里魔改的，不一定对
                ratio = torch.exp(logp_diff)
                surr1 = ratio * batch_advantages_old
                surr2 = ratio.clamp(1 - clip, 1 + clip) * batch_advantages_old
                loss_surr = torch.min(surr1, surr2).mean()
                loss_entropy = - torch.mean(batch_logp_new)
                loss_pi = - loss_surr - self.entropy_coeff * loss_entropy
                self.optimizer_pi.zero_grad()
                loss_pi.backward()
                self.optimizer_pi.step()
                self.logger.log(surr_loss=loss_surr, entropy=loss_entropy, kl_divergence=kl, pi_update=None)
                kl_all.append(kl.abs().item())
                if self.target_kl is not None and kl.abs() > 1.5 * self.target_kl:
                    break
            self.logger.log(pi_update_step=i_update)

            for i_v in range(1):
                batch_returns = returns
                batch_state = s
                if n_minibatch > 1:
                    idxs = np.random.randint(0, len(batch_total), size=batch_size)
                    [batch_returns, batch_state] = [item[idxs] for item in [batch_returns, batch_state]]
                batch_v_new = self._evalV(batch_state)
                loss_v = ((batch_v_new - batch_returns) ** 2).mean()
                self.optimizer_v.zero_grad()
                loss_v.backward()
                self.optimizer_v.step()

                var_v = ((batch_returns - batch_returns.mean()) ** 2).mean()
                rel_v_loss = loss_v / (var_v + 1e-8)
                self.logger.log(v_loss=loss_v, v_update=None, v_var=var_v, rel_v_loss=rel_v_loss)
                if rel_v_loss < self.v_thres:
                    break
            self.logger.log(v_update_step=i_update)
            self.logger.log(update=None, reward=r, value=value_old, clip=clip, returns=returns, advantages=advantages_old.abs())
        self.logger.log(agent_update_time=time.time() - time_t)
        return [r.mean().item(), loss_entropy.item(), max(kl_all)]

    def checkConverged(self, ls_info):
        return False

    def save(self, info=None):
        self.logger.save(self, info=info)

    def load(self, state_dict):
        self.load_state_dict(state_dict[self.logger.prefix])

    def load_nets(self, dir_name, episode):
        # yyx: can also add argument is_newbest, like save_nets()
        self.actors.load_state_dict(torch.load(dir_name + '/Models/' + str(episode) + 'best_actor.pt'))


    def _init_actors(self):
        # collect_pi.degree = [2,3,2] 意为一个agent与多少个agent连接（包括自己）
        collect_pi = MultiCollect(torch.matrix_power(self.adj, self.radius_pi), device=self.device)
        actors = nn.ModuleList()
        for i in range(self.n_agent):
            self.pi_args.sizes[0] = collect_pi.degree[i] * self.observation_dim
            actors.append(CategoricalActor(**self.pi_args._toDict()).to(self.device))

        return collect_pi, actors

    def _init_vs(self):
        collect_v = MultiCollect(torch.matrix_power(self.adj, self.radius_v), device=self.device)
        vs = nn.ModuleList()
        for i in range(self.n_agent):
            # 确认网络的输入维度 其中.degree[i]是agent i的邻域规模，标量
            self.v_args.sizes[0] = collect_v.degree[i] * self.observation_dim
            vs.append(MLP(**self.v_args._toDict()).to(self.device))
        return collect_v, vs

    def _process_traj(self, s, a, r, s1, d, logp):
        # 过网络得到value_old， 使用GAE计算adv和return
        """
        Input are all in shape [batch_size, T, n_agent, dim]
        """
        b, T, n, dim_s = s.shape
        s, a, r, s1, d, logp = [item.to(self.device) for item in [s, a, r, s1, d, logp]]
        # 过网络前先merge前两个维度，过网络后再复原
        value = self._evalV(s.view(-1, n, dim_s)).view(b, T, n, -1)  # 在_evalV中实现了具体的扩展值函数逻辑
        returns = torch.zeros(value.size(), device=self.device)
        deltas, advantages = torch.zeros_like(returns), torch.zeros_like(returns)
        prev_value = self._evalV(s1.select(1, T - 1))
        if not self.use_rtg:
            prev_return = prev_value
        else:
            prev_return = torch.zeros_like(prev_value)
        prev_advantage = torch.zeros_like(prev_return)
        d_mask = d.float()
        for t in reversed(range(T)):
            deltas[:, t, :, :] = r.select(1, t) + self.gamma * (1 - d_mask.select(1, t)) * prev_value - value.select(1, t).detach()
            advantages[:, t, :, :] = deltas.select(1, t) + self.gamma * self.lamda * (1 - d_mask.select(1, t)) * prev_advantage
            if self.use_gae_returns:
                returns[:, t, :, :] = value.select(1, t).detach() + advantages.select(1, t)
            else:
                returns[:, t, :, :] = r.select(1, t) + self.gamma * (1 - d_mask.select(1, t)) * prev_return

            prev_return = returns.select(1, t)
            prev_value = value.select(1, t)
            prev_advantage = advantages.select(1, t)
        if self.advantage_norm:
            advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / (advantages.std(dim=1, keepdim=True) + 1e-5)
        if self.input_args.algo in ('G2ANet', 'IPPO'):  # 用G2A或者IPPO时，没必要用reduced_adv，也即邻域的adv的均值
            return value.detach(), returns, advantages.detach(), None
        else:
            reduced_advantages = self.collect_v.reduce_sum(advantages.view(-1, n, 1)).view(advantages.size())
            if reduced_advantages.size()[1] > 1:
                reduced_advantages = (reduced_advantages - reduced_advantages.mean(dim=1, keepdim=True)) / (reduced_advantages.std(dim=1, keepdim=True) + 1e-5)
            return value.detach(), returns, advantages.detach(), reduced_advantages.detach()


