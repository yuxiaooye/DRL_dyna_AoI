from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import itertools
from algorithms.algo.agent.DPPO import DPPOAgent
from algorithms.models import MLP, CategoricalActor
from torch.optim import Adam

'''输入所有agent的obs，输出表征模块后各agent的obs embedding'''
class G2AEmbedNet(nn.Module):
    def __init__(self, obs_dim, n_agent, device,
                 hidden_dim=64, attention_dim=32, 
                 hard=True, soft=True, tau=0.01):
        super(G2AEmbedNet, self).__init__()
        assert hard or soft, print('G2ANet原文hard和soft都做，我改为只做hard')

        self.n_agent = n_agent
        self.device = device
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim  # only used in soft
        self.hard = hard
        self.soft = soft
        self.tau = tau

        # Encoding
        self.encoding = nn.Linear(obs_dim, self.hidden_dim)  # 对所有agent的obs解码
        # 2.21 现在仅根据这个做hardatt
        self.hard_encoding = nn.Linear(self.hidden_dim * 2, 2)  # 输入的乘2因为是双向GRU，hidden_state维度为2 * hidden_dim


    def forward(self, obs):
        '''
        obs.shape = (threads, n_agent, dim)
        '''

        n_thread = obs.shape[0]
        assert self.n_agent == obs.shape[1]
        size = self.n_agent * n_thread  # 从n_agent改为n_agent*n_thread适配向量环境
        # encoding
        h_out = f.relu(self.encoding(obs))  # (batch_size, n_agent, dim)

        # Hard Attention，GRU和GRUCell不同，输入的维度是(序列长度, batch_size, dim)

        '''Hard Attention前的准备(没过网络只玩维度)'''
        h = h_out
        input_hard = []
        for i in range(self.n_agent):
            h_i = h[:, i]  # (batch_size, hidden_dim)
            h_hard_i = []
            for j in range(self.n_agent):  # 对于agent i，把自己的h_i与其他agent的h分别拼接
                if j != i:
                    h_hard_i.append(torch.cat([h_i, h[:, j]], dim=-1))
            # j 循环结束之后，h_hard_i是一个list里面装着n_agents - 1个维度为(batch_size, hidden_dim * 2)的tensor
            h_hard_i = torch.stack(h_hard_i, dim=0)
            input_hard.append(h_hard_i)
        # i循环结束之后，input_hard是一个list里面装着n_agents个维度为(n_agents - 1, batch_size, hidden_dim * 2)的tensor
        input_hard = torch.stack(input_hard, dim=-2)
        # 最终得到维度(n_agents - 1, batch_size * n_agents, hidden_dim * 2)，可以输入了
        input_hard = input_hard.view(self.n_agent - 1, -1, self.hidden_dim * 2)

        h_hard = input_hard
        h_hard = h_hard.permute(1, 0, 2)  # (batch_size * n_agents, n_agents - 1, hidden_dim * 2)
        h_hard = h_hard.reshape(-1, self.hidden_dim * 2)  # (batch_size * n_agents * (n_agents - 1), hidden_dim * 2)

        hard_weights = self.hard_encoding(h_hard)  # 将(3*2, 128)映射为(3*2, 2)
        a0307 = False
        if a0307:
            a = f.softmax(hard_weights, dim=-1)
            a = a.reshape(n_thread, self.n_agent*(self.n_agent-1), 2)
            # 以softmax后的概率，连或不连
            comm_gate = torch.stack(
                [torch.stack([torch.multinomial(dis, 1).detach() for dis in n_dis]) for n_dis in
                 a])

        else:
            print('In forward(), inspect...')  # TODO DASAP
            print(hard_weights)

            hard_weights = f.gumbel_softmax(hard_weights, tau=self.tau)
        # print(hard_weights)
        hard_weights = hard_weights[:, 1].view(-1, self.n_agent, 1, self.n_agent - 1)
        hard_weights = hard_weights.permute(1, 0, 2, 3)

        return hard_weights


class G2ANetAgent(DPPOAgent):
    def __init__(self, logger, device, agent_args, input_args):
        DPPOAgent.__init__(self, logger, device, agent_args, input_args)

        if input_args.g2a_hidden_dim is None:
            self.hidden_dim = 64
        else:
            self.hidden_dim = input_args.g2a_hidden_dim
        self.attention_dim = 32

        self.g2a_embed_hard_net = G2AEmbedNet(obs_dim=agent_args.observation_dim,
                                         n_agent=agent_args.n_agent,
                                         device=device,
                                         hidden_dim=self.hidden_dim,
                                         attention_dim=self.attention_dim,
                                         soft=False,  # crucial!
                                         tau=self.input_args.tau,
                                         ).to(device)

        pi_dict, v_dict = self.pi_args._toDict(), self.v_args._toDict()
        pi_dict['sizes'][0] = self.observation_dim  # share邻居的obs时做并集而不是concat
        v_dict['sizes'][0] = self.observation_dim
        self.actors = nn.ModuleList()
        self.vs = nn.ModuleList()
        for i in range(self.n_agent):
            self.actors.append(CategoricalActor(**pi_dict).to(self.device))
            self.vs.append(MLP(**v_dict).to(self.device))
        self.optimizer_pi = Adam(self.actors.parameters(), lr=self.lr)
        self.optimizer_v = Adam(self.vs.parameters(), lr=self.lr_v)
        self.optimizer_colla = Adam(self.g2a_embed_hard_net.parameters(), lr=self.lr_colla)

        self.train_saved_hard_att = []  # 记录一个episode里agent间的邻接关系
        self.test_saved_hard_att = []




