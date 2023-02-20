from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
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

        # Hard
        # GRU输入[[h_i,h_1],[h_i,h_2],...[h_i,h_n]]与[0,...,0]，输出[[h_1],[h_2],...,[h_n]]与[h_n]， h_j表示了agent j与agent i的关系
        # 输入的iputs维度为(n_agents - 1, batch_size * n_agents, hidden_dim * 2)，
        # 即对于batch_size条数据，输入每个agent与其他n_agents - 1个agents的hidden_state的连接
        self.hard_bi_GRU = nn.GRU(self.hidden_dim * 2, self.hidden_dim, bidirectional=True)
        # 对h_j进行分析，得到agent j对于agent i的权重，输出两维，经过gumble_softmax后取其中一维即可，如果是0则不考虑agent j，如果是1则考虑
        self.hard_encoding = nn.Linear(self.hidden_dim * 2, 2)  # 乘2因为是双向GRU，hidden_state维度为2 * hidden_dim

        # Soft
        self.q = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.k = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.v = nn.Linear(self.hidden_dim, self.attention_dim)

        params_num = 0
        for name, params in self.encoding.named_parameters():  # 统计参数就先算了 反正biGRU先留着
            params_num += params.numel()


    def forward(self, obs):
        '''
        obs.shape = (threads, n_agent, dim)
        '''
        n_thread = obs.shape[0]
        assert self.n_agent == obs.shape[1]
        size = self.n_agent * n_thread  # 从n_agent改为n_agent*n_thread适配向量环境
        # encoding
        h_out = f.relu(self.encoding(obs))  # (batch_size, n_agent, dim)  # TODO 留

        # Hard Attention，GRU和GRUCell不同，输入的维度是(序列长度, batch_size, dim)
        if self.hard:
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

            h_hard = torch.zeros((2 * 1, size, self.hidden_dim)).to(self.device)  # 因为是双向GRU，每个GRU只有一层，所以第一维是2 * 1
            h_hard, _ = self.hard_bi_GRU(input_hard, h_hard)  # (n_agents - 1, batch_size * n_agents, hidden_dim * 2)
            h_hard = h_hard.permute(1, 0, 2)  # (batch_size * n_agents, n_agents - 1, hidden_dim * 2)
            h_hard = h_hard.reshape(-1, self.hidden_dim * 2)  # (batch_size * n_agents * (n_agents - 1), hidden_dim * 2)

            hard_weights = self.hard_encoding(h_hard)  # 将(6, 128)映射为(6, 2)  # TODO 留
            hard_weights = f.gumbel_softmax(hard_weights, tau=self.tau)
            # print(hard_weights)
            # TODO 明天再check一下这里
            hard_weights = hard_weights[:self.n_agent, :].view(-1, self.n_agent, 1, self.n_agent - 1)
            hard_weights = hard_weights.permute(1, 0, 2, 3)

        else:
            hard_weights = torch.ones((self.n_agent, size // self.n_agent, 1, self.n_agent - 1))  # 第二个维度是n_thread
            hard_weights = hard_weights.to(self.device)

        # 已验证邻居关系在一个episode里动态变化
        # print(f'当n_thread = {hard_weights.shape[1]}, hard_weights.sum() = {hard_weights.sum()}')

        # Soft Attention
        if self.soft:
            q = self.q(h_out).reshape(-1, self.n_agent, self.attention_dim)  # (batch_size, n_agents, self.attention_dim)
            k = self.k(h_out).reshape(-1, self.n_agent, self.attention_dim)  # (batch_size, n_agents, self.attention_dim)
            v = f.relu(self.v(h_out)).reshape(-1, self.n_agent, self.attention_dim)  # (batch_size, n_agents, self.attention_dim)
            x = []
            for i in range(self.n_agent):
                q_i = q[:, i].view(-1, 1, self.attention_dim)  # agent i的q，(batch_size, 1, self.attention_dim)
                k_i = [k[:, j] for j in range(self.n_agent) if j != i]  # 对于agent i来说，其他agent的k
                v_i = [v[:, j] for j in range(self.n_agent) if j != i]  # 对于agent i来说，其他agent的v

                k_i = torch.stack(k_i, dim=0)  # (n_agents - 1, batch_size, self.attention_dim)
                k_i = k_i.permute(1, 2, 0)  # 交换三个维度，变成(batch_size, self.attention_dim， n_agents - 1)
                v_i = torch.stack(v_i, dim=0)
                v_i = v_i.permute(1, 2, 0)

                # (batch_size, 1, attention_dim) * (batch_size, attention_dim，n_agents - 1) = (batch_size, 1，n_agents - 1)
                score = torch.matmul(q_i, k_i)

                # 归一化
                scaled_score = score / np.sqrt(self.attention_dim)

                # softmax得到权重
                soft_weight = f.softmax(scaled_score, dim=-1)  # (batch_size，1, n_agents - 1)

                # 加权求和，注意三个矩阵的最后一维是n_agents - 1维度，得到(batch_size, self.attention_dim)
                x_i = (v_i * soft_weight * hard_weights[i]).sum(dim=-1)
                x.append(x_i)

            x = torch.stack(x, dim=-1).reshape(n_thread, self.n_agent, self.attention_dim)  # (batch_size, n_agents, self.attention_dim)
            obs_embed = torch.cat([h_out, x], dim=-1)  # h_out直接shortcut到最后

            return obs_embed

        else:
            return hard_weights



class G2ANetHardSoftAgent(DPPOAgent):
    def __init__(self, logger, device, agent_args, input_args):
        DPPOAgent.__init__(self, logger, device, agent_args, input_args)

        self.hidden_dim = 64
        self.attention_dim = 32

        self.g2a_embed_net = G2AEmbedNet(obs_dim=agent_args.observation_dim,
                             n_agent=agent_args.n_agent,
                             device=device,
                             hidden_dim=self.hidden_dim,
                             attention_dim=self.attention_dim
                             ).to(device)


        pi_dict, v_dict = self.pi_args._toDict(), self.v_args._toDict()
        pi_dict['sizes'][0] = self.hidden_dim + self.attention_dim  # 修改维度
        v_dict['sizes'][0] = self.hidden_dim + self.attention_dim  # 修改维度
        self.actors = nn.ModuleList()
        self.vs = nn.ModuleList()
        for i in range(self.n_agent):
            self.actors.append(CategoricalActor(**pi_dict).to(self.device))
            self.vs.append(MLP(**v_dict).to(self.device))
        self.optimizer_pi = Adam(self.actors.parameters(), lr=self.lr)
        self.optimizer_v = Adam(self.vs.parameters(), lr=self.lr_v)

        print(1)


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






