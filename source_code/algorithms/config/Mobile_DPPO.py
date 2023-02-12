'''本文件从Catchup_DPPO魔改而来'''

import numpy as np
from gym.spaces import Box
import torch.nn
from algorithms.models import MLP
from algorithms.utils import Config


def getArgs(radius_p, radius_v, radius_pi, env):

    alg_args = Config()
    # 总训练步数 = n_iter * rollout_length，默认25K * 0.6K = 15M
    # 改为5K * 0.6K = 6M
    alg_args.n_iter = 5000  # 25000
    alg_args.n_inner_iter = 1
    alg_args.n_warmup = 0
    alg_args.n_model_update = 5
    alg_args.n_model_update_warmup = 10
    alg_args.n_test = 1  # default=5, 意为每次test 5个episode
    alg_args.test_interval = 20
    alg_args.rollout_length = 600  # 也即PPO中的T_horizon，
    alg_args.test_length = 600
    alg_args.max_episode_len = 600
    alg_args.model_based = False
    alg_args.load_pretrained_model = False
    alg_args.pretrained_model = None
    alg_args.model_batch_size = 128
    alg_args.model_buffer_size = 0

    agent_args = Config()

    # 下面这个值后续会被读，但env_ucs环境没有neighbor_mask属性，
    # 所以仿照catchup，在这里临时硬编码一个neighbor_mask
    tmp_neighbor_mask = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
    )
    # agent_args.adj = env.neighbor_mask
    agent_args.adj = tmp_neighbor_mask
    agent_args.n_agent = agent_args.adj.shape[0]
    agent_args.gamma = 0.99
    agent_args.lamda = 0.5
    agent_args.clip = 0.2
    agent_args.target_kl = 0.01
    agent_args.v_coeff = 1.0
    agent_args.v_thres = 0.
    agent_args.entropy_coeff = 0.0
    agent_args.lr = 5e-5
    agent_args.lr_v = 5e-4
    agent_args.n_update_v = 30
    agent_args.n_update_pi = 10
    agent_args.n_minibatch = 1
    agent_args.use_reduced_v = True
    agent_args.use_rtg = True
    agent_args.use_gae_returns = False
    agent_args.advantage_norm = True
    # agent_args.observation_space = env.observation_space
    agent_args.observation_dim = env.observation_space['Box'].shape[1]  # 标量1715，意为每个agent的obs的向量维度
    agent_args.action_space = env.action_space
    # agent_args.adj = env.neighbor_mask
    agent_args.radius_v = radius_v
    agent_args.radius_pi = radius_pi
    agent_args.radius_p = radius_p
    agent_args.squeeze = False

    p_args = None
    agent_args.p_args = p_args

    v_args = Config()
    v_args.network = MLP
    v_args.activation = torch.nn.ReLU
    v_args.sizes = [-1, 64, 64, 1]
    agent_args.v_args = v_args

    pi_args = Config()
    pi_args.network = MLP
    pi_args.activation = torch.nn.ReLU
    # pi_args.sizes = [-1, 64, 64, agent_args.action_space['uav'].n - 1 + agent_args.action_space['car'].n]
    pi_args.sizes = [-1, 64, 64, 9]  # 9是硬编码的离散动作数
    pi_args.squash = False
    agent_args.pi_args = pi_args

    alg_args.agent_args = agent_args

    return alg_args
