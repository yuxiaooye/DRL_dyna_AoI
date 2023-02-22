# yyx: 和Mobile_DPPO的内容保持一致

import numpy as np
from gym.spaces import Box
import torch.nn
from algorithms.models import MLP
from algorithms.utils import Config


def getArgs(radius_v, radius_pi, env, input_args=None):

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
    
    alg_args.max_episode_len = 600
    alg_args.model_based = False
    alg_args.load_pretrained_model = False
    alg_args.pretrained_model = None
    alg_args.model_batch_size = 128
    alg_args.model_buffer_size = 0

    agent_args = Config()

    agent_args.n_agent = env.UAV_NUM
    from envs.neighbor_graph import get_adj
    agent_args.adj = get_adj(env.UAV_NUM)
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
    agent_args.use_reduced_v = False  # 和dppo不同
    agent_args.use_rtg = True
    agent_args.use_gae_returns = False
    agent_args.advantage_norm = True
    # agent_args.observation_space = env.observation_space
    # agent_args.observation_dim = env.observation_space['Box'].shape[1]+400
    agent_args.observation_dim = env.observation_space['Box'].shape[1]-400+81  # 每个agent的obs的向量维度
    agent_args.action_space = env.action_space
    # agent_args.adj = env.neighbor_mask
    agent_args.radius_v = radius_v
    agent_args.radius_pi = radius_pi

    agent_args.squeeze = False

    p_args = None
    agent_args.p_args = p_args

    v_args = Config()
    v_args.activation = torch.nn.ReLU
    v_args.sizes = [-1, 64, 64, 1]
    agent_args.v_args = v_args

    pi_args = Config()
    pi_args.network = MLP
    pi_args.activation = torch.nn.ReLU
    # 最后分两个head 分别输出两个离散动作维度上9个选择、10个选择的概率分布
    pi_args.sizes = [-1, 64, 64]  # 9是硬编码的离散动作数
    pi_args.branchs = [env.action_space[0].n, env.action_space[1].n]
    pi_args.have_last_branch = False
    pi_args.squash = False
    agent_args.pi_args = pi_args

    alg_args.agent_args = agent_args

    return alg_args
