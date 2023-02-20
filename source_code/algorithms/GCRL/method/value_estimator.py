import torch.nn as nn
from algorithms.GCRL.method.base import mlp
from algorithms.GCRL.configs.config import BaseEnvConfig


class ValueEstimator(nn.Module):
    def __init__(self, config, graph_model, network_dim, device):
        super().__init__()
        # self.graph_model = graph_model
        # self.value_network = mlp(config.gcn.X_dim, network_dim)
        self.value_network = mlp(157, [128, 128, 1]).to(device)  # TODO 输入的157是硬编码的obs_dim

    def forward(self, state):
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        state (list): len=2, produced by state_predictor
        """
        tmp_config = BaseEnvConfig()
        robot_num = tmp_config.env.robot_num

        # assert len(state[0].shape) == 3
        # assert len(state[1].shape) == 3

        # only use the feature of robot node as state representation

        # state_embedding = self.graph_model(state)[:, 0:robot_num, :]  # shape = (batch, robot_num, embedding_dim)
        # values = self.value_network(state_embedding)  # shape = (batch,robot num, 1)
        # value = values.mean(dim=1)  # 不同uav的价值估计的平均

        value = self.value_network(state)
        return value
