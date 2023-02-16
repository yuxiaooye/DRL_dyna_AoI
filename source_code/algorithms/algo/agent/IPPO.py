from algorithms.algo.agent.DPPO import DPPOAgent
import torch.nn as nn
from algorithms.models import CategoricalActor, EnsembledModel, SquashedGaussianActor, ParameterizedModel_MBPPO
from torch.optim import Adam
from algorithms.models import MLP

class IPPOAgent(DPPOAgent):
    def __init__(self, logger, device, agent_args, input_args):
        DPPOAgent.__init__(self, logger, device, agent_args, input_args)

        self.use_extended_value = False  # 关键，后续不进行智能体间obs的卷

        self.actors = self._init_actors()  # collect_pi和collect_v应该一样吧？
        self.vs = self._init_vs()
        self.optimizer_pi = Adam(self.actors.parameters(), lr=self.lr)
        self.optimizer_v = Adam(self.vs.parameters(), lr=self.lr_v)

    def _init_actors(self):
        actors = nn.ModuleList()
        for i in range(self.n_agent):
            self.pi_args.sizes[0] = self.observation_dim
            actors.append(CategoricalActor(**self.pi_args._toDict()).to(self.device))
        return actors

    def _init_vs(self):
        vs = nn.ModuleList()
        for i in range(self.n_agent):
            self.v_args.sizes[0] = self.observation_dim
            vs.append(MLP(**self.v_args._toDict()).to(self.device))
        return vs