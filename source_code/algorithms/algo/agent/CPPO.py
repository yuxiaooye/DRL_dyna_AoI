from algorithms.algo.agent.DPPO import DPPOAgent
import torch.nn as nn
from algorithms.models import CategoricalActor
from torch.optim import Adam
from algorithms.models import MLP

class CPPOAgent(DPPOAgent):
    def __init__(self, logger, device, agent_args, input_args):
        DPPOAgent.__init__(self, logger, device, agent_args, input_args)


