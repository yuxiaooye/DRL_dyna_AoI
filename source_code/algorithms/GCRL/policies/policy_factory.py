from algorithms.GCRL.policies.model_predictive_rl import ModelPredictiveRL
from algorithms.GCRL.policies.random import RandomPolicy
from algorithms.GCRL.policies.gcn import GCN

def none_policy():
    return None

policy_factory=dict()
policy_factory['model_predictive_rl'] = ModelPredictiveRL
policy_factory['gcn'] = GCN
policy_factory['random'] = RandomPolicy
policy_factory['none'] = none_policy



