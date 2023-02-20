from gym.envs.registration import register

register(
    id='CrowdSim-v0',
    entry_point='algorithms.GCRL.envs.crowd_sim:CrowdSim',
)
