import numpy as np
def get_adj(agent_num, fully_collect=False):
    if fully_collect:
        return np.ones((agent_num, agent_num))

    # 硬编码的环形拓扑图
    # 当n==3时是线形，不然和CPPO等价
    if agent_num == 2:
        return np.array([
            [0, 1],
            [1, 0],
        ])
    if agent_num == 3:
        return np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ])
    else:
        adj = np.zeros((agent_num, agent_num))
        for i in range(agent_num):
            adj[i][(i+1) % agent_num] = 1
            adj[i][(i-1+agent_num) % agent_num] = 1
        return adj
