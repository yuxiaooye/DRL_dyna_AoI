import numpy as np
def get_adj(agent_num):
    if agent_num == 3:
        return np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ])
    elif agent_num == 6:
        # 日字编队
        # return np.array([
        #     [0, 1, 1, 0, 0, 0],
        #     [1, 0, 0, 1, 0, 0],
        #     [1, 0, 0, 1, 1, 0],
        #     [0, 1, 1, 0, 0, 1],
        #     [0, 0, 1, 0, 0, 1],
        #     [0, 0, 0, 1, 1, 0],
        # ])

        return np.array([
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ])

    else:
        raise NotImplementedError