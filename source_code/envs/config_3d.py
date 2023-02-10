from unicodedata import name
import json5
import numpy as np
import random
'''环境配置类'''


class Config(object):

    def __init__(self, args=None):
        '''用一个dict类型的args初始化'''
        self.default_config()
        if args is not None:
            for key in args.keys():
                self.dict[key] = args[key]


    def __call__(self, attr):
        assert attr in self.dict.keys(), print('key error[', attr, ']')
        return self.dict[attr]

    def save_config(self, outfile=None):
        if outfile is None:
            outfile = 'default_save.json5'
        json_str = json5.dumps(self.dict, indent=4)
        with open(outfile, 'w') as f:
            f.write(json_str)

    def default_config(self):
        self.dict = {
            "description": "default",
            # Env
            "task_id": 0,
            "action_mode": 0,  # 1 for continuous,  0 for discrete,
            "action_root": 13,
            "dataset": "NCSU",

            "add_emergency": False,
            "concat_obs": True,
            "weighted_mode": True,
            "poi_visible_num": -1,
            "small_obs_num": -1,

            "scale": 100,
            "uav_speed": 20,
            "time_slot": 20,

            # Energy
            "initial_energy": 719280.0,
            "epsilon": 1e-3,

            # UAV
            "agent_field": 500,

            # PoI
            "update_num": 10,
            "update_user_num": 3,  # TODO 看下这个怎么作用于环境，和上一个有啥区别
            "user_data_amount": 1,  # TODO 为调高任务难度可以尝试这个

            "collect_range": 500,
            "rate_threshold": 0.05,
            "emergency_threshold": 100,
            "emergency_reward_ratio": [0.5,0],
            "emergency_penalty": "const",

        }

