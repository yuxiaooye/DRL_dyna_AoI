import json5

'''环境配置类'''


class Config(object):

    def __init__(self, env_args, input_args):
        '''用一个dict类型的args初始化'''
        self.input_args = input_args
        self.default_config()
        self.dataset_config()
        for key in env_args.keys():
            self.dict[key] = env_args[key]

    def __call__(self, attr):
        assert attr in self.dict.keys(), print('key error[', attr, ']')
        return self.dict[attr]

    def save_config(self, outfile=None):
        if outfile is None:
            outfile = 'default_save.json5'
        json_str = json5.dumps(self.dict, indent=4)
        with open(outfile, 'w') as f:
            f.write(json_str)

    def dataset_config(self):
        dataset = self.input_args.dataset
        # rm = Roadmap(dataset)
        # self.dict['max_x'] = round(rm.max_dis_x)
        # self.dict['max_y'] = round(rm.max_dis_y)
        if dataset == 'purdue':
            self.dict['poi_num'] = 59
        elif dataset == 'NCSU':
            self.dict['poi_num'] = 33
        else:
            raise NotImplementedError


    def default_config(self):
        self.dict = {
            "description": "default",
            # Env
            "task_id": 0,
            "action_mode": 0,  # 1 for continuous,  0 for discrete,
            "action_root": 13,
            "dataset": "NCSU",
            "max_episode_step": 120,  # 适配NCSU中的human.csv~~
            "add_emergency": False,
            "weighted_mode": False,  # 先尽量简化问题。本想让weight反比于SNRth，但随着时间在变，不方便~
            "poi_visible_num": -1,

            "scale": 100,
            "uav_speed": 20,
            "time_slot": 20,

            # Energy
            "initial_energy": 719280.0,
            "epsilon": 1e-3,

            # UAV
            'uav_num': 3,
            "agent_field": 500,

            # PoI
            "update_num": 10,  # 一个uav同时服务多少个poi
            "update_user_num": 3,  # 作用于obs
            "user_data_amount": 1,  # 调环境：调高任务难度

            "collect_range": 500,
            "rate_threshold": 0.05,
            "emergency_threshold": 100,
            "emergency_reward_ratio": [0.5,0],
            "emergency_penalty": "const",

        }

