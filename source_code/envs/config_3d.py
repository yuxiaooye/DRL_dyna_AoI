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
        if dataset == 'purdue':
            poi_num = 59
        elif dataset == 'NCSU':
            poi_num = 33
        elif dataset == 'KAIST':
            poi_num = 92
        else: raise NotImplementedError
        self.dict['poi_num'] = poi_num

    def default_config(self):
        self.dict = {
            "description": "default",
            # Env
            "task_id": 0,
            "action_root": 13,
            "dataset": "NCSU",
            "max_episode_step": 120,  # 适配NCSU中的human.csv~~
            "weighted_mode": False,
            "poi_visible_num": -1,

            "scale": 100,
            "uav_speed": 20,
            "time_slot": 20,
            "uav_height": 100,

            # Energy
            "initial_energy": 719280.0,
            "epsilon": 1e-3,

            # UAV
            'uav_num': 3,
            "agent_field": 750,  # 2.18下午改之前是500

            # PoI
            "update_num": 10,  # 一个uav同时服务多少个poi
            "update_user_num": 3,  # 作用于obs
            "user_data_amount": 1,  # 调环境：调高任务难度

            "collect_range": 500,
            "RATE_THRESHOLD": 5,  # 关键 Mbps
            "AoI_THRESHOLD": 100,  # 关键
            "bonus_reward_ratio": 0.0,  # 这个ratio就算要设，也必须比penalty ratio小很多，因为他俩本身的尺度不一样
            "aoi_vio_penalty_scale": 0.2,
            "tx_vio_penalty_scale": 0.01,
            "hao02191630": False
        }

