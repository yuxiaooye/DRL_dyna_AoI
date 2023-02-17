from envs.config_3d import Config

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import math
import warnings
import torch

from gym import spaces
import os
import os.path as osp
from env_configs.roadmap_env.roadmap_utils import Roadmap


# 看看有没有必要把MIMO用进来?
class EnvMobileHao():
    ids = ['EnvMobileHao-v0']

    def __init__(self, env_args, input_args, **kwargs):
        self.config = Config(env_args, input_args)
        self.input_args = input_args
        self.debug = self.input_args.debug
        self.test = self.input_args.test
        self.phase = kwargs['phase']
        # roadmap
        self.rm = Roadmap(self.input_args.dataset, self.config.dict)

        self.USE_SNRMAP = self.input_args.use_snrmap
        self.MAP_X = self.rm.max_dis_x
        self.MAP_Y = self.rm.max_dis_y
        self.WEIGHTED_MODE = self.config("weighted_mode")
        self.SCALE = self.config("scale")
        self.UAV_NUM = self.config("uav_num")
        self.INITIAL_ENERGY = env_args['initial_energy']
        self.EPSILON = self.config("epsilon")
        self.ACTION_ROOT = self.config("action_root")
        self.MAX_EPISODE_STEP = self.config("max_episode_step")
        self.TIME_SLOT = self.config("time_slot")
        self.TOTAL_TIME = self.MAX_EPISODE_STEP * self.TIME_SLOT
        self.UAV_SPEED = self.config("uav_speed")
        self.POI_VISIBLE_NUM = self.config("poi_visible_num")

        self.UPDATE_NUM = self.config("update_num")
        self.COLLECT_RANGE = self.config("collect_range")
        self.POI_NUM = self.config("poi_num")
        self.RATE_THRESHOLD = self.config("RATE_THRESHOLD")
        self.AoI_THRESHOLD = self.config("AoI_THRESHOLD")
        self.EMERGENCY_REWARD_RATIO = self.config("emergency_reward_ratio")
        self.UPDATE_USER_NUM = self.config("update_user_num")
        self.USER_DATA_AMOUNT = self.config("user_data_amount")

        self.n_agents = self.UAV_NUM
        self.n_actions = self.ACTION_ROOT
        self.agent_field = self.config("agent_field")

        self.MAX_FIY_DISTANCE = self.TIME_SLOT * self.UAV_SPEED / self.SCALE

        # self.OBSTACLE = self.config('obstacle')
        self.OBSTACLE = []


        self._get_energy_coefficient()
        self.action_space = spaces.Discrete(9)  # 硬编码，跟昊宝的实现保持一致

        self.cell_num = 6
        self.cell_span_x = self.MAP_X / self.cell_num
        self.cell_span_y = self.MAP_Y / self.cell_num


        '''these mat is **read-only** 因此可以放在init中 而不必放在reset中每次episode开始时都读'''
        self.poi_mat = self.rm.init_pois(self.MAX_EPISODE_STEP)
        data_file_dir = f'envs/{self.input_args.dataset}'
        self.debug_index = self.poi_mat[:,0,:].sum(axis=-1).argmin()  # tmp
        # == OK 对读的这几个列表进行裁剪，时间步240->121，poi数244->33 ==
        self.poi_arrival = np.load(os.path.join(data_file_dir, 'arrival.npy'))[:self.POI_NUM, :self.MAX_EPISODE_STEP + 1]  # shape = (33, 121)，其中33是poi数，121是episode的时间步数
        # 权重在计算reward和metric时用到，先简化问题 不用权重
        self.poi_weight = np.load(os.path.join(data_file_dir, 'poi_weights.npy'))[:self.POI_NUM]
        self.poi_QoS = np.load(os.path.join(data_file_dir, f'QoS{self.MAX_EPISODE_STEP}/poi_QoS{self.input_args.dyna_level}.npy'))
        assert self.poi_QoS.shape == (self.POI_NUM, self.MAX_EPISODE_STEP)
        ''''''

        self.QoS_MAX, self.QoS_MIN = self.poi_QoS.max(), self.poi_QoS.min()

        # 位置2 + 剩余包数1 + 距离上次收集过了多久1 + 最早三个包的到达时间1 + SNR信息1
        SNR_prop_num = 1
        self.poi_property_num = 2 + 1 + 1 + self.UPDATE_USER_NUM + SNR_prop_num
        info = self.get_env_info()

        obs_dict = {
            'Box': spaces.Box(low=-1, high=1, shape=(self.n_agents, info['obs_shape'])),
            'available_actions': spaces.Box(low=0, high=1, shape=(self.n_agents, self.ACTION_ROOT)),
        }
        self.obs_space = spaces.Dict(obs_dict)
        self.observation_space = self.obs_space

        self.obs = None
        self.stacked_obs = None
        self.reset()


    def reset(self):
        self.debug_all_d = []
        self.debug_all_r = []

        self.uav_trace = [[] for i in range(self.UAV_NUM)]  # 相当于我icde用的saved_uav_trajs
        self.uav_state = [[] for i in range(self.UAV_NUM)]
        self.uav_energy_consuming_list = [[]
                                          for i in range(self.UAV_NUM)]
        self.uav_data_collect = [[]
                                 for i in range(self.UAV_NUM)]

        self.dead_uav_list = [False for i in range(self.UAV_NUM)]

        self.update_list = [[] for i in range(self.UAV_NUM)]
        self.collect_list = [[] for i in range(self.UAV_NUM)]

        self.poi_history = []  # episode结束后，长度为121
        self.aoi_vio_ratio_list = [0]
        self.task_history = []
        self.aoi_history = [0]  # episode结束后，长度为241。为什么初始时要有一个0？
        self.area_aoi_history = [0]
        self.activate_list = []
        self.total_poi = []
        self.total_data_collect = 0
        self.total_data_arrive = 0

        self.step_count = 0

        '''之前把这四个元素的初始化放在init中，导致跨episode时没有被reset'''
        self.uav_energy = np.asarray(
            [self.config("initial_energy") for i in range(self.UAV_NUM)],
            dtype=np.float64)
        self.uav_position = np.asarray(
            [[self.MAP_X / 2, self.MAP_Y / 2] for _ in range(self.UAV_NUM)],
            dtype=np.float16)
        self.poi_position = copy.deepcopy(self.poi_mat[:, 0, :])  # 0意为t=0时poi的初始位置
        self.poi_value = [[] for _ in range(self.POI_NUM)]  # 维护当前队列中尚未被收集的包，内容是poi_arrive_time的子集
        ''''''

        self.poi_arrive_time = [[-1] for _ in range(self.POI_NUM)]  # 相比poi_value数组初值多了-1，且记录所有包生成时间，并不通过pop维护
        self.poi_delta_time = [[] for _ in range(self.POI_NUM)]
        self.poi_collect_time = [[] for _ in range(self.POI_NUM)]
        self.poi_aoi = [[] for _ in range(self.POI_NUM)]
        self.poi_wait_time = [[] for _ in range(self.POI_NUM)]

        self.poi_emergency = [[] for _ in range(self.POI_NUM)]

        self.collision_count = 0

        # -- yyx 添加初始信息到一些数组中 --
        for uav_index in range(self.UAV_NUM):
            self.uav_trace[uav_index].append(self.uav_position[uav_index].tolist())
        self.poi_history.append({
            'pos': copy.deepcopy(self.poi_position),
            'val': copy.deepcopy(np.array([0 for _ in range(self.POI_NUM)])).reshape(-1),
            'aoi': np.array([0 for _ in range(self.POI_NUM)])
        })
        # --

        self.check_arrival(self.step_count)
        # self.cpu_preprocessor.reset()
        self.stacked_obs = [None for _ in range(4)]
        self.get_obs()

    def render(self, mode='human'):
        pass

    def _human_move(self):
        self.poi_position = copy.deepcopy(self.poi_mat[:, self.step_count, :])


    def step(self, action):
        # 若max_episode_step=120, 则执行120次step方法。episode结束时保存120个poi和uav的位置点，而不是icde的121个，把poi、uav的初始位置扔掉！
        self.step_count += 1

        uav_reward = np.zeros([self.UAV_NUM])
        uav_data_collect = np.zeros([self.UAV_NUM])

        '''step1. poi、uav移动和uav收集'''
        self._human_move()  # 根据self.human_df更新self._poi_position

        for uav_index in range(self.UAV_NUM):
            self.update_list[uav_index].append([])
            self.collect_list[uav_index].append([])
            new_x, new_y, distance, energy_consuming = self._cal_uav_next_pos(uav_index, action[uav_index])  # 调用关键函数，uav移动
            Flag = self._judge_obstacle(self.uav_position[uav_index], (new_x, new_y))
            if not Flag:  # 如果出界，就不更新uav的位置
                self.uav_position[uav_index] = (new_x, new_y)
            self.uav_trace[uav_index].append(self.uav_position[uav_index].tolist())  # 维护uav_trace

            self._use_energy(uav_index, energy_consuming)
            # tau - 移动时间 = 收集时间
            # 打印发现要么是12.5，要么是20，昊宝用的是周围若干个点的离散动作，distance只有两种情况
            if self.input_args.fixed_col_time:
                a = 12.5  # 先硬编码为12.5
                assert 0 <= a <= self.TIME_SLOT
                collect_time = a if not Flag else 0
            else:
                collect_time = max(0, self.TIME_SLOT - distance / self.UAV_SPEED) if not Flag else 0
            r, uav_data_collect[uav_index] = self._collect_data_from_poi(  # 调用关键函数，uav收集
                uav_index, collect_time)

            self.uav_data_collect[uav_index].append(
                uav_data_collect[uav_index])

            uav_reward[uav_index] += r * (10 ** -3)  # * (2**-4)

        done = self._is_episode_done()

        user_num = np.array([len(p) for p in self.poi_value])  # 每个poi的队列中还有多少个包没被收集
        if not done:
            self.check_arrival(self.step_count)

        '''step2. 维护当前时间步对aoi值的相关统计值'''
        now_aoi = 0  # 当前时间步所有poi的aoi值总和
        em_now = 0
        em_penalty = 0
        temp_time = self.step_count * self.TIME_SLOT
        aoi_list = []  # 当前时间步各poi的aoi值
        for i in range(self.POI_NUM):
            if len(self.poi_collect_time[i]) > 0:  # 数据被收集，AoI重置为数据传输时间
                # 我觉得这里不应该是120-108.21，而应该是120-20，其中20来自poi_arrival_time[i][-1]，昊宝也认可了
                # 更准确来说，第1次收集后，应该-poi_arrival_time[i][1]，第2次收集后，应该-[i][2]，因此可能还需要新开一个变量记录当前收集了多少个包
                aoi = temp_time - self.poi_collect_time[i][-1]
            else:  # 数据未被收集，AoI根据y=x增长
                aoi = temp_time
            if aoi > self.AoI_THRESHOLD * self.TIME_SLOT:  # 超过了AoI阈值
                self.poi_emergency[i].append(1)
                em_now += 1
                em_penalty += 1  # penalty是常数1
            now_aoi += aoi
            aoi_list.append(aoi)

        self.poi_history.append({
            'pos': copy.deepcopy(self.poi_position),
            'val': copy.deepcopy(user_num).reshape(-1),
            'aoi': np.array(aoi_list)
        })
        self.aoi_history.append(now_aoi / self.POI_NUM)

        self.aoi_vio_ratio_list.append(em_now / self.POI_NUM)  # 当前时间步有多少PoI违反了阈值

        for u in range(self.UAV_NUM):  # reward中对于违反阈值的惩罚项
            uav_reward[u] -= (em_penalty / self.POI_NUM) * self.EMERGENCY_REWARD_RATIO

        '''step3. episode结束时的后处理'''
        info = {}
        if done:
            poi_visit_ratio = sum([int(len(p) > 0) for p in self.poi_collect_time]) / self.POI_NUM
            info['f_poi_visit_ratio'] = poi_visit_ratio

            for poi_index in range(self.POI_NUM):
                # while的目的：迭代把图2黄色部分面积的计算补充完整
                # poi_value的长度随着pop操作在while的迭代中减少，直到被清空为止。
                while len(self.poi_value[poi_index]) > 0:
                    index = self.poi_arrive_time[poi_index].index(self.poi_value[poi_index].pop(0)) - 1
                    self.poi_collect_time[poi_index].append(self.TOTAL_TIME)  # episode已经结束了！并不是写poi_collect_time的时机
                    yn = self.poi_delta_time[poi_index][index]
                    tn = self.TOTAL_TIME - self.poi_arrive_time[poi_index][index + 1]

                    self.poi_wait_time[poi_index].append(self.TOTAL_TIME - self.poi_arrive_time[poi_index][index + 1])
                    self.poi_aoi[poi_index].append(yn * tn + 0.5 * yn * yn)

                    if len(self.poi_value[poi_index]) == 0:
                        self.poi_aoi[poi_index].append(0.5 * tn * tn)  # 也即最后一次进入while循环体（len=1）时，poi_aoi[poi_index]被填充了两次

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                info = self.summary_info(info)

            # self._plot_histograms(self.debug_all_r)
            # self._plot_aoi_trend(self.debug_index)
            # self.save_trajs()

        self.get_obs()
        return self.get_obs_from_outside(), uav_reward, done, info

    def summary_info(self, info):
        if self.WEIGHTED_MODE:
            poi_weights = copy.deepcopy(self.poi_weight) / np.mean(self.poi_weight)  # 权重归一化
        else:
            poi_weights = [1 for _ in range(self.POI_NUM)]
        t_e = np.sum(np.sum(self.uav_energy_consuming_list))
        total_arrive_user = 0
        collect_user = 0
        cr_for_each = []  # yyx debug 各个poi的collect ratio
        for index, p in enumerate(self.poi_collect_time):
            fenzi = np.sum([c < self.TOTAL_TIME for c in p])  # 之所以是小于号，因为episode结束时的那最后一次收集不算
            fenmu = len(p)
            collect_user += fenzi
            total_arrive_user += fenmu
            cr_for_each.append(fenzi / fenmu)
        # self._plot_histograms(cr_for_each)

        # mean_aoi的分母是以秒为单位的总时间，因此分子是图2黄色部分面积~~
        mean_aoi = np.sum([np.sum(p) for p in self.poi_aoi]) / (self.POI_NUM * self.TOTAL_TIME * self.TIME_SLOT)
        weighted_mean_aoi = np.sum([np.sum(p) * poi_weights[index] for index, p in enumerate(self.poi_aoi)]) / (self.POI_NUM * self.TOTAL_TIME * self.TIME_SLOT)

        em_coef = 480 * self.TIME_SLOT * 1000 / (self.TIME_SLOT * self.POI_NUM)
        weighted_bar_aoi = np.sum([(np.sum(p) + np.sum(self.poi_emergency[index]) * self.TIME_SLOT * em_coef) * poi_weights[index] for index, p in enumerate(self.poi_aoi)]) / (
                self.POI_NUM * self.TOTAL_TIME * self.TIME_SLOT)

        info['a_poi_collect_ratio'] = float(collect_user / total_arrive_user)
        info['b_emergency_violation_ratio'] = (np.sum(self.aoi_vio_ratio_list) / self.step_count).item()  # 论文metric：violation ratio
        info['d_aoi'] = mean_aoi.item()
        info['e_weighted_aoi'] = weighted_mean_aoi.item()  # 论文metric：episodic-aoi
        info['f_weighted_bar_aoi'] = weighted_bar_aoi.item()
        info['h_total_energy_consuming'] = t_e.item()
        info['h_energy_consuming_ratio'] = t_e / (self.UAV_NUM * self.INITIAL_ENERGY)
        info['f_episode_step'] = self.step_count

        if self.debug or self.test:
            print(f"collect_data_ratio: {info['a_poi_collect_ratio']} "
                  f"episodic_aoi: {info['e_weighted_aoi']} "
                  )

        return info


    def _cal_distance(self, pos1, pos2):
        assert len(pos1) == len(
            pos2) == 2, 'cal_distance function only for 2d vector'
        distance = np.sqrt(
            np.power(pos1[0] - pos2[0], 2) + np.power(pos1[1] - pos2[1], 2)  # 这里算距离不要再*scale了~pos已经是以米为单位的距离了
            + np.power(100, 2)  # uav飞行高度为100
        )
        return distance

    def _cal_theta(self, pos1, pos2):
        assert len(pos1) == len(
            pos2) == 2, 'cal_theta function only for 3d vector'
        r = np.sqrt(np.power(pos1[0] * self.SCALE - pos2[0] * self.SCALE, 2) + np.power(
            pos1[1] * self.SCALE - pos2[1] * self.SCALE, 2))
        h = 100
        theta = math.atan2(h, r)
        return theta

    def _cal_energy_consuming(self, move_distance):
        moving_time = move_distance / self.UAV_SPEED
        hover_time = self.TIME_SLOT - moving_time
        return self.Power_flying * moving_time + self.Power_hovering * hover_time

    def _cal_uav_next_pos(self, uav_index, action):
        dx, dy = self._get_vector_by_action(int(action))  # 形如[1.5, 0]或[sqrt(1.5), sqrt(1.5)]
        distance = np.sqrt(np.power(dx * self.SCALE, 2) +  # SCALE = 100, 将1.5缩放为150米，无人机速度为20米/秒，即在一个timeslot里飞行用时7.5秒
                           np.power(dy * self.SCALE, 2))
        energy_consume = self._cal_energy_consuming(distance)

        if self.uav_energy[uav_index] >= energy_consume:
            # yyx change
            # 昊宝的MAP_X的尺度是70，我的max_dis_x的尺度是3000，因此这里也需要缩放一下，不然画出来uav几乎不动，尺度不匹配
            new_x, new_y = self.uav_position[uav_index] + [dx * self.SCALE, dy * self.SCALE]
        else:
            new_x, new_y = self.uav_position[uav_index]

        return new_x, new_y, distance, min(self.uav_energy[uav_index], energy_consume)

    def _collect_data_from_poi(self, uav_index, collect_time=0):
        position_list = []
        reward_list = []
        if collect_time >= 0:
            for poi_index, (poi_position, poi_value) in enumerate(zip(self.poi_position, self.poi_value)):
                d = self._cal_distance(
                    poi_position, self.uav_position[uav_index])
                self.debug_all_d.append(d)
                # 固定或动态SNRth
                SNRth = self.COLLECT_RANGE if self.input_args.fixed_range else self.poi_QoS[poi_index][self.step_count - 1]
                if d < SNRth:
                    # if self.input_args.debug and poi_index == self.debug_index:
                    #     print(f'debug poi has been collected! distance: {d} collect by: uav{uav_index}')
                    # if d < self.COLLECT_RANGE and len(poi_value) > 0:  # SNRth固定，不依赖于poi_index和step_count变化。来自--snr
                    position_list.append((poi_index, d))

            position_list = sorted(position_list, key=lambda x: x[1])  # 优先收集最近的user的队列中的所有包
            update_num = min(len(position_list), self.UPDATE_NUM)  # UPDATE_NUM = 10, 也即一个无人机最多同时服务10个poi
            now_time = (self.step_count + 1) * self.TIME_SLOT - collect_time  # (0+1) * 20 - 12.5 = 7.5

            debug_collect_proportion = []
            for i in range(update_num):
                poi_index = position_list[i][0]  # 首次到达断点时，poi_index = 128
                rate = self._get_data_rate(
                    self.uav_position[uav_index], self.poi_position[poi_index])
                # 对data rate的需求，既可以用SNRth描述，也可以用RATEth~~是等价的
                if rate <= self.RATE_THRESHOLD:  # RATE_THRESHOLD = 0.05，random跑的时候一次都不触发，可以扔了
                    rate = 0
                update_user_num = min(50, len(self.poi_value[poi_index]))
                if self.input_args.amount_prop_to_SNRth:  # amount与当前时刻SNRth成反比
                    SNRth = self.poi_QoS[poi_index][self.step_count-1]
                    amount = (500 - SNRth)/2 + 1  # 将100~500映射为3~1
                else:
                    amount = self.USER_DATA_AMOUNT
                delta_t = amount / rate  # 单位为秒，值都是零点几
                weight = 1 if not self.WEIGHTED_MODE else self.poi_weight[poi_index]

                debug_packet_num_before_collect = len(self.poi_value[poi_index])
                for u in range(update_user_num):  # 对于被uav选中服务的poi，可以多次收集它的包，直到collect_time被消耗完~~
                    # 看下对于被服务的用户，收集的包数占队列中剩余包数的比例，是不是太大了，如果是的话可以调低TIME_SLOT
                    # 用在Amount=1的环境上训练的结果，在Amount=3的环境上inference，看这个比例是不是可以变大
                    # 每收集一个poi的一个包，now_time就会增加delta_t, 因此这个break是经常触发的
                    if now_time + delta_t >= (self.step_count + 1) * self.TIME_SLOT:
                        break
                    if now_time <= self.poi_value[poi_index][0]:  # 保证收集时间比数据生成时间晚
                        # assert now_time == self.poi_value[poi_index][0]  # OK 验证了我的猜测
                        break
                    # 编辑tn>500的断点，poi_arrive_time[167] = [-1, 60, ...], index = 0, 意为这是该poi第一次被数据收集
                    index = self.poi_arrive_time[poi_index].index(self.poi_value[poi_index].pop(0)) - 1  # index的物理意义：确定当前是第几个包被收集到
                    self.poi_collect_time[poi_index].append(now_time)  # 唯一写poi_collect_time的地方
                    yn = self.poi_delta_time[poi_index][index]  # yn其实就是相邻两次数据生成之间的间隔时间，也即t2g-t1g
                    tn = max(0, now_time - self.poi_arrive_time[poi_index][index + 1])  # 747.59 - 60 = 687.59，表示对于当前收集的包，生成与被收集之间的时间差

                    assert tn >= 0 and yn > 0
                    self.poi_aoi[poi_index].append(yn * tn + 0.5 * yn * yn)
                    self.poi_wait_time[poi_index].append(now_time - self.poi_arrive_time[poi_index][index + 1])
                    reward = yn
                    reward_list.append(reward * weight)
                    now_time += delta_t
                    assert now_time <= (self.step_count + 1) * self.TIME_SLOT + 1

                if debug_packet_num_before_collect != 0:
                    debug_collect_proportion.append(
                        (debug_packet_num_before_collect - len(self.poi_value[poi_index])) / debug_packet_num_before_collect)
                if now_time >= (self.step_count + 1) * self.TIME_SLOT:
                    break
            # if (self.debug or self.test) and len(debug_collect_proportion) != 0:
            #     print(np.array(debug_collect_proportion).mean())
                # self._plot_histograms(debug_collect_proportion)
        return sum(reward_list), len(reward_list)

    def _get_vector_by_theta(self, action):
        theta = action[0] * np.pi
        l = action[1] + 1
        dx = l * np.cos(theta)
        dy = l * np.sin(theta)
        return dx, dy

    def _get_vector_by_action(self, action):
        single = 1.5
        base = single / math.sqrt(2)
        action_table = [
            [0, 0],
            [-base, base],
            [0, single],
            [base, base],
            [-single, 0],
            [single, 0],
            [-base, -base],
            [0, -single],
            [base, -base],

            [0, self.MAX_FIY_DISTANCE],
            [0, -self.MAX_FIY_DISTANCE],
            [self.MAX_FIY_DISTANCE, 0],
            [-self.MAX_FIY_DISTANCE, 0],
        ]
        return action_table[action]

    def _is_uav_out_of_energy(self, uav_index):
        return self.uav_energy[uav_index] < self.EPSILON

    def _is_episode_done(self):
        if self.step_count >= self.MAX_EPISODE_STEP:
            return True
        return False

    def _judge_obstacle(self, cur_pos, next_pos):
        if (0 <= next_pos[0] <= self.MAP_X) and (0 <= next_pos[1] <= self.MAP_Y):
            return False
        else:
            return True

    def _use_energy(self, uav_index, energy_consuming):
        self.uav_energy_consuming_list[uav_index].append(
            min(energy_consuming, self.uav_energy[uav_index]))
        self.uav_energy[uav_index] = max(
            self.uav_energy[uav_index] - energy_consuming, 0)

        if self._is_uav_out_of_energy(uav_index):
            self.dead_uav_list[uav_index] = True
            self.uav_state[uav_index].append(0)
        else:
            self.uav_state[uav_index].append(1)

    def _get_energy_coefficient(self):

        P0 = 58.06  # blade profile power, W
        P1 = 79.76  # derived power, W
        U_tips = 120  # tip speed of the rotor blade of the UAV,m/s
        v0 = 4.03  # the mean rotor induced velocity in the hovering state,m/s
        d0 = 0.2  # fuselage drag ratio
        rho = 1.225  # density of air,kg/m^3
        s0 = 0.05  # the rotor solidity
        A = 0.503  # the area of the rotor disk, m^2
        Vt = self.config("uav_speed")  # velocity of the UAV,m/s ???

        self.Power_flying = P0 * (1 + 3 * Vt ** 2 / U_tips ** 2) + \
                            P1 * np.sqrt((np.sqrt(1 + Vt ** 4 / (4 * v0 ** 4)) - Vt ** 2 / (2 * v0 ** 2))) + \
                            0.5 * d0 * rho * s0 * A * Vt ** 3

        self.Power_hovering = P0 + P1

    def _get_data_rate(self, uav_position, poi_position):
        eta = 2
        alpha = 4.88
        beta = 0.43
        distance = self._cal_distance(uav_position, poi_position)
        theta = self._cal_theta(uav_position, poi_position)
        path_loss = 54.05 + 10 * eta * math.log10(distance) + (-19.9) / (1 + alpha * math.exp(-beta * (theta - alpha)))
        w_tx = 20
        w_noise = -104
        w_s_t = w_tx - path_loss - w_noise
        w_w_s_t = math.pow(10, (w_s_t - 30) / 10)
        bandwidth = 20e6
        data_rate = bandwidth * math.log2(1 + w_w_s_t)
        return data_rate / 1e6

    def get_obs_from_outside(self):  # yyx add
        if self.input_args.use_stack_frame:
            return torch.concat(self.stacked_obs, dim=-1)  # shape = (3, obs_dim*4)
        else:
            return self.obs  # shape = (3, obs_dim)

    def get_obs(self, aoi_now=None, aoi_next=None):
        agents_obs = [self.get_obs_agent(i) for i in range(self.UAV_NUM)]  # 每个元素shape = (1715, )
        agents_obs = np.vstack(agents_obs)  # shape = (3, 1715)
        obs_dict = {
            'Box': agents_obs,
            'available_actions': self.get_avail_actions()
        }
        self.obs = torch.tensor(obs_dict['Box']).float()  # yyx add
        if self.step_count == 0:
            self.stacked_obs = [self.obs for _ in range(4)]
        else:
            self.stacked_obs.pop()
            self.stacked_obs.append(self.obs)

        return obs_dict

    def get_obs_agent(self, agent_id):
        poi_position_all = self.poi_position
        poi_value_all = self.poi_value

        obs = []
        # uav的位置信息
        for i in range(self.UAV_NUM):
            if i == agent_id:
                obs.append(self.uav_position[i][0] / self.MAP_X)  # 送入obs时对位置信息进行归一化
                obs.append(self.uav_position[i][1] / self.MAP_Y)
            elif self._cal_distance(self.uav_position[agent_id], self.uav_position[i]) < self.agent_field:
                obs.append(self.uav_position[i][0] / self.MAP_X)
                obs.append(self.uav_position[i][1] / self.MAP_Y)
            else:  # 看不到观测范围外的uav
                obs.extend([0, 0])

        # user的信息
        for poi_index, (poi_position, poi_value) in enumerate(zip(poi_position_all, poi_value_all)):
            d = self._cal_distance(poi_position, self.uav_position[agent_id])
            if not d < self.agent_field:  # user不在观测范围内
                for _ in range(self.poi_property_num):  # 8
                    obs.append(0)
            else:  # user在观测范围内
                '''user的位置和队列剩余包数'''
                obs.append((poi_position[0]) / self.MAP_X)
                obs.append((poi_position[1]) / self.MAP_Y)
                obs.append(len(poi_value) / 121)

                '''距离poi上次被收集已经过了多久'''
                if len(self.poi_collect_time[poi_index]) > 0:
                    obs.append(((self.step_count) * self.TIME_SLOT - self.poi_collect_time[poi_index][-1]) / self.TOTAL_TIME)
                else:  # 一次都没被收集过
                    obs.append(((self.step_count) * self.TIME_SLOT) / self.TOTAL_TIME)

                '''队列中最早到的三个包的到达时间'''
                delta_list = []
                for arrive in poi_value:
                    index = self.poi_arrive_time[poi_index].index(arrive) - 1  # 当前队列中的包是总arrive数组中的第几个包
                    if self.poi_arrive_time[poi_index][index] < 0:  # 只有一种可能，就是等于-1，也即最初的无意义的dummy包
                        delta_list.append(0)
                    else:  # 记录队列中最早到的三个包的时间
                        delta_list.append(self.poi_arrive_time[poi_index][index] / self.TOTAL_TIME)
                    if len(delta_list) == self.UPDATE_USER_NUM:
                        break
                if len(delta_list) < self.UPDATE_USER_NUM:  # 用0补齐delta_list到长度为3，最多就记录三个信息了
                    delta_list += [0 for _ in range(self.UPDATE_USER_NUM - len(delta_list))]
                obs.extend(delta_list)

                '''下次数据收集时的SNR阈值'''
                snr = self.poi_QoS[poi_index][min(self.step_count, self.MAX_EPISODE_STEP-1)]  # 防止越界，越界的时候是终止状态，也没被agent读~
                snr_obs = (snr - self.QoS_MIN) / self.QoS_MIN  # 映射到(0, 1)
                obs.append(snr_obs)


            '''添加未来的信息供当前时刻的agent决策'''
            def check_future_arrival(poi_index, t):
                delta_step = 121 - self.MAX_EPISODE_STEP
                stub = min(delta_step + self.step_count+t+1, self.MAX_EPISODE_STEP)  # 防止episode接近结束时下一句越界
                is_arrival = self.poi_arrival[poi_index, stub]
                return is_arrival

            for t in range(self.input_args.future_obs):  # 0 or 1 or 2
                stub = min(self.step_count+t+1, self.MAX_EPISODE_STEP)
                next_pos = self.poi_mat[poi_index, stub, :]
                obs.append(next_pos[0] / self.MAP_X)
                obs.append(next_pos[1] / self.MAP_Y)
                # 这个0 or 1的特征可能网络不好学。。改成未来若干步内有多少步会来包可能更好？也降低状态维度
                is_arrival = check_future_arrival(poi_index, t)
                obs.append(is_arrival)

        # snrmap的信息
        if self.USE_SNRMAP:
            obs.extend(self._get_snrmap(i))

        # 把当前的step_count也喂到obs中
        obs.append(self.step_count / self.MAX_EPISODE_STEP)
        obs = np.asarray(obs)
        return obs

    def _get_snrmap(self, i):
        snrmap = np.zeros((self.cell_num, self.cell_num))

        '''
        计算每个(uav-user)对的ans
        队列中有包的poi距离它最近的uav有多远 即为d，当然这是很naive的做法，设计的自由度很高、可考虑的东西很多
        '''
        # 要的是下一步user的位置，所以+1
        next_poi_positions = copy.deepcopy(self.poi_mat[:, min(self.step_count+1, self.poi_mat.shape[1]-1), :])  # 终止状态越界，取min
        # 如果更精细地做，poi_value也应该是用下一时刻的，不过当前时刻新产生的包大概率也不会被下一时刻的无人机收集到（因为FIFO），所以先不做
        for poi_index, (next_poi_position, poi_value) in enumerate(zip(next_poi_positions, self.poi_value)):
            if len(self.poi_value[poi_index]) == 0: continue  # 队列中没包的条件是len为0吗？
            d_min = float('inf')
            for uav_index in range(self.n_agents):
                d = self._cal_distance(next_poi_position, self.uav_position[uav_index])
                d_min = min(d, d_min)
            if self.input_args.fixed_range:
                next_SNRth = self.COLLECT_RANGE
            else:  # self.step_count-1是当前的阈值，这里要的是下一步的阈值，所以不-1
                next_SNRth = self.poi_QoS[poi_index][self.step_count]
            # ans反映一个poi的需求没被满足的gap，物理意义是uav应该朝着ans大的cell移动
            # 一方面，d_min越大，说明当前没有无人机靠近该user
            # 另一方面，mext_SNRth越小，说明无人机必须很接近该user才能meet his/her requirement
            # 如果已经有无人机在收集范围内，ans置为0
            ans = max(d_min - next_SNRth, 0)
            x, y = next_poi_position
            i = np.clip(int(x/self.cell_span_x), 0, self.cell_num-1)
            j = np.clip(int(y/self.cell_span_y), 0, self.cell_num-1)
            snrmap[i][j] += ans  # 根据用户位置把ans加到具体的cell中

        if snrmap.max() != 0:
            snrmap = snrmap / snrmap.max()  # 归一化
        snrmap = snrmap.reshape(self.cell_num * self.cell_num, )
        # TODO 目前的实现是全局snr-map 后面改成每个uav各自部分可观
        return snrmap.tolist()



    def get_obs_size(self):
        size = 2 * self.UAV_NUM + self.POI_NUM * self.poi_property_num + 1  # 1是step_count
        # yyx add future obs
        size += self.POI_NUM * self.input_args.future_obs * 3
        # yyx add snr-map
        if self.USE_SNRMAP:
            size += self.cell_num * self.cell_num
        return size

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return np.vstack(avail_actions)

    def get_avail_agent_actions(self, agent_id):
        avail_actions = []
        temp_x, temp_y = self.uav_position[agent_id]
        for i in range(self.ACTION_ROOT):
            dx, dy = self._get_vector_by_action(i)
            if not self._judge_obstacle((temp_x, temp_y), (dx + temp_x, dy + temp_y)):
                avail_actions.append(1)
            else:
                avail_actions.append(0)

        return np.array(avail_actions)

    def get_total_actions(self):
        return self.n_actions

    def get_num_of_agents(self):
        return self.UAV_NUM

    def close(self):
        pass

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "MAX_EPISODE_STEP": self.MAX_EPISODE_STEP}
        return env_info

    def check_arrival(self, step):  # arrival指数据生成
        delta_step = 121 - self.MAX_EPISODE_STEP  # 1  这里也许应该和昊哥保持一致改成120
        time = step * self.TIME_SLOT
        temp_arrival = self.poi_arrival[:, delta_step + step]  # 在step时间步各poi是否到达。元素为0或1，0代表该poi未到达，1代表到达
        for p_index in range(len(temp_arrival)):  # 北京共有244个poi，因此len(temp_arrival) = 244
            if temp_arrival[p_index] > 0:  # 仅对到达的poi进行处理
                # 注意对以下三个数组的写的值，单位都是秒，TIME_SLOT = 20
                self.poi_value[p_index].append(time)
                self.poi_delta_time[p_index].append(time - self.poi_arrive_time[p_index][-1])  # 本次到达与上次到达之间的间隔，也即图(2)b纵轴相邻两次到达的间隔
                self.poi_arrive_time[p_index].append(time)


    def _plot_aoi_trend(self, poi_index):
        assert len(self.poi_history) == 121
        x = range(121)
        y = [self.poi_history[t]['aoi'][poi_index] for t in range(121)]
        plt.plot(x, y)
        plt.show()

    def _plot_histograms(self, data):
        plt.hist(data, bins=20, rwidth=0.8)
        plt.show()

    def save_trajs_2(self, best_trajs, total_steps=1,
                     phase='train', is_newbest=False):

        postfix = 'best' if is_newbest else str(total_steps)
        save_traj_dir = osp.join(self.input_args.output_dir, f'{phase}_saved_trajs')
        if not osp.exists(save_traj_dir): os.makedirs(save_traj_dir)
        np.savez(osp.join(save_traj_dir, f'eps_{postfix}.npz'), best_trajs)

        if is_newbest:
            from tools.post.vis import render_HTML
            render_HTML(self.input_args.output_dir, tag=phase)
            print('call vis.gif along with the training')
