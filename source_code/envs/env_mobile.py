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


class EnvMobile():
    ids = ['EnvMobile-v0']

    def __init__(self, env_args, input_args, **kwargs):
        assert input_args.fixed_col_time


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
        self.aoi_vio_penalty_ratio = self.config("aoi_vio_penalty_ratio")
        self.bonus_reward_ratio = self.config("bonus_reward_ratio")
        self.UPDATE_USER_NUM = self.config("update_user_num")
        self.USER_DATA_AMOUNT = self.config("user_data_amount")
        self.UAV_HEIGHT = self.config("uav_height")

        self.n_agents = self.UAV_NUM
        self.n_actions = self.ACTION_ROOT
        self.agent_field = self.config("agent_field")

        self.MAX_FIY_DISTANCE = self.TIME_SLOT * self.UAV_SPEED / self.SCALE

        # self.OBSTACLE = self.config('obstacle')
        self.OBSTACLE = []

        self._get_energy_coefficient()
        # self.action_space = spaces.Discrete(9)
        self.action_space = spaces.MultiDiscrete([9, 9])

        self.cell_num = 6
        self.cell_span_x = self.MAP_X / self.cell_num
        self.cell_span_y = self.MAP_Y / self.cell_num


        '''these mat is **read-only** 因此可以放在init中 而不必放在reset中每次episode开始时都读'''
        self.poi_mat = self.rm.init_pois(self.MAX_EPISODE_STEP)


        # self.poi_arrival = np.load(os.path.join(data_file_dir, 'arrival.npy'))[:self.POI_NUM, :self.MAX_EPISODE_STEP + 1]  # shape = (33, 121)，其中33是poi数，121是episode的时间步数
        # 每个时间步都生成一个包  # OK
        self.poi_arrival = np.ones((self.POI_NUM, self.MAX_EPISODE_STEP+1))  # 下标从0到120

        # self.poi_QoS = np.load(os.path.join(data_file_dir, f'QoS{self.MAX_EPISODE_STEP}/poi_QoS{self.input_args.dyna_level}.npy'))

        # self.QoS_MAX, self.QoS_MIN = self.poi_QoS.max(), self.poi_QoS.min()

        # 位置2 + aoi1
        self.poi_property_num = 2 + 1
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

        # yyx 0216 add
        self.poi_aoi = [0 for _ in range(self.POI_NUM)]
        self.poi_aoi_area = [0 for _ in range(self.POI_NUM)]

        self.uav_trace = [[] for i in range(self.UAV_NUM)]
        self.uav_state = [[] for i in range(self.UAV_NUM)]
        self.uav_energy_consuming_list = [[] for i in range(self.UAV_NUM)]

        self.dead_uav_list = [False for i in range(self.UAV_NUM)]

        self.poi_history = []  # episode结束后，长度为121
        self.aoi_vio_ratio_list = []  # 当前时间步有多大比例的PoI违反了aoi阈值
        self.tx_satis_ratio_list = []  # 当前时间步有多大比例的被服务aoi满足了data rate阈值
        # 监视下面几个reward乘上ratio之前的尺度
        self.aoi_reward_list = []
        self.aoi_penalty_reward_list = []
        self.tx_penalty_reward_list = []

        self.aoi_history = [0]  # episode结束后，长度为121。为什么初始时要有一个0？

        self.step_count = 0

        # 之前把这四个元素的初始化放在init中，导致跨episode时没有被reset
        self.uav_energy = np.asarray(
            [self.config("initial_energy") for i in range(self.UAV_NUM)],
            dtype=np.float64)
        self.uav_position = np.asarray(
            [[self.MAP_X / 2, self.MAP_Y / 2] for _ in range(self.UAV_NUM)],
            dtype=np.float16)
        self.poi_position = copy.deepcopy(self.poi_mat[:, 0, :])  # 0意为t=0时poi的初始位置

        self.poi_aoi = [0 for _ in range(self.POI_NUM)]

        self.collision_count = 0

        # 添加初始信息到一些数组中
        for uav_index in range(self.UAV_NUM):
            self.uav_trace[uav_index].append(self.uav_position[uav_index].tolist())
        self.poi_history.append({
            'pos': copy.deepcopy(self.poi_position),
            'aoi': np.array([0 for _ in range(self.POI_NUM)])
        })

        self.check_arrival()
        # self.cpu_preprocessor.reset()
        self.stacked_obs = [None for _ in range(4)]
        self.get_obs()

    def render(self, mode='human'):
        pass

    def _human_move(self):
        self.poi_position = copy.deepcopy(self.poi_mat[:, self.step_count, :])


    def _uavs_access_users(self, max_access_num):
        # 读: self.uav_position[uav_index]
        # 读: self.poi_position
        access_lists = []
        for uav_id, uav_pos in enumerate(self.uav_position):
            access_list = []
            # poi_id, rate, dis
            triple = [(poi_id,) + self._get_data_rate(uav_pos, poi_pos) for poi_id, poi_pos in enumerate(self.poi_position)]
            sorted_triple_in_range = sorted(
                list(filter(lambda x: x[2] < self.COLLECT_RANGE, triple)), key=lambda x: x[2])
            access_list = sorted_triple_in_range[:max_access_num[uav_id]+1]  # +1将0~8映射到1~9
            # 每个user实际分到的带宽需要平分
            access_list = list(map(lambda x: (x[0], x[1]/len(access_list), x[2]), access_list))
            access_lists.append(access_list)
        return access_lists


    def step(self, action):
        action1, action2 = [item[0] for item in action], [item[1] for item in action]

        # 若max_episode_step=120, 则执行120次step方法。episode结束时保存120个poi和uav的位置点，而不是icde的121个，把poi、uav的初始位置扔掉！
        self.step_count += 1

        # poi移动
        self._human_move()

        # uav移动
        for uav_index in range(self.UAV_NUM):
            new_x, new_y, distance, energy_consuming = self._cal_uav_next_pos(uav_index, action1[uav_index])  # 调用关键函数，uav移动
            Flag = self._judge_obstacle(self.uav_position[uav_index], (new_x, new_y))
            if not Flag:  # 如果出界，就不更新uav的位置
                self.uav_position[uav_index] = (new_x, new_y)
            self.uav_trace[uav_index].append(self.uav_position[uav_index].tolist())  # 维护uav_trace
            self._use_energy(uav_index, energy_consuming)

        # uav收集
        ## 确定uav的服务对象，元素是（poi_id，data rate）二元组
        access_lists = self._uavs_access_users(action2)

        ## 计算各poi的总data rate
        sum_rates = np.zeros(self.POI_NUM)

        for uav_id, access_list in enumerate(access_lists):
            for poi_id, rate, dis in access_list:
                sum_rates[poi_id] += rate


        ## 若poi的总data rate满足阈值，则更新aoi

        uav_rewards = np.zeros(self.UAV_NUM)
        for poi_id, sum_rate in enumerate(sum_rates):
            if sum_rate < self.RATE_THRESHOLD:
                continue  # 不满足阈值

            ## aoi reset
            before_reset = self.poi_aoi[poi_id]
            self.poi_aoi[poi_id] = 0  # 戴总是reset到1，不过我后面会在check_arrival()里+=1，和戴总的等价

            ## 计算poi_aoi_area
            self.poi_aoi_area[poi_id] += 1/2 * (before_reset * self.TIME_SLOT)**2  # 加一块三角形的面积

            ## 计算aoi reset reward和bonus reward
            # 当年戴总infocom2022的定义是收集前平均aoi减收集后平均aoi，和现在我用的一致
            rate_contribute_to_that_poi = np.zeros(self.UAV_NUM)
            for uav_id, access_list in enumerate(access_lists):
                for pid, rate, dis in access_list:
                    if poi_id == pid:
                        # 收集aoi越大的user，奖励越大；rate/sum_rate相当于credit assignment
                        r = (before_reset / self.MAX_EPISODE_STEP) * (rate / sum_rate)
                        uav_rewards[uav_id] += r
                        self.aoi_reward_list.append(r)
                        rate_contribute_to_that_poi[uav_id] = rate

        visit_num = sum(sum_rates > 0)
        if visit_num != 0:
            satis_ratio = sum(sum_rates >= self.RATE_THRESHOLD) / visit_num  # 不统计没被访问的poi，所以分子不是POI_NUM
            self.tx_satis_ratio_list.append(satis_ratio)
            uav_rewards *= satis_ratio  # 把aoi的收集奖励根据tx satis ratio scale一下

        done = self._is_episode_done()
        if not done:
            self.check_arrival()

        '''step2. 维护当前时间步对aoi值的相关统计值'''
        now_aoi = 0  # 当前时间步所有poi的aoi值总和
        em_now = 0
        aoi_list = []  # 当前时间步各poi的aoi值
        for i in range(self.POI_NUM):
            aoi = self.poi_aoi[i]
            # print('aoi=', aoi)
            if aoi > self.AoI_THRESHOLD:  # 超过了AoI阈值
                em_now += 1
            now_aoi += aoi
            aoi_list.append(aoi)

        self.poi_history.append({
            'pos': copy.deepcopy(self.poi_position),
            'aoi': np.array(aoi_list)
        })
        self.aoi_history.append(now_aoi / self.POI_NUM)
        self.aoi_vio_ratio_list.append(em_now / self.POI_NUM)

        # 惩罚基于当前时刻违反AoIth的user的比例，所有uav的惩罚相同
        if em_now != 0:
            penalty_r = -np.ones(self.UAV_NUM) * (em_now / self.POI_NUM) * self.aoi_vio_penalty_ratio
            self.aoi_penalty_reward_list.extend((penalty_r/self.aoi_vio_penalty_ratio).tolist())
            uav_rewards += penalty_r

        '''step3. episode结束时的后处理'''
        info = {}
        if done:
            for poi_id in range(self.POI_NUM):  # 最后补一块三角形的面积，清算
                self.poi_aoi_area[poi_id] += 1/2 * (self.poi_aoi[poi_id] * self.TIME_SLOT) ** 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                info = self.summary_info(info)

        self.get_obs()
        return self.get_obs_from_outside(), uav_rewards, done, info

    def summary_info(self, info):
        t_e = np.sum(np.sum(self.uav_energy_consuming_list))

        # 分子是图2黄色部分面积，分母前三项是以秒为单位的总时间，单位是时间步数量
        episodic_aoi = sum(self.poi_aoi_area) / (self.POI_NUM * self.MAX_EPISODE_STEP * self.TIME_SLOT * self.TIME_SLOT)
        aoi_satis_ratio = sum(1 - np.array(self.aoi_vio_ratio_list)) / self.step_count
        tx_satis_ratio = sum(self.tx_satis_ratio_list) / len(self.tx_satis_ratio_list)
        energy_consuming_ratio = t_e / (self.UAV_NUM * self.INITIAL_ENERGY)
        info['episodic_aoi'] = episodic_aoi
        info['aoi_satis_ratio'] = aoi_satis_ratio
        info['tx_satis_ratio'] = tx_satis_ratio
        info['energy_consuming'] = t_e / 10**6  # 单位：MJ
        info['energy_consuming_ratio'] = energy_consuming_ratio
        info['QoI'] = min(aoi_satis_ratio, tx_satis_ratio) / (t_e/10**6)

        info['aoi_reward'] = np.mean(self.aoi_reward_list)
        info['aoi_penalty_reward'] = np.mean(self.aoi_penalty_reward_list) if len(self.aoi_penalty_reward_list) != 0 else 0

        return info


    def _cal_distance(self, pos1, pos2):
        assert len(pos1) == len(
            pos2) == 2, 'cal_distance function only for 2d vector'
        distance = np.sqrt(
            np.power(pos1[0] - pos2[0], 2) + np.power(pos1[1] - pos2[1], 2)  # 这里算距离不要再*scale了~pos已经是以米为单位的距离了
            + np.power(self.UAV_HEIGHT, 2)
        )
        return distance

    def _cal_theta(self, pos1, pos2):
        assert len(pos1) == len(
            pos2) == 2, 'cal_theta function only for 3d vector'
        r = np.sqrt(np.power(pos1[0] * self.SCALE - pos2[0] * self.SCALE, 2) + np.power(
            pos1[1] * self.SCALE - pos2[1] * self.SCALE, 2))
        h = self.UAV_HEIGHT
        theta = math.atan2(h, r)
        return theta

    def _cal_energy_consuming(self, move_distance):
        moving_time = move_distance / self.UAV_SPEED
        hover_time = self.TIME_SLOT - moving_time
        return self.Power_flying * moving_time + self.Power_hovering * hover_time

    def _cal_uav_next_pos(self, uav_index, action):
        dx, dy = self._get_vector_by_action(int(action))  # 形如[1.5, 0]或[sqrt(1.5), sqrt(1.5)]
        distance = np.sqrt(np.power(dx * self.SCALE, 2) +
                           np.power(dy * self.SCALE, 2))
        energy_consume = self._cal_energy_consuming(distance)

        if self.uav_energy[uav_index] >= energy_consume:
            new_x, new_y = self.uav_position[uav_index] + [dx * self.SCALE, dy * self.SCALE]
        else:
            new_x, new_y = self.uav_position[uav_index]

        return new_x, new_y, distance, min(self.uav_energy[uav_index], energy_consume)



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
        
        # fc= 24
        # path_loss2 = (1 + alpha * math.exp(-beta * (theta - alpha)))*(28.0+22*math.log10(distance)+20*math.log10(fc))+(1-(1 + alpha * math.exp(-beta * (theta - alpha))))*(-17.5+(46-7*math.log10(100))*math.log10(distance)+20*math.log10(4*math.pi*fc/3))
        # print(path_loss,path_loss2)
        w_tx = 20
        w_noise = -104
        w_s_t = w_tx - path_loss - w_noise
        w_w_s_t = math.pow(10, (w_s_t - 30) / 10)
        bandwidth = 20e6
        data_rate = bandwidth * math.log2(1 + w_w_s_t)
        return data_rate / 1e6, distance  # 返回到外面的data rate的单位是Mbps

    def get_obs_from_outside(self):
        if self.input_args.use_stack_frame:
            return torch.concat(self.stacked_obs, dim=-1)  # shape = (3, obs_dim*4)
        else:
            return self.obs  # shape = (3, obs_dim)

    def get_obs(self, aoi_now=None, aoi_next=None):
        agents_obs = [self.get_obs_agent(i) for i in range(self.UAV_NUM)]  # 每个元素shape = (1715, )
        try:
            agents_obs = np.vstack(agents_obs)  # shape = (3, 1715)
        except:
            pass
        obs_dict = {
            'Box': agents_obs,
            'available_actions': self.get_avail_actions()
        }
        self.obs = torch.tensor(obs_dict['Box']).float()
        if self.step_count == 0:
            self.stacked_obs = [self.obs for _ in range(4)]
        else:
            self.stacked_obs.pop()
            self.stacked_obs.append(self.obs)

        return obs_dict

    def get_obs_agent(self, agent_id):
        # aoi退化为G-A-W 删除了obs的很多维度
        obs = []
        for i in range(self.UAV_NUM):  # uav的位置信息
            if i == agent_id:
                obs.append(self.uav_position[i][0] / self.MAP_X)  # 送入obs时对位置信息进行归一化
                obs.append(self.uav_position[i][1] / self.MAP_Y)
            elif self._cal_distance(self.uav_position[agent_id], self.uav_position[i]) < self.agent_field:
                obs.append(self.uav_position[i][0] / self.MAP_X)
                obs.append(self.uav_position[i][1] / self.MAP_Y)
            else:  # 看不到观测范围外的uav
                obs.extend([0, 0])

        # user的信息
        for poi_index, (poi_position, poi_aoi) in enumerate(zip(self.poi_position, self.poi_aoi)):
            d = self._cal_distance(poi_position, self.uav_position[agent_id])
            if not d < self.agent_field:  # user不在观测范围内
                for _ in range(self.poi_property_num):  # 3
                    obs.append(0)
            else:  # user在观测范围内
                # user的位置和当前aoi
                obs.append((poi_position[0]) / self.MAP_X)
                obs.append((poi_position[1]) / self.MAP_Y)
                obs.append(poi_aoi / 121)

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
            obs.extend(self._get_snrmap(agent_id))

        # 把当前的step_count也喂到obs中
        obs.append(self.step_count / self.MAX_EPISODE_STEP)
        obs = np.asarray(obs)
        return obs


    def _get_snrmap(self, uav_id):
        snrmap = np.zeros((self.cell_num, self.cell_num))  # 已将snrmap改为人群预测图

        # map部分可观测，根据uav位置确定哪些格子是可观测的
        visible = np.zeros((self.cell_num, self.cell_num))
        for i in range(self.cell_num):
            for j in range(self.cell_num):
                center = ((i+1/2)*self.cell_span_x, (j+1/2)*self.cell_span_y)
                if self._cal_distance(center, self.uav_position[uav_id]) < self.agent_field:
                    visible[i][j] = 1
                else:
                    visible[i][j] = 0

        # 要的是下一步user的位置，所以+1
        next_poi_positions = copy.deepcopy(self.poi_mat[:, min(self.step_count+1, self.poi_mat.shape[1]-1), :])  # 终止状态越界，取min
        for poi_index, next_poi_position in enumerate(next_poi_positions):
            x, y = next_poi_position
            i = np.clip(int(x/self.cell_span_x), 0, self.cell_num-1)
            j = np.clip(int(y/self.cell_span_y), 0, self.cell_num-1)
            if visible[i][j]:
                snrmap[i][j] += 1  # 根据用户位置把ans加到具体的cell中

        snrmap = snrmap / self.POI_NUM  # 归一化
        snrmap = snrmap.reshape(self.cell_num * self.cell_num, )
        return snrmap.tolist()



    def get_obs_size(self):
        size = 2 * self.UAV_NUM + self.POI_NUM * self.poi_property_num + 1  # 1是step_count
        # add future obs
        size += self.POI_NUM * self.input_args.future_obs * 3
        # add snr-map
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

    def check_arrival(self, step=0):  # 数据生成
        for i in range(self.POI_NUM):
            self.poi_aoi[i] += 1


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
