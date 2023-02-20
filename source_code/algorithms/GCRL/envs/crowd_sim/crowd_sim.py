import pandas as pd

import logging
import gym

import algorithms.GCRL.configs.config
from algorithms.GCRL.envs.model.utils import *
from shapely.geometry import Point
import numpy as np
import folium
from folium.plugins import TimestampedGeoJson, AntPath

from algorithms.GCRL.envs.model.mdp import JointState, build_action_space
from algorithms.GCRL.configs.config import BaseEnvConfig
from gym import spaces
from datetime import datetime


def cal_dis_to_green_region(human_df):
    '''
    param:
        human_df (pd.Dataframe) arbitrary human_df, we only utilize the column lat and lon
    return:
        ten best user to 4 green region, with the best window with 20 timesteps (根据AoI选时间，不根据距离了)
    '''
    f = open(r'F:\PycharmProjects\New folder2\logs\debug\2021_11_13_TKDE\greenRegion_sorted_userIndex.txt', 'w')
    tmp_config = configs.config.BaseEnvConfig()
    greenRegion = [(35.7696, -78.6912), (35.7749, -78.6843), (35.7758, -78.6828), (35.7724, -78.6741)]

    for i, gR in enumerate(greenRegion):
        dis = []
        for id in range(tmp_config.env.human_num):
            lats = np.array(human_df[human_df['id'] == id].latitude) - gR[0]
            lons = np.array(human_df[human_df['id'] == id].longitude) - gR[1]
            d = sum(lats * lats + lons * lons)
            dis.append(d)
        sorted_userIndex = np.argsort(dis)
        f.write('To green region {}:\nsorted user index:'.format(i + 1) + '\n')
        f.write(' '.join([str(index) for index in sorted_userIndex]) + '\n')

        if i == 3:  # 硬编码，在绿4删除效果不好的user0
            sorted_userIndex = np.delete(sorted_userIndex, np.where(sorted_userIndex == 0)[0].item())

        best_users = sorted_userIndex[:10]
        f.write('best users:' + '\n')
        f.write(' '.join([str(user) for user in best_users]) + '\n')

        span = 20
        best_start_timestep = None
        best_val = float('inf')
        for start_timestep in range(121-(span-1)):
            end_timestep = start_timestep+span-1
            dis = []
            for id in best_users:
                lats = np.array(human_df[human_df['id']==id].iloc[start_timestep:end_timestep].latitude) - gR[0]
                lons = np.array(human_df[human_df['id']==id].iloc[start_timestep:end_timestep].longitude) - gR[1]
                d = sum(lats * lats + lons * lons)
                dis.append(d)
                # aoi = np.array(human_df[human_df['id']==id].iloc[start_timestep:end_timestep].aoi)

            if sum(dis) < best_val:
                best_val = sum(dis)
                best_start_timestep = start_timestep


        f.write('best time window:' + '\n')
        f.write(str(best_start_timestep) + ' ' + str(best_start_timestep+span-1) + '\n')

    f.close()


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.time_limit = None
        self.robots = None
        self.humans = None
        self.agent = None
        self.current_timestep = None
        self.phase = None

        self.config = BaseEnvConfig()

        self.human_num = self.config.env.human_num
        self.robot_num = self.config.env.robot_num
        self.num_timestep = self.config.env.num_timestep
        self.step_time = self.config.env.step_time
        self.start_timestamp = self.config.env.start_timestamp
        self.max_uav_energy = self.config.env.max_uav_energy

        # load_dataset
        self.nlon = self.config.env.nlon
        self.nlat = self.config.env.nlat
        self.lower_left = self.config.env.lower_left
        self.upper_right = self.config.env.upper_right
        print('使用数据集：', self.config.env.dataset_dir)
        self.human_df = pd.read_csv(self.config.env.dataset_dir)
        logging.info("Finished reading {} rows".format(len(self.human_df)))
        # # 临时处理数据使用
        # sample_list=np.random.choice(self.human_num, size=[50,], replace=False)
        # sample_list=sample_list[np.argsort(sample_list)]
        # print(sample_list)
        # self.human_df= self.human_df[self.human_df["id"].isin(sample_list)]
        # for i,human_id in enumerate(sample_list):
        #     mask=(self.human_df["id"]==human_id)
        #     self.human_df.loc[mask,"id"]=i
        # self.human_df=self.human_df.sort_values(by=["id","timestamp"],ascending=[True,True])
        # print(self.human_df.head())
        # self.human_df.to_csv("50 users-5.csv",index=False)
        # exit(0)

        self.human_df['t'] = pd.to_datetime(self.human_df['timestamp'], unit='s')  # s表示时间戳转换
        self.human_df['aoi'] = -1  # 加入aoi记录aoi
        self.human_df['energy'] = -1  # 加入energy记录energy
        logging.info('human number: {}'.format(self.human_num))
        logging.info('Robot number: {}'.format(self.robot_num))

        # for debug
        self.current_human_aoi_list = np.ones([self.human_num, ])
        self.mean_aoi_timelist = np.ones([self.config.env.num_timestep + 1, ])
        self.robot_energy_timelist = np.zeros([self.config.env.num_timestep + 1, self.robot_num])
        self.robot_x_timelist = np.zeros([self.config.env.num_timestep + 1, self.robot_num])
        self.robot_y_timelist = np.zeros([self.config.env.num_timestep + 1, self.robot_num])
        self.update_human_timelist = np.zeros([self.config.env.num_timestep, ])
        self.data_collection = 0

        self.state_dim = (self.human_num + self.robot_num) * 4
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(pow(9, self.robot_num))
        self.full_action_space = build_action_space()

    def set_agent(self, agent):
        self.agent = agent

    def generate_human(self, human_id, selected_data, selected_next_data):
        human = Human(human_id, self.config)
        px, py, theta = get_human_position_from_list(self.current_timestep, human_id, selected_data, selected_next_data)
        human.set(px, py, theta, 1)  # human有aoi
        return human

    def generate_robot(self, robot_id):
        robot = Robot(robot_id, self.config)
        robot.set(self.nlon / 2, self.nlat / 2, 0, self.max_uav_energy)  # robot有energy
        return robot

    def sync_human_df(self, human_id, current_timestep, aoi):
        current_timestamp = self.start_timestamp + current_timestep * self.step_time
        current_index = self.human_df[
            (self.human_df.id == human_id) & (self.human_df.timestamp == current_timestamp)].index
        # self.human_df.loc[current_index, "aoi"] = aoi   # slower
        self.human_df.iat[current_index.values[0], 9] = aoi

    def reset(self, phase='test', test_case=None):
        # 暂时先把随机种子定死。原始代码里存在有case size迭代随机种子的操作，这是好操作。
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        base_seed = {'train': 0, 'val': 0, 'test': 0}
        # np.random.seed(base_seed[phase])

        self.current_timestep = 0

        # 生成human
        self.humans = []
        selected_data, selected_next_data = get_human_position_list(self.current_timestep, self.human_df)
        for human_id in range(self.human_num):
            self.humans.append(self.generate_human(human_id, selected_data, selected_next_data))
            self.sync_human_df(human_id, self.current_timestep, 1)

        # 生成robot
        self.robots = []
        for robot_id in range(self.robot_num):
            self.robots.append(self.generate_robot(robot_id))

        self.current_human_aoi_list = np.ones([self.human_num, ])
        self.mean_aoi_timelist = np.ones([self.config.env.num_timestep + 1, ])
        self.mean_aoi_timelist[self.current_timestep] = np.mean(self.current_human_aoi_list)
        self.robot_energy_timelist = np.zeros([self.config.env.num_timestep + 1, self.robot_num])
        self.robot_energy_timelist[self.current_timestep, :] = self.max_uav_energy
        self.robot_x_timelist = np.zeros([self.config.env.num_timestep + 1, self.robot_num])
        self.robot_x_timelist[self.current_timestep, :] = self.nlon / 2
        self.robot_y_timelist = np.zeros([self.config.env.num_timestep + 1, self.robot_num])
        self.robot_y_timelist[self.current_timestep, :] = self.nlat / 2
        self.update_human_timelist = np.zeros([self.config.env.num_timestep, ])
        self.data_collection = 0

        # full和observaeble的概念可以去掉了。
        self.plot_states = list()
        self.robot_actions = list()
        self.rewards = list()
        self.action_values = list()
        self.plot_states.append([[robot.get_obs() for robot in self.robots],
                                 [human.get_obs() for human in self.humans]])

        state = JointState([robot.get_obs() for robot in self.robots], [human.get_obs() for human in self.humans])

        if self.config.env.robot_human_states:
            return state
        else:
            robot_state, human_state = state.to_tensor()
            full_state = torch.cat([robot_state, human_state], dim=0)
            return full_state.flatten()

    def step(self, action):
        new_robot_position = np.zeros([self.robot_num, 2])
        current_enenrgy_consume = np.zeros([self.robot_num, ])
        num_updated_human = 0
        if self.config.env.robot_human_states is False:
            action = self.full_action_space[action]

        for robot_id, robot in enumerate(self.robots):
            new_robot_px = robot.px + action[robot_id][0]  # 形如[0,300] 或[210, 210]
            new_robot_py = robot.py + action[robot_id][1]
            robot_theta = get_theta(0, 0, action[robot_id][0], action[robot_id][1])
            # print(action[robot_id], robot_theta)
            is_stopping = True if (action[robot_id][0] == 0 and action[robot_id][1] == 0) else False
            is_collide = True if judge_collision(new_robot_px, new_robot_py, robot.px, robot.py) else False

            if is_stopping is True:
                consume_energy = consume_uav_energy(0, self.step_time)
            else:
                consume_energy = consume_uav_energy(self.step_time, 0)
            current_enenrgy_consume[robot_id] = consume_energy / self.config.env.max_uav_energy
            new_energy = robot.energy - consume_energy
            self.robot_energy_timelist[self.current_timestep + 1][robot_id] = new_energy

            if is_collide is True:
                new_robot_position[robot_id][0] = robot.px
                new_robot_position[robot_id][1] = robot.py
                self.robot_x_timelist[self.current_timestep + 1][robot_id] = robot.px
                self.robot_y_timelist[self.current_timestep + 1][robot_id] = robot.py
                robot.set(robot.px, robot.py, robot_theta, energy=new_energy)
            else:
                new_robot_position[robot_id][0] = new_robot_px
                new_robot_position[robot_id][1] = new_robot_py
                self.robot_x_timelist[self.current_timestep + 1][robot_id] = new_robot_px  # 值的单位？
                self.robot_y_timelist[self.current_timestep + 1][robot_id] = new_robot_py
                robot.set(new_robot_px, new_robot_py, robot_theta, energy=new_energy)

        selected_data, selected_next_data = get_human_position_list(self.current_timestep + 1, self.human_df)
        delta_human_aoi_list = np.zeros_like(self.current_human_aoi_list)
        for human_id, human in enumerate(self.humans):
            next_px, next_py, next_theta = get_human_position_from_list(self.current_timestep + 1, human_id,
                                                                        selected_data, selected_next_data)
            should_reset = judge_aoi_update([next_px, next_py], new_robot_position)
            if should_reset:
                if human.aoi > 1:
                    delta_human_aoi_list[human_id] = human.aoi
                else:
                    delta_human_aoi_list[human_id] = 1

                human.set(next_px, next_py, next_theta, aoi=1)
                num_updated_human += 1
            else:
                delta_human_aoi_list[human_id] = 0
                new_aoi = human.aoi + 1
                human.set(next_px, next_py, next_theta, aoi=new_aoi)

            self.current_human_aoi_list[human_id] = human.aoi
            self.sync_human_df(human_id, self.current_timestep + 1, human.aoi)

        self.mean_aoi_timelist[self.current_timestep + 1] = np.mean(self.current_human_aoi_list)
        self.update_human_timelist[self.current_timestep] = num_updated_human
        delta_sum_aoi = np.sum(delta_human_aoi_list)
        self.data_collection += (delta_sum_aoi * 0.3)  # Mb, 0.02M/s per person

        reward = self.mean_aoi_timelist[self.current_timestep] - self.mean_aoi_timelist[self.current_timestep + 1] \
                 - self.config.env.energy_factor * np.sum(current_enenrgy_consume)

        if self.agent is not None:
            if hasattr(self.agent.policy, 'action_values'):
                self.action_values.append(self.agent.policy.action_values)
        self.robot_actions.append(action)
        self.rewards.append(reward)
        self.plot_states.append([[robot.get_obs() for robot in self.robots],
                                 [human.get_obs() for human in self.humans]])

        next_state = JointState([robot.get_obs() for robot in self.robots],
                                [human.get_obs() for human in self.humans])

        self.current_timestep += 1
        if self.current_timestep >= self.num_timestep:
            done = True
        else:
            done = False
        info = {"performance_info": {
            "mean_aoi": np.mean(self.mean_aoi_timelist),
            "mean_energy_consumption": 1.0 - (
                    np.mean(self.robot_energy_timelist[self.current_timestep]) / self.max_uav_energy),
            "collected_data_amount": self.data_collection / (self.num_timestep * self.human_num * 0.3),
            "human_coverage": np.mean(self.update_human_timelist) / self.human_num
        },
        }

        if self.config.env.robot_human_states:
            return next_state, reward, done, info
        else:
            robot_state, human_state = next_state.to_tensor()
            full_next_state = torch.cat([robot_state, human_state], dim=0)
            return full_next_state.flatten(), reward, done, info

    def render(self, mode='traj', output_file=None, plot_loop=False, moving_line=False, args=None,
               user_uav1_att_list=None, user_uav2_att_list=None, uav1_user_att_list=None, uav2_user_att_list=None):
        # -------------------------------------------------------------------
        assert args.dataset in ['Purdue', 'NCSU', 'KAIST']
        import geopandas as gpd
        import movingpandas as mpd
        from movingpandas.geometry_utils import measure_distance_geodesic

        max_distance_x = measure_distance_geodesic(Point(self.lower_left[0], self.lower_left[1]),
                                                   Point(self.upper_right[0], self.lower_left[1]))
        max_distance_y = measure_distance_geodesic(Point(self.lower_left[0], self.lower_left[1]),
                                                   Point(self.lower_left[0], self.upper_right[1]))

        lon_per_grid = (self.upper_right[0] - self.lower_left[0]) / self.nlon
        lat_per_grid = (self.upper_right[1] - self.lower_left[1]) / self.nlat


        # 根据训练结果生成robot_df，id从-1开始递减。可以append到human的dataframe中
        robot_df = pd.DataFrame()
        for i in range(self.robot_num):
            x_list = self.robot_x_timelist[:, i]
            y_list = self.robot_y_timelist[:, i]
            id_list = np.ones_like(x_list) * (-i - 1)
            aoi_list = np.ones_like(x_list) * (-1)
            energy_list = self.robot_energy_timelist[:, i]
            timestamp_list = [self.start_timestamp + i * self.step_time for i in range(self.num_timestep + 1)]
            x_distance_list = x_list * max_distance_x / self.nlon + max_distance_x / self.nlon / 2
            y_distance_list = y_list * max_distance_y / self.nlat + max_distance_y / self.nlat / 2
            max_longitude = abs(self.lower_left[0] - self.upper_right[0])
            max_latitude = abs(self.lower_left[1] - self.upper_right[1])
            longitude_list = x_list * max_longitude / self.nlon + max_longitude / self.nlon / 2 + self.lower_left[0]
            latitude_list = y_list * max_latitude / self.nlat + max_latitude / self.nlat / 2 + self.lower_left[1]

            data = {"id": id_list, "longitude": longitude_list, "latitude": latitude_list,
                    "x": x_list, "y": y_list, "x_distance": x_distance_list, "y_distance": y_distance_list,
                    "timestamp": timestamp_list, "aoi": aoi_list, "energy": energy_list}
            tmp_df = pd.DataFrame(data)
            tmp_df['t'] = pd.to_datetime(tmp_df['timestamp'], unit='s')  # s表示时间戳转换
            robot_df = robot_df.append(tmp_df)

        # ======将训练好AoI的human_df和robot_df拼在一起写入外存======
        # case1
        mixed_df = self.human_df.copy()
        mixed_df = mixed_df.append(robot_df)
        mixed_df.to_csv(
            'envs/crowd_sim/trained_dataset_2021_11_13/trained__' + args.model_dir[args.model_dir.index('/') + 1:] + '.csv')

        # case2
        # robot_df.to_csv(
        #   'envs/crowd_sim/robot_dataset_2021_11_13/robot__' + args.model_dir[args.model_dir.index('/') + 1:] + '.csv')

        # ==========================================================



        # ----选择指定users，并裁剪数据集到指定时间范围（得到时间、user子集的mixed_df）
        # args.users = ','.join([str(i) for i in range(33)])  # 这样硬编码画全部users
        clip, users = args.clip, args.users
        clip = int(clip.split(',')[0]), int(clip.split(',')[1])
        users = users.split(',')
        users = [int(user) for user in users]

        def func(groupDf, users):
            if users[0] != -1 and groupDf.id.values[0] not in users + [-1, -2]:  # 无人机无条件绘制，当users为-1时表示全部绘制
                return pd.DataFrame()
            if clip[0] != -1:
                return groupDf.iloc[clip[0]:clip[1] + 1]
            else:
                return groupDf.iloc[0:groupDf.shape[0]]

        mixed_df = mixed_df.groupby('id').apply(func, users)
        if isinstance(mixed_df.index[0], tuple):  # 需要删除多重索引
            mixed_df.pop('id')
            mixed_df = mixed_df.reset_index()
            if 'level_1' in mixed_df.columns:
                mixed_df.pop('level_1')


        import matplotlib.pyplot as plt
        _, ax1 = plt.subplots()
        x = [i for i in range(121)] if clip[0] == -1 else [i for i in range(clip[0], clip[1] + 1)]
        for user in users:
            y_axis = np.array(mixed_df[mixed_df['id'] == user].aoi)
            # y_axis = y_axis/max(y_axis)  # optional: AoI归一化到[0,1]
            subAoI1 = np.where(y_axis == 1)
            x_fixed = [x[i] - 0.9999 if np.isin(i, subAoI1) else x[i] for i in range(len(y_axis))]
            y_fixed = [0 if np.isin(i, subAoI1) else y_axis[i] for i in range(len(y_axis))]
            if user_uav1_att_list is not None:
                ax1.plot(x_fixed, y_fixed, label='user' + str(user), color='black', linewidth=3)  # optional: att
            else:
                ax1.plot(x_fixed, y_fixed, label='user' + str(user))  # optional: AoI

        if user_uav1_att_list is not None:
            plt.xticks(x[::10])
            ax2 = ax1.twinx()  # 双y轴
            x = [i for i in range(121 - 1)]  # -1是因为在终止状态不需要做决策
            # 如果需要clip，四个列表也需要clip
            #ax2.plot(x, user_uav1_att_list, label='user_uav1_att')
            #ax2.plot(x, user_uav2_att_list, label='user_uav2_att')
            ax2.plot(x, uav1_user_att_list, label='uav1_user_att', color='blue')
            ax2.plot(x, uav2_user_att_list, label='uav2_user_att', color='red')
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
            ax2.set_ylabel('attention weight')

        else:
            plt.xticks(x[::2])
            handles1, labels1 = ax1.get_legend_handles_labels()
            plt.legend(handles1, labels1, loc='upper left')



        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Current AoI')
        plt.subplots_adjust(left=0.2, right=0.8)  # 调整距离左右的间距
        if user_uav1_att_list is not None:
            plt.savefig(f"logs/debug/2021_11_13_TKDE/{args.users}" + "_att" + ".pdf", format="pdf")  # optional:att
        else:
            plt.grid()
            plt.savefig(f"logs/debug/2021_11_13_TKDE/{args.users}" + "_AoI" + ".pdf", format="pdf")  # optional:AoI
        plt.show()

        # =====建立moving pandas轨迹，也可以选择调用高级API继续清洗轨迹。=====
        mixed_gdf = gpd.GeoDataFrame(mixed_df, geometry=gpd.points_from_xy(mixed_df.longitude, mixed_df.latitude),
                                     crs=4326)
        mixed_gdf = mixed_gdf.set_index('t').tz_localize(None)  # tz=time zone, 以本地时间为准
        mixed_gdf = mixed_gdf.sort_values(by=["id", "t"], ascending=[True, True])
        trajs = mpd.TrajectoryCollection(mixed_gdf, 'id')
        # trajs = mpd.MinTimeDeltaGeneralizer(trajs).generalize(tolerance=timedelta(minutes=1))
        # for index, traj in enumerate(trajs.trajectories):
        #     print(f"id: {trajs.trajectories[index].df['id'][0]}"
        #           + f"  size:{trajs.trajectories[index].size()}")

        start_point = trajs.trajectories[0].get_start_location()

        # 经纬度反向
        m = folium.Map(location=[start_point.y, start_point.x], tiles="cartodbpositron", zoom_start=14, max_zoom=24)

        m.add_child(folium.LatLngPopup())
        if args.dataset == 'Purdue':
            position = 'bottomleft'
        elif args.dataset == 'NCSU':
            position = 'bottomright'
        else:
            position = 'topright'
        minimap = folium.plugins.MiniMap(position=position)
        m.add_child(minimap)
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.TileLayer('Stamen Toner').add_to(m)
        folium.TileLayer('cartodbpositron').add_to(m)
        folium.TileLayer('OpenStreetMap').add_to(m)

        # 锁定范围
        # grid_geo_json = get_border(self.upper_right, self.lower_left)
        # color = "red"
        # border = folium.GeoJson(grid_geo_json,
        #                         style_function=lambda feature, color=color: {
        #                             'fillColor': color,
        #                             'color': "black",
        #                             'weight': 2,
        #                             'dashArray': '5,5',
        #                             'fillOpacity': 0,
        #                         })
        # m.add_child(border)

        for index, traj in enumerate(trajs.trajectories):
            name = f"UAV {index}" if index < self.robot_num else f"Human {traj.df['id'][0]}"  # select human
            # name = f"UAV {index}" if index < robot_num else f"Human {index - robot_num}"

            # randr = lambda: np.random.randint(0, 255)
            # color_base = (randr(), randr(), randr())
            if index == 0:
                color_base = (255, 0, 0)  # 无人机1颜色
            elif index == 1:
                color_base = (0, 0, 255)  # 无人机2颜色
            else:
                color_base = (0, 0, 0)  # user颜色

            # line
            # if index < self.robot_num:  # 仅为无人机绘制轨迹
            #     geo_col = traj.to_point_gdf().geometry
            #     # print('轨迹点的数量：', geo_col.shape[0])
            #     xy = [[y, x] for x, y in zip(geo_col.x, geo_col.y)]
            #     f1 = folium.FeatureGroup(name)
            #     # color渐变
            #     color = '#%02X%02X%02X' % color_base  # method1:强度不变
            #     # color = '#%02X%02X%02X' % hsi2rgb(rgb2hsi(color_base, seti=(s+1)/seg_len))  # method2:强度从0到1 (有点浮夸)
            #     # color = '#%02X%02X%02X' % hsi2rgb(rgb2hsi(color_base, seti=0.25+(s+1)/(2*seg_len)))  # method3:强度从0.25到0.75
            #     # print(color)
            #     weight = 6 if args.dataset == 'Purdue' else 4
            #     if moving_line:  # moving_line这个参数仅影响ui，没有实质意义
            #         AntPath(locations=xy, color=color, weight=weight, opacity=0.7, dash_array=[100, 20],
            #                 delay=1000).add_to(f1)
            #     else:
            #         # 透明度渐变，也可以随着绘制线条不断加粗
            #         # folium.PolyLine(locations=xy, color=color, weight=4+0.01*s, opacity=0.333+(s+1)/(1.5*seg_len)).add_to(f1)
            #         folium.PolyLine(locations=xy, color=color, weight=weight, opacity=1).add_to(f1)  # kaist4NCSU4Purdue6
            #     f1.add_to(m)

            # line 绘制彗星需要在这里搞~(备份，渐变)
            # if index < self.robot_num:  # 仅为无人机绘制轨迹
            #     geo_col = traj.to_point_gdf().geometry
            #     # print('轨迹点的数量：', geo_col.shape[0])
            #     seg_len = geo_col.shape[0] - 1
            #     for s in range(seg_len):
            #         xy = [[y, x] for x, y in zip(geo_col.x[s:s + 2], geo_col.y[s:s + 2])]
            #         f1 = folium.FeatureGroup(name)
            #         # color渐变
            #         color = '#%02X%02X%02X' % color_base  # method1:强度不变
            #         # color = '#%02X%02X%02X' % hsi2rgb(rgb2hsi(color_base, seti=(s+1)/seg_len))  # method2:强度从0到1 (有点浮夸)
            #         # color = '#%02X%02X%02X' % hsi2rgb(rgb2hsi(color_base, seti=0.25+(s+1)/(2*seg_len)))  # method3:强度从0.25到0.75
            #         # print(color)
            #         if moving_line:  # moving_line这个参数仅影响ui，没有实质意义
            #             AntPath(locations=xy, color=color, weight=4, opacity=0.7, dash_array=[100, 20],
            #                     delay=1000).add_to(f1)
            #         else:
            #             # 透明度渐变，也可以随着绘制线条不断加粗
            #             # folium.PolyLine(locations=xy, color=color, weight=4+0.01*s, opacity=0.333+(s+1)/(1.5*seg_len)).add_to(f1)
            #             folium.PolyLine(locations=xy, color=color, weight=6 + 0.01 * s, opacity=1).add_to(f1)  # kaist4NCSU4Purdue6
            #         f1.add_to(m)

            # point
            f2 = folium.FeatureGroup('dots')
            features = traj_to_timestamped_geojson(index, traj, self.robot_num, color_base, args)
            TimestampedGeoJson(
                {
                    "type": "FeatureCollection",
                    "features": features,
                },
                period="PT15S",
                add_last_point=True,
                transition_time=5,
                loop=plot_loop,
            ).add_to(m)  # sub_map

        folium.LayerControl().add_to(m)

        if self.config.env.tallest_locs is not None:
            # 绘制正方形
            for tallest_loc in self.config.env.tallest_locs:
                # folium.Rectangle(
                #     bounds=[(tallest_loc[0] + 0.00025, tallest_loc[1] + 0.0003),
                #             (tallest_loc[0] - 0.00025, tallest_loc[1] - 0.0003)],  # 解决经纬度在地图上的尺度不一致
                #     color="black",
                #     fill=True,
                # ).add_to(m)
                icon_square = folium.plugins.BeautifyIcon(
                    icon_shape='rectangle-dot',
                    border_color='red',
                    border_width=8,
                )
                folium.Marker(location=[tallest_loc[0], tallest_loc[1]],
                              popup=folium.Popup(html=f'<p>raw coord: ({tallest_loc[1]},{tallest_loc[0]})</p>'),
                              tooltip='High-rise building',
                              icon=icon_square).add_to(m)

        m.get_root().render()
        m.get_root().save(output_file)
        logging.info(f"{output_file} saved!")
        # -----------------------------------------------------------------------------
