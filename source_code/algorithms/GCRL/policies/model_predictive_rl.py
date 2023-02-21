import logging
import time

import copy
import numpy as np
import torch.multiprocessing as mp
from algorithms.GCRL.policies.base import Policy
from algorithms.GCRL.envs.model.mdp import build_action_space
from algorithms.GCRL.envs.model.utils import *
from algorithms.GCRL.configs.config import BaseEnvConfig
from algorithms.GCRL.method.state_predictor import StatePredictor  # 网络1
from algorithms.GCRL.method.graph_model import RGL
from algorithms.GCRL.method.value_estimator import ValueEstimator  # 网络2


class ModelPredictiveRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'ModelPredictiveRL'
        self.trainable = True
        self.epsilon = None
        self.gamma = None
        self.action_space = None
        self.action_values = None
        self.share_graph_model = None
        self.value_estimator = None
        self.state_predictor = None
        self.planning_depth = None
        self.planning_width = None
        self.do_action_clip = None
        self.robot_state_dim = None
        self.human_state_dim = None
        self.device = None

    def configure(self, config, human_df):
        self.config = config
        self.gamma = config.rl.gamma
        self.robot_state_dim = config.model_predictive_rl.robot_state_dim  # 4
        self.human_state_dim = config.model_predictive_rl.human_state_dim  # 4
        self.planning_depth = config.model_predictive_rl.planning_depth
        # self.do_action_clip = config.model_predictive_rl.do_action_clip
        self.do_action_clip = False  # hard-code
        self.planning_width = config.model_predictive_rl.planning_width  # 2021/10/19记录 仅在action_clip_single_process和V_planning中用于裁剪动作空间
        self.share_graph_model = config.model_predictive_rl.share_graph_model
        self.human_df = human_df
        self.tmp_config = BaseEnvConfig()

        if self.share_graph_model:
            raise NotImplementedError
        else:
            # graph_model1 = RGL(config, self.robot_state_dim, self.human_state_dim)
            self.value_estimator = ValueEstimator(config, None, None, self.device)
            # graph_model2 = RGL(config, self.robot_state_dim, self.human_state_dim)
            self.state_predictor = StatePredictor(config, None, self.device)
            self.model = [self.value_estimator.value_network, self.state_predictor.mlp]  # 魔改哈哈哈

        if tmp_config.env.rollout_num == 1:
            for model in self.model:
                model.to(self.device)
        else:
            mp.set_start_method('spawn')
            for model in self.model:
                model.share_memory()
                model.to(self.device)

        # logging.info('Planning depth: {}'.format(self.planning_depth))
        # logging.info('Planning width: {}'.format(self.planning_width))
        if self.planning_depth > 1 and not self.do_action_clip:
            logging.warning('Performing d-step planning without action space clipping!')

        # env config
        self.human_num = self.tmp_config.env.human_num
        self.robot_num = self.tmp_config.env.robot_num
        self.num_timestep = self.tmp_config.env.num_timestep
        self.step_time = self.tmp_config.env.step_time
        self.start_timestamp = self.tmp_config.env.start_timestamp
        self.max_uav_energy = self.tmp_config.env.max_uav_energy

    def set_device(self, device):
        self.device = device

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_value_estimator(self):
        return self.value_estimator

    def get_state_dict(self):
        if self.state_predictor.trainable:

            return {
                'state_predictor': self.state_predictor.mlp.state_dict(),
                'value_estimator': self.value_estimator.value_network.state_dict(),
            }


    def load_state_dict(self, checkpoint):
        if self.state_predictor.trainable:
            self.state_predictor.mlp.load_state_dict(checkpoint['state_predictor'])
            self.value_estimator.value_network.load_state_dict(checkpoint['value_estimator'])


    def save_model(self, file):
        torch.save(self.get_state_dict(), file)  # self.get_state_dict() returns a dict

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)

    def predict(self, state, current_timestep):
        state = torch.tensor(state).float().to(self.device)
        n_thread = state.shape[0]
        n_agent = state.shape[1]
        '''state is already a tensor with batch_size dim
        '''
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        # if self.reach_destination(state):
        #     return ActionXY(0, 0)
        if self.action_space is None:
            # self.action_space = build_action_space()
            self.action_space = [9, self.config.update_num]

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:  # 探索
            # acts = torch.zeros((n_thread, n_agent, 2)).long()
            a1 = np.random.randint(self.action_space[0], size=(n_thread, n_agent))
            a2 = np.random.randint(self.action_space[1], size=(n_thread, n_agent))
            acts = np.stack([a1, a2], axis=-1)
            acts = torch.tensor(acts)
        else:  # 利用
            # --- step1. 根据是否执行动作裁剪，为action_space_clipped定值
            if self.do_action_clip:
                raise NotImplementedError  # necessary?
                # state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
                # state_tensor = copy.deepcopy(state.reshape(n_thread, -1))
                # if tmp_config.env.rollout_num == 1:
                #     action_space_clipped = self.action_clip_single_process(state_tensor, self.action_space,
                #                                                            self.planning_width,
                #                                                            current_timestep)  # 返回的就是经过planning后选择的N步回报最大的动作
                # else:
                #     action_space_clipped = self.action_clip_multi_processing(state_tensor, self.action_space,
                #                                                              self.planning_width,                                                                 current_timestep)
            else:
                action_space_clipped = self.action_space

            # --- step2. 在裁剪后的动作空间中，根据MCTS选择最佳动作，为max_action定值
            acts = []
            for i in range(n_agent):  # 每个agent分别搜索 max_action
                # max_action = torch.full((n_thread, 2), 0).to(self.device)  # 两维动作
                # max_value = torch.full((n_thread, 1), float('-inf')).to(self.device)
                max_action = None
                max_value = float('-inf')
                from itertools import product
                adim1, adim2 = action_space_clipped[0], action_space_clipped[1]
                for action in product(range(adim1), range(adim2)):
                    action = torch.tile(torch.tensor(action).to(self.device), (n_thread, 1))  # 添加thread维度 shape = (n_thread, 2)
                    # state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
                    state_i = state[:, i, :]
                    next_state_i = self.state_predictor(state_i, action)  # p(s'|s,a)  # 所有agent共享同一个model
                    # 调用V_planning的入口。向前看N个step，获得所有下一状态的V值。
                    max_next_return, max_next_traj = self.V_planning(next_state_i, self.planning_depth, self.planning_width,
                                                                     current_timestep)
                    reward_est = self.estimate_reward(state_i, action, current_timestep)  # 所有agent共享同一个model
                    value = reward_est + self.gamma * max_next_return
                    if value.mean() > max_value:
                        max_value = value.mean()
                        max_action = action
                if max_action is None:
                    print(max_action)
                    raise ValueError('Value network is not well trained.')
                acts.append(max_action)
            acts = torch.stack(acts)  # shape = (agent, thread, dim)
            acts = torch.permute(acts, (1, 0, 2))  # shape = (thread, agent, dim)

        self.last_state = state

        return acts

    def action_clip_single_process(self, state, action_space, width, current_timestep):
        '''
        在以s为根节点的子树中，初步筛选出最佳的width个动作
        注意由于depth=1，所以这个函数相当于是个浅层的MCTS,仅根据r(s,a) + \gamma * V(s')衡量动作好坏
        '''
        values = []
        depth = 1
        # logging.info("start")
        for action in action_space:
            next_state_est = self.state_predictor(state, action)  # 每次环境预测器想要通过s和a预测s'时，需要先将s经过图网络抽取特征,得到环境预测器的输入。
            next_return, _ = self.V_planning(next_state_est, depth, width, current_timestep)
            reward_est = self.estimate_reward(state, action, current_timestep)
            value = reward_est + self.gamma * next_return
            values.append(value.item())
        # logging.info("end")
        max_indexes = np.argpartition(np.array(values), -width)[-width:]  # 选择value最大的前width个动作
        clipped_action_space = np.array([action_space[i] for i in max_indexes])

        # print(clipped_action_space)
        return clipped_action_space

    def action_value_estimate(self, current_dim, values, state, actions, current_timestep):
        for index, action in enumerate(actions):
            next_state_est = self.state_predictor(state, action)
            next_return = self.value_estimator(next_state_est)
            reward_est = self.estimate_reward(state, action, current_timestep)
            value = reward_est + self.gamma * next_return
            values[current_dim + index] = value.item()

    def any_process_alive(self, processes):
        for p in processes:
            if p.is_alive():
                return True
        return False

    def action_clip_multi_processing(self, state, action_space, width, current_timestep):
        # logging.info("start")
        values = torch.zeros([pow(9, self.tmp_config.env.robot_num), ], requires_grad=False)
        values.share_memory_()
        current_dim = 0
        processes = []

        for actions in np.array_split(action_space, tmp_config.env.rollout_num, axis=0):
            p = mp.Process(target=self.action_value_estimate,
                           args=(current_dim, values, [s.detach() for s in state], actions, current_timestep))
            current_dim += actions.shape[0]
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        while True:
            if self.any_process_alive(processes):
                time.sleep(1)
            else:
                # print(values)
                max_indexes = torch.topk(values, width).indices
                clipped_action_space = np.array([action_space[i] for i in max_indexes])
                del values
                while True:
                    if self.any_process_alive(processes):
                        for p in processes:
                            p.close()
                    else:
                        break
                # logging.info("end")
                return clipped_action_space

    def V_planning(self, state, depth, width, current_timestep):  # 递归
        '''
        以state为根节点的子树中，返回最大return。
        '''
        current_state_value = self.value_estimator(state)
        if depth == 1:  # 递归出口
            return current_state_value, [(state, None, None)]

        if self.do_action_clip:
            if tmp_config.env.rollout_num == 1:
                action_space_clipped = self.action_clip_single_process(state, self.action_space, width,
                                                                       current_timestep)
            else:
                action_space_clipped = self.action_clip_multi_processing(state, self.action_space, width,
                                                                         current_timestep)
        else:
            action_space_clipped = self.action_space

        returns = []
        trajs = []

        for action in action_space_clipped:
            next_state_est = self.state_predictor(state, action)
            reward_est = self.estimate_reward(state, action, current_timestep)
            next_value, next_traj = self.V_planning(next_state_est, depth - 1, self.planning_width, current_timestep)  # 递归调用V_planning,深度-1
            return_value = current_state_value / depth + (depth - 1) / depth * (self.gamma * next_value + reward_est)  # 关键代码，论文中的式13

            returns.append(return_value.item())
            trajs.append([(state, action, reward_est)] + next_traj)

        max_index = np.argmax(returns)
        max_return = returns[max_index]
        max_traj = trajs[max_index]

        return max_return, max_traj

    # 不太好改
    def estimate_reward(self, state, action, current_timestep):
        '''
        param：
            state：当前状态s
            action：假设当前采取的动作a
        return：
            reward：r(s,a)
        '''
        # if isinstance(state, list) or isinstance(state, tuple):
        #     state = tensor_to_joint_state(state)
        # human_states = state.human_states
        # robot_states = state.robot_states
        # current_human_aoi_list = np.zeros([self.human_num, ])
        # next_human_aoi_list = np.zeros([self.human_num, ])
        # current_uav_position = np.zeros([self.robot_num, 2])
        # new_robot_position = np.zeros([self.robot_num, 2])
        # current_robot_enenrgy_list = np.zeros([self.robot_num, ])
        # next_robot_enenrgy_list = np.zeros([self.robot_num, ])
        # current_enenrgy_consume = np.zeros([self.robot_num, ])
        # num_updated_human = 0
        #
        # for robot_id, robot in enumerate(robot_states):
        #     new_robot_px = robot.px + action[robot_id][0]
        #     new_robot_py = robot.py + action[robot_id][1]
        #     is_stopping = True if (action[robot_id][0] == 0 and action[robot_id][1] == 0) else False
        #     is_collide = True if judge_collision(new_robot_px, new_robot_py, robot.px, robot.py) else False
        #
        #     if is_stopping is True:
        #         consume_energy = consume_uav_energy(0, self.step_time)
        #     else:
        #         consume_energy = consume_uav_energy(self.step_time, 0)
        #     current_enenrgy_consume[robot_id] = consume_energy / tmp_config.env.max_uav_energy
        #     new_energy = robot.energy - consume_energy
        #
        #     current_uav_position[robot_id][0] = robot.px
        #     current_uav_position[robot_id][1] = robot.py
        #     if is_collide:
        #         new_robot_position[robot_id][0] = robot.px
        #         new_robot_position[robot_id][1] = robot.py
        #     else:
        #         new_robot_position[robot_id][0] = new_robot_px
        #         new_robot_position[robot_id][1] = new_robot_py
        #     current_robot_enenrgy_list[robot_id] = robot.energy
        #     next_robot_enenrgy_list[robot_id] = new_energy
        #
        # selected_data, selected_next_data = get_human_position_list(current_timestep + 1, self.human_df)
        #
        # for human_id, human in enumerate(human_states):  # 遍历所有用户，判断用户的数据是否被收集
        #     current_human_aoi_list[human_id] = human.aoi
        #     next_px, next_py, next_theta = get_human_position_from_list(current_timestep + 1, human_id, selected_data,
        #                                                                 selected_next_data)
        #     should_reset = judge_aoi_update([next_px, next_py], new_robot_position)
        #     if should_reset:
        #         next_human_aoi_list[human_id] = 1
        #         num_updated_human += 1
        #     else:
        #         next_human_aoi_list[human_id] = human.aoi + 1
        #
        # reward = np.mean(current_human_aoi_list - next_human_aoi_list) \
        #          - tmp_config.env.energy_factor * np.sum(current_enenrgy_consume)

        n_thread = state.shape[0]
        reward = torch.zeros((n_thread, 1)).to(self.device)  # TODO 只有形状对，先这样，没有训练痕迹再回来加
        return reward
