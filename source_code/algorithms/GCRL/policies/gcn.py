from algorithms.GCRL.policies.base import Policy
from algorithms.GCRL.envs.model.utils import *
from algorithms.GCRL.method.graph_model import RGL
from algorithms.GCRL.method.value_estimator import ValueEstimator
from algorithms.GCRL.configs.config import BaseEnvConfig

class GCN(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'ModelPredictiveRL'
        self.trainable = True
        self.epsilon = None
        self.gamma = None
        self.action_space = None
        self.action_values = None
        self.value_estimator = None
        self.do_action_clip = None
        self.robot_state_dim = None
        self.human_state_dim = None
        self.device = None

    def configure(self, config, human_df):
        self.gamma = config.rl.gamma
        self.robot_state_dim = config.gcn.robot_state_dim
        self.human_state_dim = config.gcn.human_state_dim
        self.human_df = human_df
        self.tmp_config = BaseEnvConfig()

        graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
        self.value_estimator = ValueEstimator(config, graph_model, config.gcn.planning_dims)
        self.model = [graph_model, self.value_estimator.value_network]

        for model in self.model:
            model.to(self.device)

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
        return {
            'graph_model': self.value_estimator.graph_model.state_dict(),
            'value_network': self.value_estimator.value_network.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.value_estimator.graph_model.load_state_dict(state_dict['graph_model'])
        self.value_estimator.value_network.load_state_dict(state_dict['value_network'])

    def save_model(self, file):
        torch.save(self.get_state_dict(), file)

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)

    def predict(self, state, current_timestep):
        if self.action_space is None:
            self.action_space = build_action_space()
        action_value=np.zeros([len(self.action_space),])
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            max_action = None
            max_value = float('-inf')

            for action_index,action in enumerate(self.action_space):
                state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
                reward,next_state = self.estimate_reward_nextstate(state_tensor, action, current_timestep)
                next_return=self.value_estimator(next_state)
                value = reward + self.gamma * next_return
                if value > max_value:
                    max_value = value
                    max_action = action

        # print(max_action)
        self.last_state = state.to_tensor(device=self.device)

        return max_action

    def estimate_reward_nextstate(self, state, action, current_timestep):
        if isinstance(state, list) or isinstance(state, tuple):
            state_clone = tensor_to_joint_state(state)  # 还原尺度！
        else:
            raise NotImplementedError
        human_states = state_clone.human_states
        robot_states = state_clone.robot_states
        current_human_aoi_list = np.zeros([self.human_num, ])
        next_human_aoi_list = np.zeros([self.human_num, ])
        current_uav_position = np.zeros([self.robot_num, 2])
        new_robot_position = np.zeros([self.robot_num, 2])
        current_robot_enenrgy_list = np.zeros([self.robot_num, ])
        next_robot_enenrgy_list = np.zeros([self.robot_num, ])
        current_enenrgy_consume = np.zeros([self.robot_num, ])
        num_updated_human = 0
        next_robot_state_list = []
        next_human_state_list = []

        for robot_id, robot in enumerate(robot_states):
            new_robot_px = robot.px + action[robot_id][0]
            new_robot_py = robot.py + action[robot_id][1]
            robot_theta = get_theta(0, 0, action[robot_id][0], action[robot_id][1])
            is_stopping = True if (action[robot_id][0] == 0 and action[robot_id][1] == 0) else False
            is_collide = True if judge_collision(new_robot_px, new_robot_py, robot.px, robot.py) else False

            if is_stopping is True:
                consume_energy = consume_uav_energy(0, self.step_time)
            else:
                consume_energy = consume_uav_energy(self.step_time, 0)
            current_enenrgy_consume[robot_id] = consume_energy / tmp_config.env.max_uav_energy
            new_energy = robot.energy - consume_energy

            current_uav_position[robot_id][0] = robot.px
            current_uav_position[robot_id][1] = robot.py
            if is_collide:
                new_robot_position[robot_id][0] = robot.px
                new_robot_position[robot_id][1] = robot.py
            else:
                new_robot_position[robot_id][0] = new_robot_px
                new_robot_position[robot_id][1] = new_robot_py
            current_robot_enenrgy_list[robot_id] = robot.energy
            next_robot_enenrgy_list[robot_id] = new_energy
            next_robot_state_list.append([new_robot_position[robot_id][0]/tmp_config.env.nlon,
                                    new_robot_position[robot_id][1]/tmp_config.env.nlat,
                                    robot_theta/tmp_config.env.rotation_limit,
                                    new_energy/ tmp_config.env.max_uav_energy])

        selected_data, selected_next_data = get_human_position_list(current_timestep + 1, self.human_df)

        for human_id, human in enumerate(human_states):
            current_human_aoi_list[human_id] = human.aoi
            next_px, next_py, next_theta = get_human_position_from_list(current_timestep + 1, human_id, selected_data,
                                                                        selected_next_data)
            should_reset = judge_aoi_update([next_px, next_py], new_robot_position)
            if should_reset:
                next_human_aoi_list[human_id] = 1
                num_updated_human += 1
            else:
                next_human_aoi_list[human_id] = human.aoi + 1

            next_human_state_list.append([next_px / tmp_config.env.nlon,
                                    next_py / tmp_config.env.nlat,
                                    next_theta / tmp_config.env.rotation_limit,
                                    next_human_aoi_list[human_id] / tmp_config.env.num_timestep])

        reward = np.mean(current_human_aoi_list - next_human_aoi_list) \
                 - tmp_config.env.energy_factor * np.sum(current_enenrgy_consume)

        next_robot_states = torch.tensor(next_robot_state_list, dtype=torch.float32)
        next_human_states = torch.tensor(next_human_state_list, dtype=torch.float32)

        return reward, [next_robot_states.unsqueeze(0).to(self.device),
                        next_human_states.unsqueeze(0).to(self.device)]
