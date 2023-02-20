import logging
import matplotlib.pyplot as plt
import torch
import numpy as np
from algorithms.GCRL.envs.model.utils import *

class Explorer(object):
    def __init__(self, envs, robot, device, writer=None, memory=None, gamma=None, target_policy=None):
        self.envs = envs
        self.robot = robot
        self.device = device
        self.writer = writer
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.statistics = None

    # @profile
    def run_k_episodes(self, k, phase, plot_index, update_memory=False):

        self.robot.policy.set_phase(phase)
        cumulative_rewards = []
        average_return_list = []
        mean_aoi_list = []
        mean_energy_consumption_list = []
        collected_data_amount_list = []
        update_human_coverage_list = []


        from algorithms.GCRL.policies.gcn import GCN
        # flag = 0 if isinstance(self.robot.policy, GCN) else 1  # 模型为gcn时，没有A1
        # A0_list = []
        # A0_std_list = []
        # if flag:
        #     A1_list = []
        #     A1_std_list = []


        for ep_i in range(k):
            state = self.envs.reset()  # shape = (n_thread, n_agent, dim)  # 157 here
            done = False
            states = []
            actions = []
            rewards = []
            returns = []

            s = 0
            while not done:
                action = self.robot.act(state, self.envs.get_step_count())
                # if s % 2 == 0:
                #     action = noisy_action(action)
                # print(action)
                # print(self.envs.start_timestamp+self.envs.current_timestep*self.envs.step_time,action)
                state, reward, done, info = self.envs.step(np.array(action.cpu()))  # 东西存在info里
                done = done.any()
                print('step {}, reward={}'.format(s, reward))

                s += 1
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)
                if done:
                    pass
                    # TODO 应该调用我的summary_info()

            if update_memory:
                self.update_memory(states, actions, rewards)  # 这里存的actions不对
                # if isinstance(info, ReachGoal) or isinstance(info, Collision):
                #     # only add positive(success) or negative(collision) experience in experience set
                #     self.update_memory(states, actions, rewards, imitation_learning)

            # calculate Bellman cumulative reward
            cumulative_rewards.append(sum([pow(self.gamma, t) * reward for t, reward in enumerate(rewards)]))
            for step in range(len(rewards)):
                step_return = sum([pow(self.gamma, t) * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            average_return_list.append(average(returns))



            logging.info(f"cumulative_rewards:{average(cumulative_rewards)}, "
                         f"return:{average(average_return_list)},  "
                         f"mean_aoi: {average(mean_aoi_list)},  "
                         f"mean_energy_consumption: {average(mean_energy_consumption_list)}  "
                         f"collected_data_ratio: {average(collected_data_amount_list)}  "
                         f"user_coverage: {average(update_human_coverage_list)}")

            self.statistics = average(cumulative_rewards), average(average_return_list), average(mean_aoi_list), \
                              average(mean_energy_consumption_list), average(collected_data_amount_list), \
                              average(update_human_coverage_list)




        return self.statistics


    def update_memory(self, states, actions, rewards):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states[:-1]):  # 这里面的肯定不对。。。
            # action = actions[i]
            reward = rewards[i]
            next_state = states[i + 1]
            if i == len(states) - 1:
                # terminal state
                value = reward
            else:
                value = 0

            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)
            action = actions[i]
            if not type(action) == torch.Tensor:
                action = torch.Tensor(action).to(self.device)
            else:
                action = action.to(self.device)

            self.memory.push((state, action, value, reward, next_state))


    def log(self, tag_prefix, global_step):
        reward, avg_return, aoi, energy_consumption, collected_data_amount, ave_human_coverage = self.statistics
        # good_reward_list stuck here

        self.writer.add_scalar(tag_prefix + '/reward', np.mean(reward), global_step)
        self.writer.add_scalar(tag_prefix + '/avg_return', np.mean(avg_return), global_step)
        self.writer.add_scalar(tag_prefix + '/mean_human_aoi', aoi, global_step)
        self.writer.add_scalar(tag_prefix + '/energy_consumption (J)', energy_consumption, global_step)
        self.writer.add_scalar(tag_prefix + '/collected_data_amount (MB)', collected_data_amount, global_step)
        self.writer.add_scalar(tag_prefix + '/avg user coverage', ave_human_coverage, global_step)



def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
