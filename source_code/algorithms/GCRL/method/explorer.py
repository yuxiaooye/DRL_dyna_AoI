import logging
import matplotlib.pyplot as plt
import torch
import numpy as np
from algorithms.GCRL.envs.model.utils import *
from algorithms.algo.main import write_output

class Explorer(object):
    def __init__(self, envs, dummy_env, robot, device, input_args, logger=None,
                 memory=None, gamma=None, target_policy=None):
        self.envs = envs
        self.dummy_env = dummy_env
        self.robot = robot
        self.device = device
        self.input_args = input_args
        self.logger = logger
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.statistics = None

        self.best_episode_reward = float('-inf')

    # @profile
    def run_k_episodes(self, k, phase, plot_index, update_memory=False):

        self.robot.policy.set_phase(phase)
        cumulative_rewards = []
        average_return_list = []
        mean_aoi_list = []
        mean_energy_consumption_list = []
        collected_data_amount_list = []
        update_human_coverage_list = []


        for ep_i in range(k):
            print(f'run episodes {ep_i} / {k}')
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
                state, reward, done, env_info = self.envs.step(np.array(action.cpu()))  # 东西存在info里
                done = done.any()
                # print('step {}, reward={}'.format(s, reward))

                s += 1
                self.logger.log(interaction=None)
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)
                if done:
                    ep_r = np.transpose(np.array(rewards), (1, 0, 2))  # (threads, 120, 3)
                    ep_r = ep_r.sum(axis=-1).sum(axis=-1)  # (threads, )
                    self.logger.log(mean_episode_reward=ep_r.mean(), episode_len=self.input_args.max_episode_step, episode=None)
                    self.logger.log(max_episode_reward=ep_r.max(), episode_len=self.input_args.max_episode_step, episode=None)

                    self.logger.log(QoI=sum(d['QoI'] for d in env_info) / len(env_info),
                                    episodic_aoi=sum(d['episodic_aoi'] for d in env_info) / len(env_info),
                                    aoi_satis_ratio=sum(d['aoi_satis_ratio'] for d in env_info) / len(env_info),
                                    data_satis_ratio=sum(d['data_satis_ratio'] for d in env_info) / len(env_info),
                                    energy_consuming=sum(d['energy_consuming'] for d in env_info) / len(env_info),
                                    good_reward=sum(d['good_reward'] for d in env_info) / len(env_info),
                                    aoi_penalty_reward=sum(d['aoi_penalty_reward'] for d in env_info) / len(env_info),
                                    knn_reward=sum(d['knn_reward'] for d in env_info) / len(env_info),
                                    )

                    if ep_r.max() > self.best_episode_reward:
                        max_id = ep_r.argmax()
                        self.best_episode_reward = ep_r.max()
                        best_train_trajs = self.envs.get_saved_trajs()
                        poi_aoi_history = self.envs.get_poi_aoi_history()
                        serves = self.envs.get_serves()
                        write_output(env_info[max_id], self.input_args.output_dir)
                        self.dummy_env.save_trajs_2(
                            best_train_trajs[max_id], poi_aoi_history[max_id], serves[max_id], phase='train', is_newbest=True)

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





def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
