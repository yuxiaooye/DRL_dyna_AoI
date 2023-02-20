import abc
import logging
from algorithms.GCRL.envs.model.mdp import *


class Agent():
    def __init__(self):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.policy = None

    def print_info(self):
        logging.info('Agent is visible and has "holonomic" kinematic constraint')

    def set_policy(self, policy):
        '''
        这里把agent和policy解耦，但我感觉根本没什么意义
        '''
        self.policy = policy

    def act(self, state, current_timestep):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        action = self.policy.predict(state, current_timestep)
        return action


class Human():
    def __init__(self, id, config):
        self.id = id
        self.config = config
        self.px = None
        self.py = None
        self.theta = None
        self.aoi = None

    def set(self, px, py, theta, aoi):
        self.px = px  # position
        self.py = py
        self.theta = theta
        self.aoi = aoi


    def get_obs(self):
        return HumanState(self.px / self.config.env.nlon,
                          self.py / self.config.env.nlat,
                          self.theta / self.config.env.rotation_limit,
                          self.aoi / self.config.env.num_timestep)


class Robot():
    def __init__(self, id, config):
        self.id = id
        self.config = config
        self.px = None  # position
        self.py = None
        self.theta = None
        self.energy = None

    def set(self, px, py, theta, energy):
        self.px = px  # position
        self.py = py
        self.theta = theta
        self.energy = energy


    def get_obs(self):
        return RobotState(self.px / self.config.env.nlon,
                          self.py / self.config.env.nlat,
                          self.theta / self.config.env.rotation_limit,
                          self.energy / self.config.env.max_uav_energy)


