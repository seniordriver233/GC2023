from environments.multiagentenv import MultiAgentEnv
import numpy as np
from collections import namedtuple
from envs.config import *
from envs.system_model import SystemModel





def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


class MecBCEnv(MultiAgentEnv):
    def __init__(self, kwargs):
        args = kwargs

        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        self.n_agents = N_AGENT
        self.action_space = action_size
        self.observation_space = state_size

        # 初始化state
        self.Bandwidth = np.zeros((self.n_agents, K_CHANNEL))
        self.S_channel = np.zeros(self.n_agents)
        self.S_energy = np.zeros(self.n_agents)
        self.S_power = np.zeros(self.n_agents)

        self.MU_1 = np.random.normal(MU_1, 0.1, size=self.n_agents)
        self.MU_2 = np.random.normal(MU_2, 0.1, size=self.n_agents)
        self.MU_3 = np.random.normal(MU_3, 0.1, size=self.n_agents)


        self.S_gain = np.zeros((self.n_agents, K_CHANNEL))
        for n in range(self.n_agents):
            self.S_gain[n] = np.random.normal(S_GAIN, 1, size=K_CHANNEL)
            self.Bandwidth[n] = np.random.normal(S_BandWidth, 1, size=K_CHANNEL)
        self.S_size = np.zeros(self.n_agents)
        self.S_cycle = np.zeros(self.n_agents)
        self.S_resolu = np.zeros(self.n_agents)
        self.S_ddl = np.zeros(self.n_agents)
        self.S_com = np.zeros(self.n_agents)
        self.S_servercom = np.zeros((self.n_agents, K_CHANNEL))
        self.S_epsilon = np.zeros(self.n_agents)
        self.new_energy = np.zeros(self.n_agents)
        # 设定连续动作空间范围
        # self.action = [compression, channel, local_computation, transmission_power]
        self.action_lower_bound = [0, 0, MIN_COM, MIN_POWER]
        self.action_higher_bound = [1, K_CHANNEL, MAX_COM, MAX_POWER]

        # 初始epoch数
        self.epoch = 0

    # 重置

    def reset(self):
        """ 重置状态 """
        self.epoch = 0

        # 随机化开始的state
        for n in range(self.n_agents):
            for k in range(K_CHANNEL):
                # self.S_gain[n][k] = np.random.normal(self.S_gain[n][k], self.S_gain[n][k]/10)
                self.S_gain[n][k] = np.random.normal(10, 1)
                self.S_servercom[n][k] = np.random.normal(Server_Com, Server_Com/10)
            self.S_size[n] = np.random.normal(S_SIZE, S_SIZE/10)
            self.S_cycle[n] = 18000 * self.S_size[n] / (1 * 10 ** 6)
            self.S_ddl[n] = np.random.normal(S_DDL, S_DDL/10)
            self.S_epsilon[n] = np.random.normal(S_EPSILON, S_EPSILON/10)
            self.S_energy[n] = np.random.normal(S_Energy, S_Energy/10)
            self.new_energy[n] = np.random.normal(Energy_supply, Energy_supply/10)

        return self.state

    def step(self, action):
        # 根据action改state
        for n in range(self.n_agents):
            self.S_resolu[n] = action[n][0]
            self.S_channel[n] = action[n][1]
            self.S_com[n] = action[n][2]
            self.S_power[n] = action[n][3]

            # manual reset
            for k in range(K_CHANNEL):
                # self.S_gain[n][k] = np.random.normal(self.S_gain[n][k], self.S_gain[n][k]/10)
                self.S_gain[n][k] = np.random.normal(10, 1)
            self.S_size[n] = np.random.normal(S_SIZE, S_SIZE / 10)
            self.S_cycle[n] = 18000 * self.S_size[n] / (1 * 10 ** 6)
            self.S_ddl[n] = np.random.normal(S_DDL, S_DDL / 10)
            self.new_energy[n] = np.random.normal(Energy_supply, Energy_supply / 10)

        # 求reward
        system_model = SystemModel(self.n_agents, self.S_channel, self.S_power, self.S_gain, self.S_size, self.S_cycle,
                                   self.S_resolu, self.S_ddl, self.S_energy, self.S_com, self.S_epsilon,self.Bandwidth,self.S_servercom,
                                   self.MU_1,self.MU_2,self.MU_3)

        self.S_energy = self.S_energy + self.new_energy - system_model.Energy

        # TODO:【训练】这里可以适当改变每个episode的步数
        self.epoch += 1
        done = False
        if self.epoch > 50:
            self.reset()
            done = True

        return self.state, system_model.Reward, done, None

    def get_avail_actions(self):
        """return available actions for all agents
        """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))

        return np.expand_dims(np.array(avail_actions), axis=0)

    def get_avail_agent_actions(self, agent_id):
        """ return the available actions for agent_id
        """
        action = np.array([1] * self.action_space)

        return action

    def get_num_of_agents(self):
        """return the number of agents
        """
        return self.n_agents

    def get_obs(self):
        """return the obs for each agent in the power system
           the default obs: voltage, active power of generators, bus state, load active power, load reactive power
           each agent can only observe the state within the zone where it belongs
        """
        return self.state

    def get_obs_size(self):
        """return the observation size
        """
        return self.observation_space

    def get_total_actions(self):
        """return the total number of actions an agent could ever take
        """
        return self.action_space

    @property
    def state(self):
        """ 返回当前全体observation """
        # state_ = [[self.Bandwidth[n], self.S_gain[n], np.array(self.S_size[n]), np.array(self.S_ddl[n]), np.array(self.S_energy[n]),
        #            self.S_servercom[n]] for n in range(self.n_agents)]
        # state_ = [[self.Bandwidth[n], self.S_gain[n], self.S_size[n], self.S_ddl[n], self.S_energy[n], self.S_servercom[n]]
        #           for n in range(self.n_agents)]
        state_ = [[self.Bandwidth[n][0], self.Bandwidth[n][1], self.Bandwidth[n][2], self.Bandwidth[n][3],
                   self.S_gain[n][0], self.S_gain[n][1], self.S_gain[n][2], self.S_gain[n][3],
                   self.S_size[n], self.S_ddl[n], self.S_energy[n],
                   self.S_servercom[n][0], self.S_servercom[n][1], self.S_servercom[n][2], self.S_servercom[n][3]]
                  for n in range(self.n_agents)]
        # state_ = [
        #     [self.Bandwidth[n], self.S_gain[n], np.array([self.S_size[n],self.S_size[n],self.S_size[n],self.S_size[n]]),
        #      np.array([self.S_ddl[n],self.S_ddl[n],self.S_ddl[n],self.S_ddl[n]]),
        #      np.array([self.S_energy[n],self.S_energy[n],self.S_energy[n],self.S_energy[n]]), self.S_servercom[n]]
        #     for n in range(self.n_agents)]
        # print(type(state_))
        # print("----------state---------")
        # print(state_)
        state_ = np.array(state_)
        return state_
