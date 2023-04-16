import numpy as np
from .config import *
from .system_model import SystemModel


class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self):

        self.agent_num = N_AGENT  # 设置智能体(小飞机)的个数，这里设置为两个 # set the number of agents(aircrafts), here set to two
        self.obs_dim = state_size  # 设置智能体的观测维度 # set the observation dimension of agents
        self.action_dim = action_size  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional

        self.n_agents = N_AGENT
        self.action_space = action_size
        self.observation_space = state_size

        # 初始化state
        self.Bandwidth = np.zeros((self.n_agents, K_CHANNEL))
        self.S_channel = np.zeros(self.n_agents)
        self.S_energy = np.zeros(self.n_agents)
        self.S_power = np.zeros(self.n_agents)

        self.MU_1 = np.random.normal(MU_1, MU_1 / 10, size=self.n_agents)
        self.MU_2 = np.random.normal(MU_2, MU_2 / 10, size=self.n_agents)
        self.MU_3 = np.random.normal(MU_3, MU_3 / 10, size=self.n_agents)

        self.S_gain = np.zeros((self.n_agents, K_CHANNEL))
        for n in range(self.n_agents):
            self.S_gain[n] = np.random.normal(S_GAIN, S_GAIN / 10, size=K_CHANNEL)
            self.Bandwidth[n] = np.random.normal(S_BandWidth, S_BandWidth / 10, size=K_CHANNEL)
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
        self.action_lower_bound = [0.8, 0, MIN_COM, MAX_POWER / 100]
        self.action_higher_bound = [0.9, K_CHANNEL, MAX_COM, MAX_POWER]

        # 初始epoch数
        self.epoch = 0

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        self.epoch = 0

        # 随机化开始的state
        for n in range(self.n_agents):
            for k in range(K_CHANNEL):
                self.S_gain[n][k] = np.random.normal(S_GAIN, S_GAIN / 10)
                self.S_servercom[n][k] = np.random.normal(Server_Com, Server_Com / 10)
            self.S_size[n] = np.random.normal(S_SIZE, S_SIZE / 10)
            self.S_cycle[n] = 18000 * self.S_size[n]
            self.S_ddl[n] = np.random.normal(S_DDL, S_DDL / 10)
            self.S_epsilon[n] = S_EPSILON
            self.S_energy[n] = np.random.normal(S_Energy, S_Energy / 10)
            self.new_energy[n] = np.random.normal(Energy_supply, Energy_supply / 10)

        sub_agent_obs = []
        for n in range(self.agent_num):
            sub_agent_obs.append(
                np.array([self.Bandwidth[n][0], self.Bandwidth[n][1], self.Bandwidth[n][2], self.Bandwidth[n][3],
                          self.S_gain[n][0], self.S_gain[n][1], self.S_gain[n][2], self.S_gain[n][3],
                          self.S_size[n], self.S_ddl[n], self.S_energy[n],
                          self.S_servercom[n][0], self.S_servercom[n][1], self.S_servercom[n][2],
                          self.S_servercom[n][3]]))
        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        for n in range(self.n_agents):
            for k in range(K_CHANNEL):
                self.S_gain[n][k] = np.random.normal(S_GAIN, S_GAIN / 10)
            self.S_size[n] = np.random.normal(S_SIZE, S_SIZE / 10)
            self.S_cycle[n] = 18000 * self.S_size[n]
            self.S_ddl[n] = np.random.normal(S_DDL, S_DDL / 10)
            self.S_epsilon[n] = S_EPSILON
            self.new_energy[n] = np.random.normal(Energy_supply, Energy_supply / 10)

        lower_bound = self.action_lower_bound
        higher_bound = self.action_higher_bound
        for n in range(self.n_agents):
            self.S_resolu[n] = (higher_bound[0] - lower_bound[0]) * actions[n][0] + lower_bound[0]
            self.S_channel[n] = ((higher_bound[1] - lower_bound[1]) * actions[n][1] + lower_bound[1]).round()
            self.S_com[n] = (higher_bound[2] - lower_bound[2]) * actions[n][2] + lower_bound[2]
            self.S_power[n] = (higher_bound[3] - lower_bound[3]) * actions[n][3] + lower_bound[3]
        # print(self.S_channel)

        system_model = SystemModel(self.n_agents, self.S_channel, self.S_power, self.S_gain, self.S_size, self.S_cycle,
                                   self.S_resolu, self.S_ddl, self.S_energy, self.S_com, self.S_epsilon, self.Bandwidth,
                                   self.S_servercom,
                                   self.MU_1, self.MU_2, self.MU_3)
        self.S_energy = self.S_energy + self.new_energy - system_model.Energy
        # TODO:【训练】这里可以适当改变每个episode的步数
        self.epoch += 1
        done = False
        if self.epoch > 60:
            self.reset()
            done = True

        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for n in range(self.agent_num):
            sub_agent_obs.append(
                np.array([self.Bandwidth[n][0], self.Bandwidth[n][1], self.Bandwidth[n][2], self.Bandwidth[n][3],
                          self.S_gain[n][0], self.S_gain[n][1], self.S_gain[n][2], self.S_gain[n][3],
                          self.S_size[n], self.S_ddl[n], self.S_energy[n],
                          self.S_servercom[n][0], self.S_servercom[n][1], self.S_servercom[n][2],
                          self.S_servercom[n][3]]))
            sub_agent_done.append(done)
            sub_agent_reward.append(np.array([np.mean(system_model.Reward)]))
            sub_agent_info.append(np.array([np.mean(system_model.Energy)]))
        # print(np.mean(sub_agent_reward))
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
