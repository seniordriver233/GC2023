""" Formulas to calculate """


from .config import *
import numpy as np

# TODO：【环境】【验证】之后再检查下，尤其是多维公式的计算（考虑ENV来的输入）


class SystemModel:
    def __init__(self, n_agents, S_channel, S_power, S_gain, S_size, S_cycle, S_resolu, S_ddl, S_energy, S_com,
                 S_epsilon, S_Bandwidth, S_servercom, MU_1, MU_2, MU_3):
        # Parameters
        self.S_channel = S_channel
        self.S_power = S_power
        self.S_gain = S_gain
        self.S_size = S_size
        self.S_cycle = S_cycle
        self.S_resolu = S_resolu
        self.S_ddl = S_ddl
        self.S_com = S_com
        self.S_epsilon = S_epsilon
        self.S_servercom = S_servercom
        self.S_energy = S_energy
        self.S_decision = (self.S_channel > 0).astype(int)
        self.Phi_local = V_L * np.log(1 + self.S_resolu / THETA_L)  # expectation=0.772 when resolu=0.3
        self.Phi_off = V_E * np.log(1 + self.S_resolu / THETA_E)  # expectation=0.811 when resolu=0.3
        self.MU_1 = MU_1
        self.MU_2 = MU_2
        self.MU_3 = MU_3

        # print("resolu")
        # print(self.S_resolu)

        # TODO :【重要】【验证】这里计算公式中检查下是否正确
        DataRate = np.zeros(n_agents)
        Server_fre = np.zeros(n_agents)
        for n in range(n_agents):
            if self.S_channel[n] == 0:
                DataRate[n] = 0.1
                Server_fre[n] = 0.1  # 很小值即可
                continue
            # 只有占据同一个channel的会互相干扰
            # SNR = np.sum([0 if (n_ == n or int(self.S_channel[n_]) != int(self.S_channel[n])) else self.S_decision[n_] *
            #                                                                                        self.S_power[n_] *
            #                                                                                        self.S_gain[n_][int(
            #                                                                                            S_channel[
            #                                                                                                n_]) - 1] for
            #               n_ in range(n_agents)])
            SNR = np.sum([0 if (int(self.S_channel[n_]) != int(self.S_channel[n])) else self.S_decision[n_] *
                                                                                        self.S_power[n_] *
                                                                                        self.S_gain[n_][
                                                                                            int(S_channel[n_]) - 1] for
                          n_ in range(n_agents)])
            DataRate[n] = S_Bandwidth[n][self.S_channel[n].astype(int) - 1] * np.log(
                1 + self.S_power[n] * self.S_gain[n][int(self.S_channel[n]) - 1] /
                (NOISE_VARIANCE + SNR)) / np.log(2)

            # print("S_Bandwidth")
            # print(S_Bandwidth[n][self.S_channel[n].astype(int)-1])
            # print("datarate")
            # print(DataRate[n])

            # 分享server算力
            CNR = np.sum([0 if (int(self.S_channel[n_]) != int(self.S_channel[n])) else self.S_decision[n_] *
                                                                                        self.S_cycle[n_] for
                          n_ in range(n_agents)])
            Server_fre[n] = self.S_cycle[n] / (CNR + 1e-8) * self.S_servercom[n][self.S_channel[n].astype(int) - 1]


        self.Time_proc = self.S_resolu * self.S_cycle / Server_fre
        self.Time_local = self.S_resolu * self.S_cycle / self.S_com  # expectation=1.08*10**(-11) where S_com=5*10**8
        self.Time_off = self.S_resolu * self.S_size / DataRate

        self.Energy_local = K_ENERGY_LOCAL * S_size * S_resolu * (S_com ** 2)  # expectation=1.5*10**(-3)
        # self.Energy_off = S_power * self.Time_off * 10 ** (-6)  # expectation=1*10**(-7)*Time_off
        self.Energy_off = S_power * self.Time_off

        self.total_com = np.sum(self.S_com)
        self.T_mean = np.mean(self.Time)

    @property
    def Accuracy(self):
        # print("accuracy")
        # print((1 - self.S_decision) * self.Phi_local + self.S_decision * self.Phi_off)
        # print("_______TESTTTT________", self.Phi_local, self.Phi_off)
        accuracy = (1 - self.S_decision) * self.Phi_local + self.S_decision * self.Phi_off
        accuracy = np.random.normal(accuracy, accuracy/10)
        return accuracy  # expectation=0.8 when resolu=0.3

    @property
    def Accuracy_penalty(self):
        return np.maximum((0.5 - self.Accuracy) * 5, 0)

    @property
    def Time(self):
        # print("time")
        # print((1 - self.S_decision) * self.Time_local + self.S_decision * (self.Time_off + self.Time_proc))
        return (1 - self.S_decision) * self.Time_local + self.S_decision * (self.Time_off + self.Time_proc)

    @property
    def Time_penalty(self):
        return np.maximum((self.Time - self.S_ddl), 0)

    @property
    def Energy(self):
        # print("energy")
        # print((1 - self.S_decision) * self.Energy_local + self.S_decision * self.Energy_off)
        return (1 - self.S_decision) * self.Energy_local + self.S_decision * self.Energy_off

    @property
    def Energy_penalty(self):
        return np.maximum((self.Energy - self.S_energy), 0)

    @property
    def Reward(self):
        # print("real accuracy")
        # print(self.Accuracy)
        # print("total accuracy")
        # print(self.Accuracy - self.Accuracy_penalty)
        # print("energy")
        # print(self.Energy)
        # print("total energy")
        # print(self.Energy + self.Energy_penalty)
        # print("total time")
        # print(self.Time + self.Time_penalty)
        return self.MU_1 * (self.Accuracy - self.Accuracy_penalty) - self.MU_2 * (
                self.Energy + self.Energy_penalty) - self.MU_3 * (self.Time + self.Time_penalty)
