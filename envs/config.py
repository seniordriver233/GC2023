# TODO：【训练】这里可以再调整下数值
state_size = 15  # 状态空间纬度
action_size = 4  # 动作空间纬度
N_AGENT = 8   # Unit: 1

# 有物理意义的值

S_DDL = 1  # Unit: s
S_EPSILON = 0.86  # 不重要。因为没有公式用到epsilon
# W_BANDWIDTH = 20 * 10 ** 6  # Unit: MHz -> Hz
S_POWER = 10 ** (float((30 - 30) / 10))  # Unit: dBm -> W where 1 dBm = 10lg(W) + 30
S_GAIN = 10  # Unit: ???
S_SIZE = 1 * 10 ** 6  # 传输数据量大小 Unit: MB -> B
  # 计算要求的cycle数量，此处不重要，在ENV_GC中有根据S_SIZE的比例计算 Unit: 1
# S_RESOLU = 0.6  # 压缩率 Unit: 1
# S_RES = 1.5
# S_COM = 5 * 10 ** 9  # Unit: GHz -> Hz
S_Energy = 0 * 10 ** (-3)  # Unit: mW -> W
S_BandWidth = 20 * 10 ** 6  # Unit: MHz -> Hz
Energy_supply = 5 * 10 ** (-3)  # Unit: mW -> W
Server_Com = 10 * 10 ** 9  # Unit: GHz -> Hz
Client_Com = 1 * 10 ** 9  # Unit: GHz -> Hz

# 信道数量
K_CHANNEL = 4  # Unit: 1

# 本地算力范围
MIN_COM = 0.1 * 10 ** 9  # Unit: GHz -> Hz
MAX_COM = 1 * 10 ** 9  # Unit: GHz -> Hz

# 传输功率
MAX_POWER = 10 ** (float((24 - 30) / 10))  # Unit: dBm -> W where 1 dBm = 10lg(W) + 30
MIN_POWER = 0

MAX_GAIN = 10
MIN_GAIN = 5

# 传输数据量范围
MIN_SIZE = 0.1 * 10 ** 6  # Unit: MB -> B
MAX_SIZE = 5 * 10 ** 6  # Unit: MB -> B

# CPU cycle范围
MIN_CYCLE = 1 * 10 ** 3  # Unit: 1
MAX_CYCLE = 1 * 10 ** 6  # Unit: 1

# 勿动。文献值。
V_L = 0.125
V_E = 0.13
THETA_L = 1 / 1600
THETA_E = 1 / 1700

# 系数
LAMBDA_E = 0.4
LAMBDA_PHI = 0.6

# 系数，用于平衡accuracy, energy和time的相对大小，调整reward的值
MU_1 = 100
MU_2 = 1000
MU_3 = 10

# 系数
K_ENERGY_LOCAL = 5 * 10 ** (-26)  # k = 5 * 10 ^(-27) * M * G^2
# K_ENERGY_MEC = 0.7 * 10 ** (-26)

NOISE_VARIANCE = 10 ** (float((-100 - 30) / 10))  # Unit: dBm -> W

# 系数
OMEGA = 0.9 * 10 ** (-11)  # w = 0.9*10*(-11) * G

# CAPABILITY_E = 5

# 仅占位，无意义
MIN_EPSILON = 0.56
MAX_EPSILON = 0.93

# MIN_RES = 0.4  # Unit: ???
# MAX_RES = 2.3  # Unit: ???

# KSI = 0.5  # Unit: ???
# LAMBDA = 0.5
# ALPHA = 0.5

# MIN_DDL = 0.4 #
# MAX_DDL = 2 # 没用到
