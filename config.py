### config.py
# 配置超参数和路径
import torch

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集参数
data_root = "./data"
batch_size = 64
num_workers = 0

# 训练参数
lr = 0.001
epochs = 10

# 神经元类型选择
from spikingjelly.activation_based import neuron, surrogate
spiking_neuron = neuron.LIFNode  # 选择 LIF 神经元
surrogate_function = surrogate.ATan()
detach_reset = True
