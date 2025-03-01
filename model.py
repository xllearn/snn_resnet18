### model.py
# 构建 SNN 模型
import torch.nn as nn
from spikingjelly.activation_based import model as sj_model
from config import spiking_neuron, surrogate_function, detach_reset


from spikingjelly.activation_based.model import spiking_resnet

def build_model():
    model = spiking_resnet.spiking_resnet18(
        pretrained=False,
        spiking_neuron=spiking_neuron,
        surrogate_function=surrogate_function,
        detach_reset=detach_reset
    )
    return model


