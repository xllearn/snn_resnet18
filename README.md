# Spiking ResNet-18 for FashionMNIST Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![SpikingJelly](https://img.shields.io/badge/SpikingJelly-0.0.0.14+-blue.svg)](https://github.com/fangwei123456/spikingjelly)

A PyTorch implementation of a Spiking Neural Network (SNN) based on ResNet-18 architecture for FashionMNIST classification using the SpikingJelly framework.

## Features
- **Spiking ResNet-18**: Pulse-based ResNet-18 implementation with LIF neurons
- **FashionMNIST Support**: Preprocessing pipeline for 28x28 grayscale images
- **Mixed-Precision Training**: Automatic gradient scaling and device optimization
- **State Management**: Neuron state resetting after each batch
- **Training Metrics**: Real-time loss/accuracy monitoring

## Requirements
- Python 3.8+
- PyTorch 1.12+
- SpikingJelly 0.0.0.14+
- torchvision

## Model Structure

The Spiking ResNet-18 architecture combines traditional ResNet-18 components with spiking neuron layers. Here's the detailed structure:

```python
SpikingResNet18(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64)
  (sn1): LIFNode(
    v_threshold=1.0, v_reset=0.0, 
    surrogate_function=ATan(), detach_reset=True
  )
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1)
  (layer1-4): Sequential(
    # Residual Blocks with Spiking Neurons
    (0-1): BasicBlock(
      (conv1): Conv2d(...)
      (bn1): BatchNorm2d(...)
      (sn1): LIFNode(...)
      (conv2): Conv2d(...)
      (bn2): BatchNorm2d(...)
      (sn2): LIFNode(...)
      (downsample): Sequential(...)  # When needed
    )
    ...
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=10, bias=False)
  (sn_out): LIFNode(...)
)


## Installation
```bash
pip install torch torchvision spikingjelly



