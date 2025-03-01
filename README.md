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
SpikingResNet18(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64)
  (sn1): LIFNode()
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1)
  ... # ResNet-18 layers
  (fc): Linear(in_features=512, out_features=10, bias=False)
)


## Installation
```bash
pip install torch torchvision spikingjelly



