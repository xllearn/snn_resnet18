### data.py
# 处理 FashionMNIST 数据集
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import data_root, batch_size, num_workers

import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # 将单通道复制 3 次变成 3 通道
])


def get_dataloaders():
    train_dataset = torchvision.datasets.FashionMNIST(root=data_root, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root=data_root, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader