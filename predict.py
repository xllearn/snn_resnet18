### predict.py
# 载入模型并进行预测
import torch
from data import get_dataloaders
from model import build_model
from config import device


def predict():
    _, test_loader = get_dataloaders()
    model = build_model().to(device)
    model.load_state_dict(torch.load("snn_model.pth"))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).mean(0)  # 添加平均步骤
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")
