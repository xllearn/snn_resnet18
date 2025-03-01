### train.py
# 训练 SNN 模型
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from spikingjelly.activation_based import functional
from data import get_dataloaders
from model import build_model
from config import device, lr, epochs


def train():
    train_loader, test_loader = get_dataloaders()
    model = build_model().to(device)

    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).mean(0)  # 添加平均步骤
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        functional.reset_net(model)  # 重置 SNN 状态

    torch.save(model.state_dict(), "snn_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
