import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import spiking_resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化 SNN
snn = spiking_resnet.spiking_resnet18(
    pretrained=False,
    spiking_neuron=neuron.LIFNode,
    surrogate_function=surrogate.ATan(),
    detach_reset=True
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(snn.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    snn.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = snn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 立即重置网络状态（禁用梯度）
        with torch.no_grad():
            functional.reset_net(snn)

        # 统计和打印
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 100 == 99:
            avg_loss = running_loss / 100
            acc = 100. * correct / total
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
            running_loss = 0.0
            correct = 0
            total = 0

# 保存模型
torch.save(snn.state_dict(), 'snn_model.pth')