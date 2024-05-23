import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


torch.manual_seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=256, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,transform=transform)
testloader = DataLoader(testset,batch_size=256,shuffle=False)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1,6,5,1,2),
                                 nn.Sigmoid(),
                                 nn.AvgPool2d(2,2),
                                 nn.Sigmoid(),
                                 nn.Conv2d(6,16,5,1,0),
                                 nn.Sigmoid(),
                                 nn.AvgPool2d(2,2),
                                 nn.Flatten(),
                                 nn.LazyLinear(120),
                                 nn.Sigmoid(),
                                 nn.LazyLinear(84),
                                 nn.Sigmoid(),
                                 nn.LazyLinear(10),
                                 nn.Softmax(dim=1))
    

    def forward(self, x):
        return self.net(x)
    
net = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001) # 相较于此前的SGD，训练效果好很多，之前甚至怀疑梯度消失了

def train_and_evaluate(net,trainloader,testloader, criterion, optimizer, epochs):
    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (batch_X, batch_y) in enumerate(trainloader, 0):
            optimizer.zero_grad()
            outputs = net(batch_X)
            loss = criterion(outputs,batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += len(batch_y)
            correct += (predicted == batch_y).sum().item()
    
        train_losses.append(running_loss / (i + 1))
        train_accs.append(correct / total)

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in testloader:
                outputs = net(batch_X)
                _ , predicted = torch.max(outputs.data, 1)
                total += len(batch_y)
                correct += (predicted == batch_y).sum().item()
        
        test_accs.append(correct / total)
        print(f'Epoch {epoch + 1}, Loss: {running_loss / (i + 1):.4f}, '
              f'Train Accuracy: {100 * train_accs[-1]:.2f}%, '
              f'Test Accuracy: {100 * test_accs[-1]:.2f}%')

    return train_losses, train_accs, test_accs

# 训练和评估模型
epochs = 10
train_losses, train_accuracies, test_accuracies = train_and_evaluate(net, trainloader, testloader, criterion, optimizer, epochs)

# 绘制训练过程中的损失和精度变化曲线
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(epochs_range, train_losses, 'o-', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs_range, train_accuracies, 'o-', label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs_range, test_accuracies, 'o-', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 在测试集上随机抽取10个样本并进行分类
dataiter = iter(testloader)
images, labels = next(dataiter)
outputs = net(images)
_, predicted = torch.max(outputs, 1)

# 显示图像和预测结果
fig, axes = plt.subplots(1, 10, figsize=(12, 2))
for i in range(10):
    ax = axes[i]
    ax.imshow(images[i].numpy().squeeze(), cmap='gray')
    ax.set_title(f'Pred: {predicted[i].item()}')
    ax.axis('off')
plt.show()

