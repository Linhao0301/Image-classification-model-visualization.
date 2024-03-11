import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet, BasicBlock
import numpy as np

class CustomResNet(ResNet):
    def __init__(self):
        super(CustomResNet, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10)  # 类似于ResNet34
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# 其余的函数定义（get_feature_maps_hook, visualize_and_save_feature_maps, visualize_kernels, visualize_gradients, plot_metrics）保持不变

losses = []
accuracies = []
feature_maps = []  # 用于存储特征图
2
def get_feature_maps_hook(module, input, output):
    global feature_maps
    feature_maps.append(output)

def visualize_and_save_feature_maps(epoch):
    global feature_maps
    # 选择最后一批数据的特征图进行可视化
    fmap = feature_maps[-1].detach().cpu().numpy()
    # 选择可视化的特征图数量
    fmap_count = min(5, fmap.shape[1])  # 假设我们只可视化前5个特征图

    fig, axs = plt.subplots(1, fmap_count, figsize=(15, 5))
    for i in range(fmap_count):
        axs[i].imshow(fmap[0, i], cmap='viridis')
        axs[i].set_title(f'Feature Map {i+1}')
        axs[i].axis('off')

    plt.savefig(f'feature_maps_epoch_{epoch}.png')
    plt.close(fig)  # 防止显示图像
    feature_maps = []  # 清空特征图列表以节省内存


def visualize_kernels(layer, epoch, layer_name='conv1'):
    # 获取卷积核权重
    kernels = layer.weight.detach().cpu().numpy()
    num_kernels = kernels.shape[0]

    # 设置可视化的卷积核数量
    kernels_to_visualize = min(5, num_kernels)

    # 创建画布
    fig, axs = plt.subplots(1, kernels_to_visualize, figsize=(15, 5))
    for i in range(kernels_to_visualize):
        # 选择第一个输入通道的卷积核进行可视化
        kernel = kernels[i, 0]
        axs[i].imshow(kernel, cmap='gray')
        axs[i].set_title(f'Kernel {i + 1}')
        axs[i].axis('off')

    plt.savefig(f'{layer_name}_kernels_epoch_{epoch}.png')
    plt.close(fig)


def visualize_gradients(layer, epoch, layer_name='conv1'):
    # 获取梯度
    gradients = layer.weight.grad.detach().cpu().numpy()
    num_gradients = gradients.shape[0]

    # 设置可视化的梯度数量
    gradients_to_visualize = min(5, num_gradients)

    # 创建画布
    fig, axs = plt.subplots(1, gradients_to_visualize, figsize=(15, 5))
    for i in range(gradients_to_visualize):
        # 选择第一个输入通道的梯度进行可视化
        gradient = gradients[i, 0]
        axs[i].imshow(gradient, cmap='viridis')
        axs[i].set_title(f'Gradient {i + 1}')
        axs[i].axis('off')

    plt.savefig(f'{layer_name}_gradients_epoch_{epoch}.png')
    plt.close(fig)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f'End of Epoch {epoch}: Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')

    losses.append(avg_loss)
    accuracies.append(accuracy)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f'checkpoint_epoch_{epoch}.pth')

    # 在每个epoch结束后可视化并保存特征图
    visualize_and_save_feature_maps(epoch)

    # 在每个epoch结束后可视化卷积核和梯度
    visualize_kernels(model.conv1, epoch)
    visualize_gradients(model.conv1, epoch)

def plot_metrics():
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    device = torch.device("mps")  # 使用MPS，如果不支持则自动回退到CPU

    model = CustomResNet().to(device)
    # 在模型的第一个卷积层注册钩子
    hook = model.conv1.register_forward_hook(get_feature_maps_hook)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 6):
        train(model, device, train_loader, optimizer, epoch)

    hook.remove()  # 训练结束后，移除钩子

    plot_metrics()

if __name__ == '__main__':
    main()
