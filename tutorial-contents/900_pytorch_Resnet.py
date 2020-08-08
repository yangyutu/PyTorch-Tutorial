# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:09:42 2020

@author: yangy
"""

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torchsummary import summary
# device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
EPOCHS = 20
batch_size = 64
learning_rate = 0.001
classes = 10


# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
])

# cifar10 32*32*3
train_dataset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False, transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False) # only the first one take input stride to expand /shrink the dimensionality
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride = 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample: # we might need to perform linear map to initial residual to match the dimension between x and residual
            residual = self.downsample(input)         
        x = x + residual
        output = self.relu(x)
        return output


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block = block, out_channels = 16, n_blocks = layers[0])
        self.layer2 = self.make_layer(block = block, out_channels = 32, n_blocks = layers[1], stride = 2)
        self.layer3 = self.make_layer(block = block, out_channels = 64, n_blocks = layers[2], stride = 2)
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block = None, out_channels = 16, n_blocks = 2, stride=1):
        downsample = None
        # if stride is not 1 or in_channels and out_channels are mismatched, we need to use conv layers as linear maps to match dimensionality 
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, n_blocks):
            layers.append(block(out_channels, out_channels, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
#summary(model, (3, 64, 64))
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# for updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# train model
pre_epoch_total_step = len(trainloader)
current_lr = learning_rate
for epoch in range(EPOCHS):
    for i, (x, y) in enumerate(trainloader):
        x = x.to(device)
        y = y.to(device)

        # forward
        prediction = model(x)
        loss = criterion(prediction, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            template = r"Epoch:{}/{}, step:{}/{}, Loss:{:.6f}"
            print(template.format(epoch+1, EPOCHS, i+1, pre_epoch_total_step, loss.item()))

    # decay learning rate
    if (epoch+1) % 20 == 0:
        current_lr = current_lr/2
        update_lr(optimizer, current_lr)


# test model
model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for x, y in testloader:
        x = x.to(device)
        y = y.to(device)
        prediction = model(x)
        _, predic = torch.max(prediction.data, dim=1)
        total += y.size(0)
        correct += (predic == y).sum().item()

    print("Accuracy:{}%".format(100 * correct / total))

# save model
torch.save(model.state_dict(), "cifar10_resnet.ckpt")