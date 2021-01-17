"""
@author: Li Jiahao
"""

import torch
from torch import nn
import torch.nn.functional as F
#from torchsummary import summary

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(2,1)
        self.fc1 = nn.Linear(33856, 128)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, img):
        img = F.relu(self.conv1(img))
        img = F.relu(self.conv2(img))
        img = self.pool1(img)
        feature = F.relu(self.fc1(img.view(img.shape[0], -1)))
        feature = self.dropout1(feature)
        output = self.fc2(feature)
        return output

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(2,1)
        self.fc1 = nn.Linear(23*23*128, 256)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, img):
        img = F.relu(self.conv1(img))
        img = F.relu(self.conv2(img))
        img = self.pool1(img)
        feature = F.relu(self.fc1(img.view(img.shape[0], -1)))
        feature = self.dropout1(feature)
        output = self.fc2(feature)
        return output

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 512)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, img):
        img = self.dropout1(F.relu(self.fc1(img.view(img.shape[0], -1))))
        img = self.dropout2(F.relu(self.fc2(img)))
        output = self.fc3(img)
        return output

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net1 = Net1().to(device)
    print("Net1:")
    summary(net1, (1, 28, 28))
    net2 = Net2().to(device)
    print("Net2:")
    summary(net2, (1, 28, 28))
    net3 = Net3().to(device)
    print("Net3:")
    summary(net3, (1, 28, 28))

if __name__ == '__main__':
    main()