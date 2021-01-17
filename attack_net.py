"""
@author: Li Jiahao
"""

import torch
from torch import nn
import torch.nn.functional as F
#from torchsummary import summary

class UPSET(nn.Module):
    def __init__(self):
        super(UPSET, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 1*28*28)

    def forward(self, img):
        img = F.relu(self.fc1(img.view(img.shape[0], -1)))
        img = F.leaky_relu(self.fc2(img))
        img = F.leaky_relu(self.fc3(img))
        img = F.leaky_relu(self.fc4(img))
        img = F.leaky_relu(self.fc5(img))
        output = F.tanh(self.fc6(img))
        return output.view(-1, 1, 28, 28)
        
class ANGRI(nn.Module):
    def __init__(self):
        super(ANGRI, self).__init__()
        self.fc11 = nn.Linear(10, 128)
        self.fc12 = nn.Linear(128, 256)
        self.fc13 = nn.Linear(256, 512)
        self.fc21 = nn.Linear(28*28, 128)
        self.fc22 = nn.Linear(128, 256)
        self.fc23 = nn.Linear(256, 512)

        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 1*28*28)

    def forward(self, t, img):
        t = F.leaky_relu(self.fc11(t))
        t = F.leaky_relu(self.fc12(t))
        t = F.leaky_relu(self.fc13(t))

        img = F.relu(self.fc21(img.view(img.shape[0], -1)))
        img = F.leaky_relu(self.fc22(img))
        img = F.leaky_relu(self.fc23(img))

        sum = torch.cat([t,img], dim=1)
        sum = F.leaky_relu(self.fc4(sum))
        sum = F.leaky_relu(self.fc5(sum))
        output = F.tanh(self.fc6(sum))
        return output.view(-1, 1, 28, 28)
        
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(7*7*128, 28*28)

    def forward(self, img):
        img = F.relu(self.bn1(self.conv1(img)))
        img = self.pool1(img)
        img = F.relu(self.bn2(self.conv2(img)))
        img = self.pool2(img)
        img = F.relu(self.bn3(self.conv3(img)))
        img = self.fc1(img.view(img.shape[0], -1))
        output = F.tanh(img)
        return output.view(-1, 1, 28, 28)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net1 = UPSET().to(device)
    summary(net1, (1, 28, 28))
    net2 = MyNet().to(device)
    summary(net2, (1, 28, 28))

if __name__ == '__main__':
    main()