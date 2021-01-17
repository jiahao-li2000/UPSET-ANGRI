"""
@author: Li Jiahao
"""

import time
import torch
import dataloader as dl
from torch import nn
from matplotlib import pyplot as plt
import Nets

def train(net, save_path, lr=0.001, num_epochs=10, batch_size=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net
    batch_size = batch_size
    train_iter, test_iter = dl.load_MNIST(batch_size=batch_size, cut=10)
    learning_rate, num_epochs = lr, num_epochs

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    loss_list, train_acc_list, test_acc_list = [], [], []

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.long())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.4f, test acc %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        loss_list.append(train_l_sum / batch_count)
        train_acc_list.append(train_acc_sum / n)
        test_acc_list.append(test_acc)
    torch.save(net.state_dict(), save_path)
    print("done!")
    #draw_curves(num_epochs, loss_list, train_acc_list, test_acc_list)

def evaluate_accuracy(data_iter, net, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train() # 改回训练模式
            n += y.shape[0]
    return acc_sum / n

def draw_curves(length, loss_list, train_acc_list, test_acc_list):
    plt.figure(1, figsize=(5, 5))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.semilogy(range(1, length + 1), loss_list)
    
    plt.figure(2, figsize=(5, 5))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(range(1, length + 1), train_acc_list)
    plt.plot(range(1, length + 1), test_acc_list)
    plt.legend(['train','test'])

def main():
    net = Nets.Net3()
    PATH = './model/Net3.pth'
    train(net, PATH, lr=0.001, num_epochs=30, batch_size=512)

if __name__ == '__main__':
    main()
