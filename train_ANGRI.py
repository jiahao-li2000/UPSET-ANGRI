"""
@author: Li Jiahao
"""

import time
import torch
import dataloader as dl
import attack_net
import Nets

def train(net_list, lr, num_epochs, w, batch_size=512, s=2.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    R = attack_net.ANGRI().to(device)
    batch_size = batch_size
    train_iter, test_iter = dl.load_MNIST(batch_size=batch_size, cut=10)

    learning_rate, num_epochs = lr, num_epochs
    optimizer = torch.optim.Adam(R.parameters(), lr=learning_rate)
    print("training on", device)

    cross_entropy = torch.nn.CrossEntropyLoss()
    MSE_loss = torch.nn.MSELoss(reduction='sum')
    loss_list, train_acc_list = [], []
    turn = 0

    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, total, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, _ in train_iter:
            victim = net_list[turn]
            m = X.shape[0]
            total += m
            X = X.to(device)
            t = (torch.floor(10*torch.rand(m, 1))).long()
            t_one_hot = torch.zeros(m, 10).scatter_(1,t,1)
            t = t.to(device)
            t_one_hot = t_one_hot.to(device)
            X_adv = torch.clamp(X+s*R(t_one_hot, X), min=-1.0, max=1.0)

            y_adv_hat = victim(X_adv)
            loss1 = cross_entropy(y_adv_hat, t.reshape(-1))
            loss2 = MSE_loss(X, X_adv)
            loss = loss1 + w*loss2/m
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.cpu().item()
            train_acc_sum += (y_adv_hat.argmax(dim=1) == t.reshape(-1)).sum().cpu().item()
            batch_count += 1
            turn = (turn+1)%len(net_list)
        test_acc = evaluate_accuracy(test_iter, net_list[0], R, s)   
        print('epoch %d, loss %.4f, train acc %.4f, test acc %.4f, time %.1f sec'
              % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / total, test_acc, time.time() - start))
        if((epoch+1)%10 == 0):
            save_path='./model/ANGRI Net1 w=0.06/epoch_' + str(epoch+1) + '.pth'
            torch.save(R.state_dict(), save_path)
        loss_list.append(train_loss_sum / batch_count)
        train_acc_list.append(train_acc_sum / total)
    print('done!')

def evaluate_accuracy(data_iter, victim, R, s=2, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, total = 0.0, 0
    with torch.no_grad():
        for X, _ in data_iter:
            m = X.shape[0]
            X = X.to(device)
            R.eval() # 评估模式, 这会关闭dropout
            t = (torch.floor(10*torch.rand(m, 1))).long()
            t_one_hot = torch.zeros(m, 10).scatter_(1,t,1)
            t = t.to(device)
            t_one_hot = t_one_hot.to(device)
            
            X_adv = torch.clamp(X+s*R(t_one_hot, X), min=-1.0, max=1.0)
            y_adv_hat = victim(X_adv)
            acc_sum += (y_adv_hat.argmax(dim=1) == t.reshape(-1)).float().sum().cpu().item()
            R.train() # 改回训练模式
            total += m
    return acc_sum / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net1 = Nets.Net1().to(device)
    net1.load_state_dict(torch.load('./model/Net1.pth'))
    net3 = Nets.Net3().to(device)
    net3.load_state_dict(torch.load('./model/Net3.pth'))
    net = [net1]
    train(net, lr=0.001, num_epochs=200, batch_size=512, s=2.0, w=0.06)
    
if __name__ == '__main__':
    main()