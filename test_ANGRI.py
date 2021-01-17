import torch
import numpy as np
import dataloader as dl
import attack_net
import Nets
from matplotlib import pyplot as plt
import random
import torch.nn.functional as F

def test1(s=2.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    goal = 1
    index = random.randint(0, 9999)
    
    R = attack_net.ANGRI()
    victim = Nets.Net1()
    victim.load_state_dict(torch.load('./model/Net1.pth'))
    victim = victim.to(device)
    PATH = './model/ANGRI Net1 w=0.12/epoch_200.pth'
    R.load_state_dict(torch.load(PATH, map_location=device))
    R = R.to(device)

    _, test_iter = dl.load_MNIST(batch_size=512, cut=10)
    TFR, MR, FS, C = evaluate_accuracy(test_iter, victim, R, s)
    print("Targeted fooling rate:", TFR)
    print("Misclassification rate:", MR)
    print("Fidelity score:", FS)
    print("Confidence:", C)

    test_images = dl.load_test_images()
    test_labels = dl.load_test_labels()
    X = test_images[index]
    y = test_labels[index]
    X = torch.from_numpy(X.reshape(1,1,28,28)).float().to(device)
    t = torch.zeros(1, 10)
    t[0, goal] = 1
    t = t.to(device)
    
    with torch.no_grad():    
        y_hat = victim(X)
        perturb = s*R(t, X)
        X_adv = torch.clamp(X + perturb, min=-1.0, max=1.0)
        y_adv_hat = victim(X_adv)
        print("image index:", index)
        print('The label of number is',int(y))
        print('The evaluated number is', (y_hat.argmax(dim=1)).cpu().item())
        print('After attack, the evaluated number is', (y_adv_hat.argmax(dim=1)).cpu().item())
        
        test = X.cpu().numpy().reshape(28, 28)
        test = test*127.5+127.5
        plt.figure(figsize=(3, 3))
        plt.imshow(test, cmap='gray')
        
        test_adv = X_adv.cpu().numpy().reshape(28, 28)
        test_adv = test_adv*127.5+127.5
        plt.figure(figsize=(3, 3))
        plt.imshow(test_adv, cmap='gray')
        
        perturb = X_adv-X
        perturb = perturb.cpu().numpy().reshape(28,28)
        perturb = perturb*127.5+127.5
        plt.figure(figsize=(3, 3))
        plt.imshow(perturb, cmap='gray')
        
        plt.show()

def test2(src, goal, s=2.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    R = attack_net.ANGRI()
    victim = Nets.Net1()
    victim.load_state_dict(torch.load('./model/Net1.pth'))
    victim = victim.to(device)
    PATH = './model/ANGRI Net1 w=0.12/epoch_200.pth'
    R.load_state_dict(torch.load(PATH, map_location=device))
    R = R.to(device)

    _, test_iter = dl.load_MNIST(batch_size=512, cut=src)
    TFR, total = 0.0, 0
    with torch.no_grad():
        for X, y in test_iter:
            m = X.shape[0]
            X = X.to(device)
            y = y.to(device)
            R.eval() # 评估模式, 这会关闭dropout
            # t = (torch.floor(10*torch.rand(m, 1))).long()
            t = goal*torch.ones(m, 1).long()
            t_one_hot = torch.zeros(m, 10).scatter_(1,t,1)
            t = t.to(device)
            t_one_hot = t_one_hot.to(device)
            
            X_adv = torch.clamp(X + s*R(t_one_hot, X), min=-1.0, max=1.0)
            y_adv_hat = victim(X_adv)
            TFR += (y_adv_hat.argmax(dim=1) == t.reshape(-1)).float().sum().cpu().item()
            
            index = (y_adv_hat.argmax(dim=1) == t.reshape(-1))
            y_adv_hat_tmp = (F.softmax(y_adv_hat, dim=1))[index]
            y_adv_hat_tmp = torch.max(y_adv_hat_tmp, dim=1)[0]

            R.train() # 改回训练模式
            total += m
    return TFR/total

def test3():
    s = 2.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    R = attack_net.ANGRI()
    victim = Nets.Net1()
    victim.load_state_dict(torch.load('./model/Net1.pth'))
    victim = victim.to(device)
    PATH = './model/ANGRI Net1 w=0.12/epoch_200.pth'
    R.load_state_dict(torch.load(PATH, map_location=device))
    R = R.to(device)

    t_one_hot = (torch.eye(10)).to(device)
    fig, ax = plt.subplots(nrows = 10, ncols = 10, sharex = True, sharey = True)
    #ax = ax.flatten()
    i = 0
    for src in range(0,10):
        _, test_iter = dl.load_MNIST(batch_size=512, cut=src)
        with torch.no_grad():
            for X, _ in test_iter:
                # print(src)
                X = X[0]
                X = torch.cat((X,X,X,X,X,X,X,X,X,X),dim=0)
                X = X.reshape(10,-1,28,28)
                # print(X.shape)
                X = X.to(device)
                R.eval() # 评估模式, 这会关闭dropout
                #t = goal*torch.ones(m, 1).long()
                #t_one_hot = torch.zeros(m, 10).scatter_(1,t,1)
                #t = t.to(device)
                #t_one_hot = t_one_hot.to(device)
                X_adv = torch.clamp(X + s*R(t_one_hot, X), min=-1.0, max=1.0)
                # print(X_adv.shape)
                X_adv = X_adv.cpu().numpy().reshape(-1, 28, 28)
                X_adv = X_adv*127.5+127.5
                for j in range(0, 10):
                    if src != j:
                        ax[src,j].imshow(X_adv[j], cmap='gray')
                    else:
                        ax[src,j].imshow(255*np.ones((28,28)), cmap='gray')
                R.train() # 改回训练模式
                break
    plt.show()

def evaluate_accuracy(data_iter, victim, R, s=2, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    TFR, MR, FS, C, total = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            m = X.shape[0]
            X = X.to(device)
            y = y.to(device)
            R.eval() # 评估模式, 这会关闭dropout
            t = (torch.floor(10*torch.rand(m, 1))).long()
            t_one_hot = torch.zeros(m, 10).scatter_(1,t,1)
            t = t.to(device)
            t_one_hot = t_one_hot.to(device)
            
            X_adv = torch.clamp(X + s*R(t_one_hot, X), min=-1.0, max=1.0)
            perturb = X_adv-X
            FS += (perturb*perturb).sum().cpu().item()
            y_adv_hat = victim(X_adv)
            TFR += (y_adv_hat.argmax(dim=1) == t.reshape(-1)).float().sum().cpu().item()
            MR += ((y_adv_hat.argmax(dim=1) != y)+(y_adv_hat.argmax(dim=1) == t.reshape(-1))).float().sum().cpu().item()
            
            index = (y_adv_hat.argmax(dim=1) == t.reshape(-1))
            y_adv_hat_tmp = (F.softmax(y_adv_hat, dim=1))[index]
            y_adv_hat_tmp = torch.max(y_adv_hat_tmp, dim=1)[0]
            C += y_adv_hat_tmp.sum().cpu().item()

            R.train() # 改回训练模式
            total += m
    return TFR/total, MR/total, FS/total/(28*28), C/TFR
 

if __name__ == '__main__':
    test1()
    #test3()
    '''
    for i in range(0,10):
        for goal in range(0,10):
            if i!=goal:
                print('%.4f'%(test2(i, goal)), end="  ")
            else:
                print('-----', end="  ")
        print()
    '''
                