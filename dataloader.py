# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 20:06:27 2020

@author: Li Jiahao
"""

import numpy as np
import struct
import sys
import random
import torch
import matplotlib.pyplot as plt

train_images_idx3_ubyte_file = './data/MNIST/train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = './data/MNIST/train-labels.idx1-ubyte'
test_images_idx3_ubyte_file = './data/MNIST/t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = './data/MNIST/t10k-labels.idx1-ubyte'

def decode_idx3_ubyte(idex3_ubyte_file):
    bin_data = open(idex3_ubyte_file,'rb').read()
    offset = 0
    fmt_header = '>IIII'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    #print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))
    
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' +str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows,num_cols))
        offset += struct.calcsize(fmt_image)
    images = (images-127.5)/127.5
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>II'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    #print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))
    
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)

def load_MNIST(batch_size, cut):
    train_images = load_train_images()
    train_labels = load_train_labels()
    index = np.where(train_labels != cut)
    train_images = train_images[index]
    train_labels = train_labels[index]
    
    test_images = load_test_images()
    test_labels = load_test_labels()
    if cut>=0 and cut<=9:
        index = np.where(test_labels == cut)
        test_images = test_images[index]
        test_labels = test_labels[index]

    # train_images = (train_images-127.5)/127.5
    # test_images = (test_images-127.5)/127.5
    
    train_images = torch.from_numpy(train_images.reshape(train_images.shape[0],1,train_images.shape[1],train_images.shape[2]))
    test_images = torch.from_numpy(test_images.reshape(test_images.shape[0],1,test_images.shape[1],test_images.shape[2]))
    train_images = train_images.float()
    test_images = test_images.float()
    
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)
    train_labels = train_labels.float()
    test_labels = test_labels.float()
    
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_iter, test_iter

def main():
    load_MNIST(512, 0)
    train_images = load_train_images()
    #train_labels = load_train_labels()
    #print(train_labels.shape)
    #test_images = load_test_images()
    #test_labels = load_test_labels()
    fig, ax = plt.subplots(nrows = 5, ncols = 5, sharex = True, sharey = True)
    ax = ax.flatten()
    for i in range(25):
        ax[i].imshow(train_images[random.randint(0,49999)],cmap='gray')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.show()

if __name__ == '__main__':
    main()