import pandas as pd
from torch._C import dtype
import torch.nn as nn
import numpy as np
import torch
import time
# import sklearn

from sklearn.model_selection import train_test_split
from model import CNNLSTM
from data_process import ts_train_test
from torch.utils.data import DataLoader, Dataset, TensorDataset


def train():
    # 超参数
    ################

    learning_rate = 0.005
    epoch = 20000
    batch_size = 100

    ################
    # training start 
    X_train, y_train, X_test = ts_train_test()
    train_data = torch.tensor(X_train, dtype=torch.float)
    train_label = torch.tensor(y_train, dtype=torch.float)
    test_data = torch.tensor(X_test, dtype=torch.float)

    dataset = TensorDataset(train_data, train_label)
    dataset = DataLoader(dataset=dataset, batch_size=batch_size)

    model = CNNLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    loss_all = []
    for epoch in range(epoch):
        for index, data_ in enumerate(dataset):
            data, label = data_
            if data.shape[0] < 10:
                continue
            model.zero_grad()
            predict = model(data)
            loss = loss_function(predict, label)
            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch}, loss: {loss.item()/batch_size}')
        loss_all.append(loss.item()/batch_size)
    print(model(test_data))
    # print(test_data)

    test_predict = model(test_data)
    loss = np.array(loss_all)

    np.save('../save/loss.npy', loss)
    np.save('../save/test_predict.npy', test_predict.detach().numpy())

    torch.save(model.state_dict(), '../save/model.pth')


def plot(label, predict, flag):
    import matplotlib.pyplot as plt
    t = np.linspace(0, label.shape[0], label.shape[0])
    plt.plot(t, label, 'r', label='True')
    plt.plot(t, predict, 'b', label='Predict')
    plt.title(f'{flag}')
    plt.legend()
    plt.savefig(f'../save/{flag}2.png')


def plots():

    test_label = np.load('../save/test_label.npy')
    test_predict = np.load('../save/test_predict.npy')
    test_label = test_label.squeeze()
    # plot(train_label, predict, 'train')
    plot(test_label, test_predict, 'test')

    print(test_predict)


def plot_loss():
    loss = np.load('../save/loss.npy')
    t = np.linspace(0, loss.shape[0], loss.shape[0])
    import matplotlib.pyplot as plt
    plt.plot(t, loss)
    plt.savefig('../save/loss.png')


if __name__ == '__main__':
    # load_data()
    # train()

    # suan_number()
    # plot_loss()
    plots()
