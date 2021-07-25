import torch.nn as nn

import torch
import time
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from datetime import timedelta

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import normalize

import main_file

data_dim = 21


class AE(nn.Module):
    """
    Autoencoder class
    """

    def __init__(self):
        super(AE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(data_dim, 16),
            nn.Tanh(),
            # nn.Linear(64, 32),
            # nn.Tanh(),
            # nn.Linear(32, 16),
            # nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 3),
            nn.Tanh(),
            # nn.Linear(4, 2),
            # nn.Tanh()
        )
        self.dec = nn.Sequential(
            # nn.Linear(3, 4),
            # nn.Tanh(),
            nn.Linear(3, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
            # nn.Linear(16, 32),
            # nn.Tanh(),
            # nn.Linear(32, 64),
            # nn.Tanh(),
            nn.Linear(16, data_dim),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Encode and decode
        :param x: the data
        :return: encoded and decoded data
        """
        encode = self.enc(x)
        decode = self.dec(encode)
        return encode, decode


# Core training parameters.
batch_size = 1  # 32
lr = 1e-3  # 2  # learning rate
w_d = 1e-5  # weight decay
momentum = 0.9
epochs = 40  # 15


class Loader(torch.utils.data.Dataset):
    """
    Load data
    """

    def __init__(self):
        super(Loader, self).__init__()
        self.dataset = ''

    def __len__(self):
        """
        length of dataset
        :return: length of data set
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        # row = row.drop(labels={'Class'})
        data = torch.from_numpy(np.array(row) / 255).float()
        return data


class Train_Loader(Loader):
    """
    Load training set
    """

    def __init__(self, train_df):
        super(Train_Loader, self).__init__()
        self.dataset = train_df


class Val_Loader(Loader):
    """
    Load validation set
    """

    def __init__(self, val_df):
        super(Val_Loader, self).__init__()
        self.dataset = val_df  # train with 20%


class Test_Loader(Loader):
    """
    Load testing set
    """

    def __init__(self, test_df):
        super(Test_Loader, self).__init__()
        self.dataset = test_df  # main_file.get_dataset()


def main():
    """
    Main. perform autoencoder - 20% training set.
    :return: x,y,z
    """
    df = main_file.get_dataset()
    x_train, x_val = train_test_split(df, train_size=0.2)
    train_set = Train_Loader(x_train)
    val_set = Val_Loader(x_val)
    val_ = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    train_ = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    metrics = defaultdict(list)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AE()
    model.to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=w_d)

    # train the model
    model.train()
    start = time.time()
    for epoch in range(epochs):
        ep_start = time.time()
        running_loss = 0.0
        for bx, (data) in enumerate(train_):
            _, sample = model(data.to(device))
            loss = criterion(data.to(device), sample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # validation
        with torch.no_grad():
            val_loss = 0.0
            model.eval()
            for bx, (data) in enumerate(val_):
                _, sample = model(data.to(device))
                loss = criterion(data.to(device), sample)
                val_loss += loss.item()

        epoch_loss = running_loss / len(train_set)
        metrics['train_loss'].append(epoch_loss)
        epoch_val_loss = val_loss / len(val_set)
        metrics['val_loss'].append(epoch_val_loss)
        ep_end = time.time()
        print('-----------------------------------------------')
        print('[EPOCH] {}/{}\n[Train LOSS] {}  [Validation Loss] {}'.format(epoch + 1, epochs, epoch_loss,
                                                                            epoch_val_loss))
        print('Epoch Complete in {}'.format(timedelta(seconds=ep_end - ep_start)))
    end = time.time()
    print('-----------------------------------------------')
    print('[System Complete: {}]'.format(timedelta(seconds=end - start)))

    # plot whether the model converges
    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_title('Loss')
    ax.plot(metrics['train_loss'], label="Training Loss")
    ax.plot(metrics['val_loss'], label="Validation Loss")
    plt.legend()
    plt.show()

    # prediction
    model.eval()
    loss_dist = []
    test_set = Test_Loader(df)
    test_ = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,  # batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    # try
    encoded_data = []
    with torch.no_grad():
        for bx, data in enumerate(test_):
            encoded_sample, sample = model(data.to(device))
            # print(encoded_sample)
            encoded_data.append(encoded_sample.detach().numpy())
            loss = criterion(data.to(device), sample)
            loss_dist.append(loss.item())
    loss_sc = []
    for i in loss_dist:
        loss_sc.append((i, i))
    plt.scatter(*zip(*loss_sc))
    upper_threshold = np.percentile(loss_dist, 98)
    lower_threshold = 0.0
    plt.axvline(upper_threshold, 0.0, 1)
    plt.show()
    df = pd.DataFrame(main_file.get_dataset())
    anomalies = np.array([0 for _ in range(len(loss_dist))])
    for i in range(len(loss_dist)):
        if loss_dist[i] >= upper_threshold:  # if anomaly
            anomalies[i] = 1
    print(anomalies)
    print('number of anomalies', anomalies.sum(), 'out of ', len(anomalies), 'points')

    df['is anomaly'] = anomalies
    df.to_csv("data with anomalies.csv")

    # points = model.enc(df.to_numpy())  # dimension reduction
    final_encode = []
    with torch.no_grad():
        for step, bx in enumerate(test_set):
            encoded = model.enc(bx.float())
            final_encode.append(encoded.tolist())
    points = final_encode
    x, y, z = zip(*points)
    return x, y, z


if __name__ == '__main__':
    x, y, z = main()
    import csv

    with open("dimension_reduction/ae_3d.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([x, y, z])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.title.set_text("real")
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Third Principal Component')
    ax.scatter3D(x, y, z, c=main_file.get_real_labels(), alpha=0.8, s=8)
    plt.show()
