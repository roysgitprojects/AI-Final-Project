import sklearn
import torch
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.cluster import normalized_mutual_info_score
import seaborn as sns


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        # Inputs to first hidden layer linear transformation
        self.input_ = nn.Linear(21, 16)
        # # Inputs to second hidden layer linear transformation
        # self.hidden1 = nn.Linear(64, 32)
        # self.hidden2 = nn.Linear(32, 16)
        # Output layer, 19 units - one for each odor
        self.output = nn.Linear(16, 9)

        # Define softmax output
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.input_(x)
        x = torch.relu(x)
        # x = self.hidden1(x)
        # x = torch.relu(x)
        # x = self.hidden2(x)
        # x = torch.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


def train(epoch, model, x_train, y_train, optimizer, criterion):
    for e in range(epoch):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    return model


def test(model, x_test, y_test, criterion):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in zip(x_test, y_test):
            output = model(data)
            predict = np.argmax(output)
            if predict == target:
                correct += 1
    all_output = model(x_test)
    test_loss = criterion(all_output, y_test)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(x_test),
        100. * correct / len(x_test)))
    acc = 100. * correct / len(x_test)
    return test_loss, acc


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = acc * 100

    return acc


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def main(df, odor):
    # Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(df, odor, test_size=0.6, random_state=69)

    # Split train into train-val
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=21)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    df = np.array(df)
    odor = np.array(odor)
    all_dataset = ClassifierDataset(torch.from_numpy(df).float(), torch.from_numpy(odor).long())
    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]

    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0007
    all_loader = DataLoader(dataset=all_dataset, batch_size=1)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = NN()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    print("Begin training.")

    for e in tqdm(range(1, EPOCHS + 1)):

        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            y_val_pred_list = []
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                y_val_pred_softmax = torch.log_softmax(y_val_pred, dim=1)
                _, y_val_pred_tags = torch.max(y_val_pred_softmax, dim=1)
                y_val_pred_list.append(y_val_pred_tags.cpu().numpy())

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        y_val_pred_list = [a.squeeze().tolist() for a in y_val_pred_list]

        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        print(
            f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(val_loader):.3f}')

    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})  # Plot the dataframes

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))

    sns.lineplot(data=train_val_acc_df, x="epochs", y="value", hue="variable", ax=axes[0]).set_title(
        'Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable", ax=axes[1]).set_title(
        'Train-Val Loss/Epoch')
    plt.show()
    y_pred_list = []

    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:  # all_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_softmax = torch.log_softmax(y_test_pred, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list))
    sns.heatmap(confusion_matrix_df, annot=True, fmt="d")
    plt.show()
    print(classification_report(y_test, y_pred_list))
    print(f"nmi score: {normalized_mutual_info_score(y_test, y_pred_list)}")

    print("val report:")
    print(classification_report(y_val, y_val_pred_list))

    print(X_test)
    print(y_pred_list)
    print(y_val)
    print(y_val_pred_list)
    print(len(X_test))
    print(len(y_pred_list))
    print(len(y_val))
    print(len(y_val_pred_list))

    return X_test, y_pred_list, y_val, y_val_pred_list
