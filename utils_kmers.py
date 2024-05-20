import itertools
import torch
from torch.utils.data import Dataset
import os
import time
import numpy as np
import glob
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DNADataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_data(dataset):
    data = []
    label = []
    with open(dataset, 'r') as f:
        for line in f:
            seq, l = line.split()
            data += [seq]
            label += [int(l)]
    return data, label

def compute_test(model, loader, device):
    loss_fn = nn.CrossEntropyLoss()

    correct = 0.0
    loss_test = 0.0

    for i, (x, label) in enumerate(loader):
        x = x.to(device)
        label = label.to(device)
        out, h_n = model(x)
        pred = out.max(dim=1)[1]  # get the index of the max log-probability
        correct += pred.eq(label).sum().item()
        loss_test += loss_fn(out, label).mean()

    return correct / len(loader.dataset), loss_test, pred


def train(model, train_loader, val_loader, epochs, lr, patience, device):
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")
    t = time.time()
    model.train()
    model.to(device)

    for epoch in range(epochs):
        loss_train = 0.0
        correct = 0
        for i, (x, label) in enumerate(train_loader):
            x = x.to(device)
            label = label.to(device)
            out, h_n = model(x)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()  # do gradient descent over the batch
            optimizer.zero_grad()  # clear the gradient

            loss_train += loss.item()
            pred = out.max(dim=1)[1]  # get the index of the max log-probability
            correct += pred.eq(label).sum().item()

        acc_train = correct / len(train_loader.dataset)
        # acc_val, loss_val, tmp = compute_test(model, val_loader, device)
        with torch.no_grad():
            acc_val, loss_val, tmp = compute_test(model, val_loader, device)

        print('Epoch: {:04d}'.format(epoch + 1), 'acc_train: {:.6f}'.format(acc_train),
              'loss_train: {:.6f}'.format(loss_train), 'loss_val: {:.6f}'.format(loss_val),
              'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))

        val_loss_values.append(loss_val)

        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_cnt += 1

        if patience_cnt == patience:
            print(epoch + 1)
            break

        if torch.isnan(loss_val):
            print("Validation loss is NaN. Breaking the training loop.")
            loss_val = -1000
            break

    # print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return model, best_epoch

def stratified_split(data, labels, test_size, val_size, random_state):
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size, stratify=labels, random_state=random_state)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=val_size, stratify=train_labels, random_state=random_state)
    return train_data, test_data, val_data, train_labels, test_labels, val_labels

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def one_hot_encode_pad(sequences, n):
    n_words = [''.join(n_word) for n_word in itertools.product('ACGT', repeat=n)]

    # Create a dictionary of n-words and their corresponding one-hot-encoding
    n_word_dict = {n_word: i for i, n_word in enumerate(n_words)}

    # Find the maximum length of all sequences
    max_len = max([len(seq) for seq in sequences])

    all_one_hot = []
    for seq in sequences:
        seq_len = len(seq)
        one_hot = torch.zeros(max_len, len(n_word_dict))

        # Fill in the one-hot-encoding for the given sequence
        for i in range(seq_len-n+1):
            n_word = seq[i:i+n]
            one_hot[i, n_word_dict[n_word]] = 1

        all_one_hot.append(one_hot)

    return all_one_hot


def compute_metrics(model, data_loader, device):
    model.eval()
    model.to(device)

    # initialize counters for metrics
    total = 0
    correct = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            out, h_n = model(x)
            y_pred = out.max(dim=1)[1]  # get the index of the max log-probability
            total += y.size(0)
            correct += (y_pred == y).sum().item()

            for i in range(y.size(0)):
                if y[i] == 1 and y_pred[i] == 1:
                    true_positives += 1
                elif y[i] == 0 and y_pred[i] == 1:
                    false_positives += 1
                elif y[i] == 1 and y_pred[i] == 0:
                    false_negatives += 1
                else:
                    true_negatives += 1

    accuracy = correct / total
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1


def compute_metrics_inference(model, data_loader, device):
    model.eval()
    model.to(device)

    # initialize counters for metrics
    total = 0
    correct = 0
    y_pred_all = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            out, h_n = model(x)
            
            total += y.size(0)
            correct += (out.max(dim=1)[1] == y).sum().item()

            # Append predicted labels to the list
            y_pred_all.extend(out.max(dim=1)[1].tolist())

    # Convert the list to a tensor at the end
    y_pred_all = torch.tensor(y_pred_all, dtype=torch.long, device=device)

    return y_pred_all


def clear_files():
    # Search files with .pth extension in current directory
    pattern = "*.pth"
    files = glob.glob(pattern)

    # deleting the files with txt extension
    for file in files:
        os.remove(file)
    print("All previous .pth files have been removed")


if __name__ == '__main__':
    os.chdir('..')
    pass

