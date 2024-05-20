import torch, os, glob
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import itertools

class myGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p, num_classes):
        super(myGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, h_n = self.gru(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out, h_n

class DNADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class kmersClassifier():
    def __init__(self, input_size=256, hidden_dim=64, batch_size=64, drop_out=0.3, num_classes=2):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.drop_out = drop_out
        self.input_size = input_size

    def one_hot_encode_pad(self, sequences, n):
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
    
    def predict_model(self, dna_seq):
        data = DNADataset(self.one_hot_encode_pad([dna_seq], 4))  # Wrap the single sequence in a list
        data_loader = DataLoader(data, batch_size=1, shuffle=False)  # Use batch_size=1 for single sequence

        self.model = myGRUModel(self.input_size, self.hidden_dim, self.drop_out, self.num_classes)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('./best_model_kmers.pth'))
        
        self.model.eval()

        with torch.no_grad():
            for x in data_loader:
                x = x.to(self.device)
                out, _ = self.model(x)

                # Apply softmax to get class probabilities
                probabilities_batch = torch.nn.functional.softmax(out, dim=1)

                # Extract the predicted probability for class 1
                prediction = probabilities_batch[0, 1].cpu().numpy()  

        return prediction
