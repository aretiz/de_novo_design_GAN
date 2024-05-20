import torch
import torch.nn as nn
from utils_kmers import*
from torch.utils.data import DataLoader
import numpy as np
import time


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


if __name__ == '__main__':
    seed = 2
    torch.manual_seed(seed)
    np.random.seed(seed)

    # check if gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # load the dataset
    dna_seq, labels = load_data('./data/AMP_dataset.fa')
    
    # data preprocessing
    k = 4
    data = one_hot_encode_pad(dna_seq, k)
    
    # split the data for training, validation and testing
    val_size = 0.5
    test_size = 0.4

    # model parameters
    epochs = 1000
    lr = 1e-3
    dropout = 0.3
    patience_cnt = 60
    hidden_size = 64
    batch_size = 64 
    input_size = 4 ** k
    num_classes = 2

    acc_all = []
    prec_all = []
    rec_all = []
    f1_all = []

    for i in range(1):
        start_time = time.time()

        ############################# I fixed i=2 to take the best_model.pth ##########################
        random_state = 2 
        X_train, X_test, X_val, y_train, y_test, y_val = stratified_split(data, labels, test_size, val_size, random_state)

        # dataloaders
        train_dataset = DNADataset(X_train, y_train)
        val_dataset = DNADataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # define the model
        model = myGRUModel(input_size, hidden_size, dropout, num_classes)
        model.to(device)
        print(count_parameters(model))

        model, best_epoch = train(model, train_loader, val_loader, epochs, lr, patience_cnt, device)
        print("--- Total training time %s seconds ---" % (time.time() - start_time))

        model.load_state_dict(torch.load('best_model_kmers.pth'))

        # predict on the test set
        dataset = DNADataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        accuracy, precision, recall, f1 = compute_metrics(model, test_loader, device)

        # Measure inference time
        # inference_start_time = time.time()
        y_pred = compute_metrics_inference(model, test_loader, device)
        # inference_time = time.time() - inference_start_time

        # print("--- Inference time: %s seconds ---" % inference_time)
       
        print(accuracy)

        acc_all.append(accuracy)
        print(acc_all)
        prec_all.append(precision)
        print(prec_all)
        rec_all.append(recall)
        print(rec_all)
        f1_all.append(f1)
        print(f1_all)

    print(np.mean(acc_all))
    print(np.mean(prec_all))
    print(np.mean(rec_all))
    print(np.mean(f1_all))

    print(np.std(acc_all))
    print(np.std(prec_all))
    print(np.std(rec_all))
    print(np.std(f1_all))
