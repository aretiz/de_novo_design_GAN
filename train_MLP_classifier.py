import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

seed = 2023
torch.manual_seed(seed)
np.random.seed(seed)

# Load your protein sequence embeddings from CSV
data = pd.read_csv("mean_embeddings_esm2_t12.csv")

# Assuming columns 1 to 5120 contain the embeddings, extract them
embeddings = data.iloc[:, 0:480].values 
# print(embeddings.shape)
# Load your labels from CSV (assuming you have a 'label' column)
labels = data['Label1']

# Encode your labels as integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(embeddings, labels, test_size=0.4, random_state=1, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1, stratify=y_temp)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
# print(X_train.shape)

X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
y_val = torch.tensor(y_val, dtype=torch.int64)
y_test = torch.tensor(y_test, dtype=torch.int64)

# Create DataLoader for training, validation, and testing data
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, drop_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(drop_out)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        return out

# Initialize your model
input_size = len(embeddings[0])
# print(input_size)
hidden_size = 128  
num_classes = len(label_encoder.classes_)  
drop_out = 0.2

model = MLP(input_size, hidden_size, num_classes, drop_out)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize variables to keep track of the lowest validation loss and corresponding model weights
lowest_val_loss = float('inf')
best_model_weights = None
# Training loop
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation on the validation set to check for the lowest validation loss
    model.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()

        # Calculate the average validation loss
        avg_val_loss = total_val_loss / len(val_loader)

        # Check if the current model has a lower validation loss than the lowest recorded
        if avg_val_loss < lowest_val_loss:
            lowest_val_loss = avg_val_loss
            best_model_weights = model.state_dict()
            # Save the best model weights
            torch.save(best_model_weights, 'best_model_ESM2.pth')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}')

model.load_state_dict(torch.load('best_model_ESM@.pth'))

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')
