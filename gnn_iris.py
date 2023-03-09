import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the graph structure
adj_matrix = np.zeros((X_train.shape[0], X_train.shape[0]))
for i in range(X_train.shape[0]):
    for j in range(i+1, X_train.shape[0]):
        dist = np.linalg.norm(X_train[i] - X_train[j])
        adj_matrix[i][j] = dist
        adj_matrix[j][i] = dist

# Convert the adjacency matrix to a tensor
adj_matrix = torch.FloatTensor(adj_matrix)

# Define the dataset class
class IrisDataset(Dataset):
    def __init__(self, X, y, adj_matrix):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.adj_matrix = adj_matrix

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.adj_matrix[idx]

# Define the GNN model
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, adj_matrix):
        x = self.fc1(x)
        x = self.relu(torch.mm(adj_matrix, x))
        x = self.fc2(x)
        return x

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the hyperparameters
input_dim = X_train.shape[1]
hidden_dim = 16
output_dim = len(np.unique(y_train))
lr = 0.01
epochs = 100

# Create the model, loss function, and optimizer
model = GNN(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Create the dataloaders
train_dataset = IrisDataset(X_train, y_train, adj_matrix)
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
test_dataset = IrisDataset(X_test, y_test, adj_matrix)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

# Train the model
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        x_batch, y_batch, adj_matrix_batch = batch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        adj_matrix_batch = adj_matrix_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch, adj_matrix_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch, adj_matrix_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            adj_matrix_batch = adj_matrix_batch.to(device)
            y_pred = model(x_batch, adj_matrix_batch)
            test_loss = criterion(y_pred, y_batch)
            _, predicted = torch.max(y_pred.data, 1)
            test_acc = (predicted == y_batch).sum().item() / len(y_batch)

print(f'Epoch {epoch+1}/{epochs}:')
print(f'Train Loss: {loss.item():.4f}')
print(f'Test Loss: {test_loss.item():.4f}')
print(f'Test Accuracy: {test_acc:.4f}')

